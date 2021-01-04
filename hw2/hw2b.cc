#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <pthread.h>
#include <iostream>
#include <unistd.h> // 測試用
#include <time.h>
#include <mpi.h>
#include <omp.h>
#include <pmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>

using namespace std;

double compStart, compEnd;
double commStart, commEnd;
double IOStart, IO_end;


struct timespec start, End, temp;

double diff(struct timespec start, struct timespec end){
    if((end.tv_nsec - start.tv_nsec) < 0){
        temp.tv_sec = end.tv_sec - start.tv_sec - 1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    }
    else{
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    double time_used = temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;
    return time_used;
}


void cal_image_color(int iters, int width, int height, int color_channel, int* repeats, png_bytep image,const int k_start, const int k_end) {
    
    size_t row_size = color_channel * width * sizeof(png_byte);

    #pragma omp parallel for 
    for(int k = k_start; k < k_end; ++k){
        // cout << "rank "<< rank << "cal image color" << endl;
        // cout << "thread " << rank << " k " << k << endl; 
        int y = k / width;
        int x = k % width;

        png_bytep row = image + y * row_size;

        int p = repeats[k];
        png_bytep color = row + x * 3;
        if (p != iters) {
            if (p & 16) {
                color[0] = 240;
                color[1] = color[2] = p % 16 * 16;
            } else {
                color[0] = p % 16 * 16;
            }
        }
    }
}


void continue_each_pixel_repeats(int *image, const int k, const double x0, const double y0, const int iters, const double x_res, const double y_res)
{
    // calculate
    int repeats = image[k];
    double x = x_res;
    double y = y_res;
    double length_squared = x_res * x_res + y_res * y_res;

    while (repeats < iters && length_squared < 4)
    {
        double temp = x * x - y * y + x0;
        y = 2 * x * y + y0;
        x = temp;
        length_squared = x * x + y * y;
        ++repeats;
    }

    image[k] = repeats;
}



void cal_two_pixels_repeats(int *image, const int* k, const double* x0, const double* y0, const int iters, double* x_res, double* y_res){

    __m128d x0_vec = _mm_load_pd(x0);
    __m128d y0_vec = _mm_load_pd(y0);
    __m128i iters_vec = _mm_set_epi64x(iters, iters);

    __m128i repeats_vec = _mm_setzero_si128();;
    __m128d x_vec = _mm_setzero_pd();
    __m128d y_vec = _mm_setzero_pd();
    __m128d len_square_vec = _mm_setzero_pd();


    __m128i repeat_comp;
    __m128d len_squa_comp;
    __m128i one_vec = _mm_set_epi64x(1, 1);
    __m128d two_vec = _mm_set_pd(2, 2);
    __m128d four_vec = _mm_set_pd(4 ,4);

    do{
        // double temp = x * x - y * y + x0;
        __m128d  x_square_vec = _mm_mul_pd(x_vec, x_vec);
        __m128d y_square_vec = _mm_mul_pd(y_vec, y_vec);
        __m128d temp_vec = _mm_sub_pd(x_square_vec, y_square_vec);
        temp_vec = _mm_add_pd(temp_vec ,x0_vec) ;

        // y = 2 * x * y + y0;
        __m128d double_x_mul_y = _mm_mul_pd(x_vec, y_vec);
        double_x_mul_y = _mm_mul_pd( double_x_mul_y, two_vec);
        y_vec = _mm_add_pd(double_x_mul_y, y0_vec);

        // x = temp;
        x_vec = temp_vec;

        // length_squared = x * x + y * y;
        x_square_vec = _mm_mul_pd(x_vec, x_vec);
        y_square_vec = _mm_mul_pd(y_vec, y_vec);
        len_square_vec = _mm_add_pd(x_square_vec, y_square_vec);

        // ++repeats;
        repeats_vec = _mm_add_epi64(repeats_vec, one_vec);

        // repeats < iters
        repeat_comp = _mm_cmpgt_epi64(iters_vec, repeats_vec);

        // length_squared < 4
        len_squa_comp = _mm_cmple_pd(len_square_vec, four_vec);

    }while(_mm_test_all_ones(repeat_comp) && _mm_test_all_ones(_mm_castpd_si128(len_squa_comp)));
    
    // cout << "store x,y res" << endl;
    _mm_store_pd(x_res, x_vec);
    _mm_store_pd(y_res, y_vec);

    // save repeats
    long long repeats[2];
    
    _mm_store_si128((__m128i*)&repeats, repeats_vec);
    image[k[0]] = (int)repeats[0];
    image[k[1]] = (int)repeats[1];
}




void write_png(const char* filename, png_bytep image, int width, int height, int color_channel) {

    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    

    int row_size = color_channel * width * sizeof(png_byte);
    png_bytep row_pointers[height];
    #pragma omp parallel for 
    for (int k = 0; k < height; k++){
        row_pointers[k] = image + (height - 1 - k) * row_size;
    }

    // write data to disk
    png_write_image(png_ptr, row_pointers);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);    
}



void cal_each_pixel_repeats(int* image,const int k, const double x0, const double y0, const int iters){
    // calculate
    int repeats = 0;
    double x = 0;
    double y = 0;
    double length_squared = 0;

    while (repeats < iters && length_squared < 4) {
        double temp = x * x - y * y + x0;
        y = 2 * x * y + y0;
        x = temp;
        length_squared = x * x + y * y;
        ++repeats;
    }
    
    image[k] = repeats;
}


void cal_repeats(int iters, double left, double right, double lower, double upper, int width, int height, int* image, const int k_start, const int k_end){
    
    int size = (k_end - k_start) / 2;
    #pragma omp parallel for schedule(dynamic)
    for(int idx = 0; idx < size; ++idx){
        int k[2] = {k_start + 2*idx, k_start + 2*idx + 1};
        int j[2], i[2];
        double x0[2], y0[2];
        for(int x = 0; x < 2; ++x){
            j[x] = k[x] / width;
            i[x] = k[x] % width;
            y0[x] = j[x] * ((upper - lower) / height) + lower;
            x0[x] = i[x] * ((right - left) / width) + left;
        }

        double x_res[2], y_res[2];
        cal_two_pixels_repeats(image, k, x0, y0, iters, x_res, y_res);

        for(int x = 0; x < 2; ++x){
            continue_each_pixel_repeats(image, k[x], x0[x], y0[x], iters, x_res[x], y_res[x]);
        }
    }

    if(size % 2 == 1){
        int j = (k_end - 1) / width;
        int i = (k_end - 1) % width;
        double y0 = j * ((upper - lower) / height) + lower;
        double x0 = i * ((right - left) / width) + left;
        cal_each_pixel_repeats(image, k_end-1, x0, y0, iters);
    }
    
}

void reduceImage(png_bytep image, png_bytep collect, const int pixels,const int color_channel, const int avg_task ,const int size, const int k_start, const int k_end){
    int disp[size];
    int recvCnt[size];
    int pixel_size = color_channel * sizeof(png_byte);
    #pragma omp parallel for
    for(int i = 0; i< size; ++i){
        disp[i] = i * avg_task * pixel_size;
        if (i == size-1){
            recvCnt[i] = (pixels - (avg_task * (size - 1))) * pixel_size;
        }
        else{
            recvCnt[i] =  avg_task * pixel_size;
        }
    }
    MPI_Gatherv(image + k_start * color_channel, (k_end - k_start) * pixel_size, MPI_UNSIGNED_CHAR, collect, recvCnt, disp, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
}



int main(int argc, char** argv) {

    // MPI init
    int rank, size;
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Barrier(MPI_COMM_WORLD);
    compStart = MPI_Wtime();
    clock_gettime(CLOCK_MONOTONIC, &start);


    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);


    // basic info for image
    int color_channel = 3; // RGB or Black
    size_t row_size = color_channel * width * sizeof(png_byte);
    int pixel_size = color_channel * sizeof(png_byte);

    /* allocate memory for image */
    int* repeats = (int*)malloc(width * height * sizeof(int));
    png_bytep collect = (png_bytep)malloc(height * row_size);
    assert(repeats);
    assert(collect);
   

    // calculate thread task size
    int pixels = height * width;
    int avg_task = pixels / size;
    if(avg_task % size != 0){
        avg_task += 1;
    }
    
    // for each process calculate their idx that be calculated
    int k_start = rank * avg_task;
    int k_end;
    if(rank == (size-1)){
        k_end = pixels;
    }
    else{
        k_end = (rank + 1) * avg_task;
    }
    

    
    cal_repeats(iters, left, right, lower, upper, width, height, repeats, k_start, k_end);

    png_bytep image =  (png_bytep)malloc( height * row_size );
    cal_image_color(iters, width, height, color_channel, repeats, image, k_start, k_end);
    clock_gettime(CLOCK_MONOTONIC, &End);
    MPI_Barrier(MPI_COMM_WORLD);
    compEnd = MPI_Wtime();
    cout << "rank " << rank << "load " << diff(start, End) << " s" << endl;


    // colect data in each process
    MPI_Barrier(MPI_COMM_WORLD);
    commStart = MPI_Wtime();
    reduceImage(image, collect, pixels, color_channel,avg_task, size, k_start, k_end);
    commEnd = MPI_Wtime();
    free(image);

    /* draw and cleanup */
    // write image to disk
    if(rank == 0){
        IOStart = MPI_Wtime();
        write_png(filename, collect, width, height, color_channel);
        IO_end = MPI_Wtime();
        cout << "compute time " << compEnd - compStart << endl;
        cout << "comucation time " << commEnd - commStart << endl;
        cout << "IO time " << IO_end - IOStart << endl;
    }

    free(repeats);
    free(collect);

    MPI_Finalize();
}
