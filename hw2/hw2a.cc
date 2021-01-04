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
#include <time.h>
#include <pmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <omp.h>
#include <math.h>


using namespace std;

pthread_mutex_t lock;
pthread_mutex_t print_lock;
double mutWait = 0;

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

// calculate a pixel repeat
void cal_each_pixel_repeats(int *image, const int k, const double x0, const double y0, const int iters)
{
    // calculate
    int repeats = 0;
    double x = 0;
    double y = 0;
    double length_squared = 0;

    do{
        double temp = x * x - y * y + x0;
        y = 2 * x * y + y0;
        x = temp;
        length_squared = x * x + y * y;
        ++repeats;
    }while(repeats < iters && length_squared < 4);

    image[k] = repeats;
}

// after calculate two pixels , run this function in order to check pixels already statify conditions
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

// vectorize version of cal_each_pixel_repeats
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
    
    // save result for run continue_each_pixel_repeats
    _mm_store_pd(x_res, x_vec);
    _mm_store_pd(y_res, y_vec);

    // save repeats
    long long repeats[2];
    _mm_store_si128((__m128i*)&repeats, repeats_vec);
    image[k[0]] = (int)repeats[0];
    image[k[1]] = (int)repeats[1];
}


// data structure for threads
typedef struct
{
    // get info from main
    int *k_ptr; // share variable
    int color_channel;

    // input data
    int iters;
    double left;
    double right;
    double lower;
    double upper;
    int width;
    int height;

    // image info
    int *image_repeats;
    png_bytep image;
} Thread_data;

int checkValue(int value){
    if (value < 20){
        return 20;
    }
    else{
        return value;
    }
}

void *thread_process(void *thread_data)
{
    struct timespec threadStart, threadEnd;
    clock_gettime(CLOCK_MONOTONIC, &threadStart);

    Thread_data *thread = (Thread_data *)thread_data;

    // info from main
    int *k_ptr = thread->k_ptr;
    int color_channel = thread->color_channel;

    int iters = thread->iters;
    double left = thread->left;
    double right = thread->right;
    double lower = thread->lower;
    double upper = thread->upper;
    int width = thread->width;
    int height = thread->height;

    // image info
    int *image_repeats = thread->image_repeats;
    png_bytep image = thread->image;

    // preprocess info
    int k_start, k_size;
    int k[2]; // the vectorization index
    size_t pixel_size = color_channel * sizeof(png_byte);
    size_t row_size = width * pixel_size;
    int pixels = width * height;

    // thread calculate their pixels
    while(1)
    {
        // get two pixel to calculate
        double start = omp_get_wtime();
        pthread_mutex_lock(&lock);

        double end = omp_get_wtime();
        mutWait += (end - start);
        k_start = (*k_ptr);
        k_size = checkValue((int)ceil(log(pixels - k_start)));
        // k_size = 20;
        (*k_ptr) += k_size;
        pthread_mutex_unlock(&lock);
        // cout << "start " << k_start << endl;
        // cout << "size " << k_size << endl;

        for(int l = 0; l < k_size; l = l + 2){
            k[0] = k_start + l;
            k[1] = k_start + l+1;
             // all out of idx
            if(k[0] >= pixels){
                pthread_exit(NULL);
            }
            // only one idx need to cal
            else if(k[1] >= pixels){
                int j = k[0] / width;
                int i = k[0] % width;
                double y0 = j * ((upper - lower) / height) + lower;
                double x0 = i * ((right - left) / width) + left;
                cal_each_pixel_repeats(image_repeats, k[0], x0, y0, iters);

                png_bytep row = image + (height - 1 - j) * row_size;
                int p = image_repeats[k[0]];
                png_bytep color = row + i * color_channel;
                if (p != iters)
                {
                    if (p & 16)
                    {
                        color[0] = 240;
                        color[1] = color[2] = p % 16 * 16;
                    }
                    else
                    {
                        color[0] = p % 16 * 16;
                    }
                }
                pthread_exit(NULL);
            }
            // cal two pixels
            else{
                int j[2], i[2];
                double x0[2], y0[2];
                for(int x = 0; x < 2; ++x){
                    j[x] = k[x] / width;
                    i[x] = k[x] % width;
                    y0[x] = j[x] * ((upper - lower) / height) + lower;
                    x0[x] = i[x] * ((right - left) / width) + left;
                }

                double x_res[2], y_res[2];
                cal_two_pixels_repeats(image_repeats, k, x0, y0, iters, x_res, y_res);

                for(int x = 0; x < 2; ++x){
                    continue_each_pixel_repeats(image_repeats, k[x], x0[x], y0[x], iters, x_res[x], y_res[x]);
                }
                
                // calculate color
                for(int x = 0; x < 2; ++x){
                    png_bytep row = image + (height - 1 - j[x]) * row_size;
                    int p = image_repeats[k[x]];
                    png_bytep color = row + i[x] * color_channel;
                    if (p != iters)
                    {
                        if (p & 16)
                        {
                            color[0] = 240;
                            color[1] = color[2] = p % 16 * 16;
                        }
                        else
                        {
                            color[0] = p % 16 * 16;
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char **argv){
    // calculate run time
    clock_gettime(CLOCK_MONOTONIC, &start);

    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int ncpus = CPU_COUNT(&cpu_set);
    printf("%d cpus available\n", ncpus);

    /* argument parsing */
    assert(argc == 9);
    const char *filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    // default setting
    int color_channel = 3; // RGB or WB
    int row_size = color_channel * width * sizeof(png_byte);

    // Init mutex lock
    pthread_mutex_init(&lock, NULL);
    pthread_mutex_init(&print_lock, NULL);

    /* allocate memory for image */
    int *image_repeats = (int *)malloc(width * height * sizeof(int));
    png_bytep image = (png_bytep)malloc(height * row_size);
    memset(image, 0, height * row_size);
    png_bytep row_pointers[height]; // for write io
    assert(image_repeats);
    assert(image);
    assert(row_pointers);

    // info about thread data
    pthread_t threads[ncpus];
    Thread_data thread_task[ncpus];

    // cal thread task size
    int pixels = height * width;

    // share variable
    int share_k = 0;

    // create thread to calculate mandelbrot set
    for (int cpu = 0; cpu < ncpus; ++cpu)
    {
        // info for each process
        thread_task[cpu].k_ptr = &share_k;
        thread_task[cpu].color_channel = 3;

        // input info
        thread_task[cpu].iters = iters;
        thread_task[cpu].left = left;
        thread_task[cpu].right = right;
        thread_task[cpu].lower = lower;
        thread_task[cpu].upper = upper;
        thread_task[cpu].width = width;
        thread_task[cpu].height = height;

        // result buffer
        thread_task[cpu].image_repeats = image_repeats;
        thread_task[cpu].image = image;

        pthread_create(&threads[cpu], NULL, thread_process, (void *)&thread_task[cpu]);
    }

    // get image row start position
    for (int k = 0; k < height; k++)
    {
        row_pointers[k] = image + k * row_size;
    }

    // wait thread
    for (int cpu = 0; cpu < ncpus; ++cpu)
    {
        pthread_join(threads[cpu], NULL);
    }

    // set write png info
    FILE *fp = fopen(filename, "wb");
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

    clock_gettime(CLOCK_MONOTONIC, &End);
    cout << "cal each process " << diff(start, End) << " s" << endl;
    

    // write data to disk
    clock_gettime(CLOCK_MONOTONIC, &start);
    png_write_image(png_ptr, row_pointers);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);

    clock_gettime(CLOCK_MONOTONIC, &End);
    cout << "in write img IO " << diff(start, End) << " s" << endl;


    // release memory
    pthread_mutex_destroy(&lock);
    free(image_repeats);
    free(image);
    cout << "mutex wait time " << mutWait << endl;
    return 0;
}
