#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define B 64
#define THREAD 32 

const int INF = ((1 << 30) - 1);

int n, m;
int total_round;
int padding;

static int *Dist;
int* dev_Dist[2];
int num_gpu;


size_t pitch;
size_t row_size;

int ceil(int a, int b) { return (a + b - 1) / b; }



void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    total_round = ceil(n, B);
    padding = B * total_round;

    cudaMallocHost((void **)&Dist, padding*padding*sizeof(int));

    #pragma omp parallel for
    for (int i = 0; i < padding; ++i) {
        for (int j = 0; j < padding; ++j) {
            if (i == j) {
                Dist[i*padding + j] = 0;
            } else {
                Dist[i*padding + j] = INF;
            }
        }
    }

    int* buffer = (int *)malloc(3 * m * sizeof(int));
    fread(buffer, sizeof(int), 3*m, file);
    
    #pragma omp parallel for
    for (int i = 0; i < m; ++i) {
        Dist[buffer[i * 3]*padding + buffer[i *3 + 1]] = buffer[i * 3 + 2];
    }
    free(buffer);
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    // for (int i = 0; i < n; ++i) {
    //     fwrite(&Dist[i*padding], sizeof(int), n, outfile);
    // }
    fwrite(&Dist[0], sizeof(int), n*n, outfile);
    fclose(outfile);
}


__global__ void phase_1(int *dev_Dist, int Round, int total_round, int row_size){
    __shared__ int myself[B][B];
    int tid_x = threadIdx.y;
    int tid_y = threadIdx.x;

    // my block info
    int b_i = Round;
    int b_j = Round;

    int block_internal_start_x = b_i * B;
    int block_internal_start_y = b_j * B;


    // move myself to share mem
    #pragma unroll
    for(int i=0; i<B; i+=THREAD){
        for(int j=0; j<B; j+=THREAD){
            myself[tid_x+i][tid_y+j] = dev_Dist[(block_internal_start_x+tid_x+i)*row_size+(block_internal_start_y+tid_y+j)];
        }
    }
    __syncthreads();

    

    // calculate
    for (int k = 0; k<B; ++k) {
        #pragma unroll
        for(int i=0; i<B; i+=THREAD){
            for(int j=0; j<B; j+=THREAD){
                myself[tid_x+i][tid_y+j] = min(myself[tid_x+i][tid_y+j], myself[tid_x+i][k] + myself[k][tid_y+j]);
            }
        }
        __syncthreads();
    }

    // restore data to global mem
    #pragma unroll
    for(int i=0; i<B; i+=THREAD){
        for(int j=0; j<B; j+=THREAD){
            dev_Dist[(block_internal_start_x+tid_x+i)*row_size+(block_internal_start_y+tid_y+j)] = myself[tid_x+i][tid_y+j];
        }
    }
}

__global__ void phase_2(int* dev_Dist,const int Round, const int total_round, int row_size) {
    __shared__ int myself[B][B];
    __shared__ int pivot[B][B];
    int tid_x = threadIdx.y;
    int tid_y = threadIdx.x;

    int b_i = -999;
    int b_j = -999;

    // get block idx
    if(blockIdx.x == 0){ // 同row 
        b_i = Round;
        b_j = (Round+1+blockIdx.y) % total_round;
    }
    if(blockIdx.x ==1){ // 同col
        b_i =  (Round+1+blockIdx.y) % total_round;
        b_j = Round;
    }

    // pivot info
    int pivot_block_internal_start_x = Round * B;
    int pivot_block_internal_start_y = Round * B;

    
    // move pivot to share
    #pragma unroll
    for(int i=0; i < B; i+=THREAD){
        for(int j=0; j<B; j+=THREAD){
            pivot[tid_x+i][tid_y+j] = dev_Dist[(pivot_block_internal_start_x+tid_x+i)*row_size + (pivot_block_internal_start_y+tid_y+j)];
        }
    }


    // my block info
    int block_internal_start_x = b_i * B;
    int block_internal_start_y = b_j * B;

    // move my block to share mem
    #pragma unroll
    for(int i=0; i<B; i+=THREAD){
        for(int j=0; j<B; j+=THREAD){
            myself[tid_x+i][tid_y+j] = dev_Dist[(block_internal_start_x+tid_x+i)*row_size+(block_internal_start_y+tid_y+j)];
        }
    }
    __syncthreads();

    // calculate
    if(b_j == Round){ // 同col
        #pragma unroll
        for (int k = 0; k < B; ++k) {
            #pragma unroll
            for(int i=0; i < B; i+=THREAD){
                for(int j=0; j<B; j+=THREAD){
                    myself[tid_x+i][tid_y+j] = min(myself[tid_x+i][tid_y+j], myself[tid_x+i][k] + pivot[k][tid_y+j]);
                }
            }
        }
    }
    else{
        #pragma unroll
        for (int k = 0; k < B; ++k) {
            #pragma unroll
            for(int i=0; i < B; i+=THREAD){
                for(int j=0; j<B; j+=THREAD){
                    myself[tid_x+i][tid_y+j] = min(myself[tid_x+i][tid_y+j], pivot[tid_x+i][k] + myself[k][tid_y+j]);
                }
            }
        }
    }
    
    
    // move data to global mem
    #pragma unroll
    for(int i=0; i<B; i+=THREAD){
        for(int j=0; j<B; j+=THREAD){
            dev_Dist[(block_internal_start_x+tid_x+i)*row_size+(block_internal_start_y+tid_y+j)] = myself[tid_x+i][tid_y+j];
        }
    }
}


__global__ void phase_3(int* dev_Dist, int Round,int total_round, int row_size, int start_x) {

    __shared__ int myself[B][B];
    __shared__ int same_row[B][B];
    __shared__ int same_col[B][B];
    int tid_x = threadIdx.y;
    int tid_y = threadIdx.x;
    

    // get internal block idx
    int b_i = start_x+blockIdx.x;
    int b_j = (Round+1+blockIdx.y) % total_round;
    int block_internal_start_x = b_i * B;
    int block_internal_start_y = b_j * B;

    if(b_i == Round) return;

    // need block
    int same_row_block_internal_start_x =  b_i * B;
    int same_row_block_internal_start_y =  Round * B;
    int same_col_block_internal_start_x =  Round * B;
    int same_col_block_internal_start_y =  b_j * B;


    // move same row to share mem
    #pragma unroll
    for(int i=0; i<B; i+=THREAD){
        for(int j=0; j<B; j+=THREAD){
            same_row[tid_x + i][tid_y + j] = dev_Dist[(same_row_block_internal_start_x+tid_x+i)*row_size+(same_row_block_internal_start_y+tid_y+j)];
        }
    }

    // move same col to share mem
    #pragma unroll
    for(int i=0; i<B; i+=THREAD){
        for(int j=0; j<B; j+=THREAD){
            same_col[tid_x+i][tid_y+j] = dev_Dist[(same_col_block_internal_start_x+tid_x+i)*row_size+(same_col_block_internal_start_y+tid_y+j)];
        }
    }

    // move myself block to share mem
    #pragma unroll
    for(int i=0; i<B; i+=THREAD){
        for(int j=0; j<B; j+=THREAD){
            myself[tid_x+i][tid_y+j] = dev_Dist[(block_internal_start_x+tid_x+i)*row_size+(block_internal_start_y+tid_y+j)];
        }
    }
    __syncthreads();
    
    // cal
    #pragma unroll
    for (int k = 0; k<B; ++k) {
        #pragma unroll
        for(int i=0; i < B; i+=THREAD){
            for(int j=0; j<B; j+=THREAD){
                myself[tid_x+i][tid_y+j] = min(myself[tid_x+i][tid_y+j], same_row[tid_x+i][k] + same_col[k][tid_y+j]);
                // printf("Round:%d k:%d i:%d j:%d myself:%d\n", Round, k, block_internal_start_x+tid_x+i, block_internal_start_y+tid_y+j, myself[tid_x+i][tid_y+j]);            
            }
        }
    }

    // restore to global memory
    #pragma unroll
    for(int i=0; i < B; i+=THREAD){
        for(int j=0; j<B; j+=THREAD){
            dev_Dist[(block_internal_start_x+tid_x+i)*row_size + (block_internal_start_y+tid_y+j)] = myself[tid_x+i][tid_y+j];
        }
    }
}

// __global__ void print(int** dev_Dist, int thread_id, int average, int row_size,int height){
//     for(int i=thread_id*average; i<height; ++i){
//         for(int j=0; j< padding;++j){
//             printf("%d ", dev_Dist[thread_id][i*row_size+j]);
//         }
//         printf("\n");
//     }
//     printf("\n");
// }

void block_FW(){
    // cudaError_t err;
    
    
    dim3 threads(THREAD ,THREAD);

    
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        cudaSetDevice(thread_id);
        
        int average = ceil(total_round, num_gpu);

        dim3 p1(1 ,1);
        dim3 p2(2, total_round-1);
        dim3 p3(average, total_round-1);
        if(thread_id == (num_gpu-1)){
            p3.x = total_round - average*(num_gpu-1);
        }

        cudaError_t err;
        for (int r = 0; r < total_round; ++r) {
            if(r != 0){
                // printf("tid:%d round:%d thread_id*average:%d (thread_id+1)*average):%d\n", thread_id, r, thread_id*average, (thread_id+1)*average);
                if(((thread_id*average)<=r) && (r<((thread_id+1)*average))){
                    err = cudaMemcpy2D(dev_Dist[(thread_id+1)%num_gpu]+r*B*row_size, pitch, dev_Dist[thread_id]+r*B*row_size, pitch, padding * sizeof(int), B, cudaMemcpyDeviceToDevice);
                    if(err != cudaSuccess){
                        printf("thread:%d copy dev_Dist error", thread_id);
                    }
                }
            }
            #pragma omp barrier
            /* Phase 1*/
            phase_1<<< p1, threads>>>(dev_Dist[thread_id], r, total_round, row_size);
            err = cudaGetLastError();
            if ( err != cudaSuccess )
            {
                printf("phase 1 r:%d, CUDA Error: %s\n",r , cudaGetErrorString(err));       
            }

            /* Phase 2*/
            phase_2<<< p2, threads>>>(dev_Dist[thread_id], r, total_round, row_size);
            err = cudaGetLastError();
            if ( err != cudaSuccess )
            {
                printf("phase 2 r:%d, CUDA Error: %s\n", r, cudaGetErrorString(err));       
            }

            /* Phase 3*/
            phase_3<<< p3, threads>>>(dev_Dist[thread_id], r, total_round, row_size, thread_id*average);
            // printf("tid:%d phase3 start x: %d, height:%d\n", thread_id,thread_id*average, p3.x);
            err = cudaGetLastError();
            if ( err != cudaSuccess )
            {
                printf("phase 3 r:%d, CUDA Error: %s\n", r,cudaGetErrorString(err));       
            }

            // omp_set_lock(&writelock);
            // print<<<1,1>>>(dev_Dist, thread_id, average, padding, p3.x);
            // omp_unset_lock(&writelock);
            #pragma omp barrier
        }
        if(thread_id !=0){
            cudaMemcpy2D(dev_Dist[0]+thread_id*average*B*row_size, pitch ,dev_Dist[thread_id]+thread_id*average*B*row_size, pitch, padding*sizeof(int), p3.x*B, cudaMemcpyDeviceToDevice);
        }
    }   
}


int main(int argc, char* argv[]) {

    cudaGetDeviceCount(&num_gpu);
    omp_set_num_threads(num_gpu);

    // double start = omp_get_wtime();
    input(argv[1]);
    // double end = omp_get_wtime();
    // printf("read %fs\n", end-start);

    #pragma omp parallel
    {
        cudaError_t err;
        int thread_id = omp_get_thread_num();
        // printf("thread_id: %d\n", thread_id);
        cudaSetDevice(thread_id);
        cudaMallocPitch((void **)&dev_Dist[thread_id], &pitch, padding*sizeof(int), padding);
        err = cudaMemcpy2D(dev_Dist[thread_id], pitch, Dist, padding * sizeof(int), padding * sizeof(int), padding, cudaMemcpyHostToDevice);
        if(err != cudaSuccess){
            printf("thread:%d copy dev_Dist error", thread_id);
        }
        for(int i=0; i<num_gpu; ++i){
            if(i != thread_id){
                cudaDeviceEnablePeerAccess(i, 0);
            }
        }
    }
    row_size = pitch/sizeof(int);
    

    // start = omp_get_wtime();
    block_FW();
    // end = omp_get_wtime();
    // printf("cal %fs\n", end-start);

    
    // start = omp_get_wtime();
    cudaMemcpy2D(Dist, n*sizeof(int), dev_Dist[0], pitch, n * sizeof(int), n, cudaMemcpyDeviceToHost);
    // end = omp_get_wtime();
    // printf("D2H %fs\n", end-start);


    // start = omp_get_wtime();
    output(argv[2]);
    // end = omp_get_wtime();
    // printf("output %fs\n", end-start);

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        cudaFree(dev_Dist[thread_id]);
    }
    cudaFreeHost(Dist);
    return 0;
}