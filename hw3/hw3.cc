#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <iostream>
#include <string.h>
#include <memory>

using namespace std;

#define INF 1073741823

void printArray(int* a, int size){
    for(int j=0; j<size;++j){
        for(int i=0; i<size;++i){
            printf("%d ",a[ j*size + i]);
        }
        printf("\n");
    }	
}

double IOinput = 0;
double IOout = 0;
double comptime = 0;


int main(int argc, char **argv){

    int N, E;
    int *W;
    FILE *fp;

    double ioStart, ioEnd;

    ioStart = omp_get_wtime();
    fp = fopen(argv[1], "rb");
    fread(&N, sizeof(int), 1, fp);
    fread(&E, sizeof(int), 1, fp);
    ioEnd = omp_get_wtime();
    IOinput += (ioEnd - ioStart);

    W = new int[N * N];
    uninitialized_fill_n(W, N*N, INF);
    // init 
    #pragma omp parallel for
    for(int j=0; j<N;++j){
        W[j * N + j] = 0;
    }

    ioStart = omp_get_wtime();
    for(int i = 0; i < E; ++i){
        int src, dst, weight;
        fread(&src, sizeof(int), 1, fp);
        fread(&dst, sizeof(int), 1, fp);
        fread(&weight, sizeof(int), 1, fp);
        W[src * N + dst] = weight;
    }
    fclose(fp);
    ioEnd = omp_get_wtime();
    IOinput += (ioEnd - ioStart);

    int threads = omp_get_max_threads();
    // cout << "threads " << threads <<endl;
    double *thread_run = new double[threads];
    memset(thread_run, 0, threads * sizeof(double));

    double compStart = omp_get_wtime();
    for(int k = 0; k < N; ++k){
        #pragma omp parallel for schedule(dynamic)
        for(int i = 0; i < N; ++i){
            double threadStart = omp_get_wtime();
            int idx = i * N + k;
            int temp = k * N;
                
            for(int j = 0; j < N; ++j){
                if(W[temp + j] == INF)
                    continue;
                int dis = W[idx] + W[temp + j];
                int target_idx =  i * N + j;
                if (dis < W[target_idx]){
                    W[target_idx] = dis;
                }
            }
            double threadEnd = omp_get_wtime();
            int threadId = omp_get_thread_num();
            thread_run[threadId] += (threadEnd - threadStart);
        }
    }
    double compEnd = omp_get_wtime();
    comptime += (compEnd - compStart);

    ioStart = omp_get_wtime();
    fp = fopen(argv[2], "wb");
    fwrite(W, N * N, sizeof(int), fp);
    fclose(fp);
    ioEnd = omp_get_wtime();
    IOout += (ioEnd - ioStart);


    cout << "input time " << IOinput << endl;
    cout << "output time " << IOout << endl;
    cout << "comp time " << comptime << endl;

    for(int i = 0; i < threads; ++i){
        cout << i << " run " << thread_run[i] << endl;
    }

    delete [] W;
}