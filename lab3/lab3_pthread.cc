#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <iostream>

using namespace std;

typedef struct info{
//	unsigned long long x;
	unsigned long long k;
	unsigned long long r;
	unsigned long long task;
	unsigned long long result;
	int threadID;
	int *t;
}Cal_info;

void test(int i){
	printf("hello this is %d\n", i);
}

void *cal_func(void *info){
	Cal_info* ptr = (Cal_info *)info;
	unsigned long long pixels = 0;
	unsigned long long r = ptr->r;
	unsigned long long k = ptr->k;
	unsigned long long task = ptr->task;
	int rank = ptr->threadID;
	ptr->t[rank] = rank+10;
	
	for ( unsigned long long x = rank*task; x<(rank+1)*task && x<r; x++){
		unsigned long long y = ceil(sqrtl(r*r - x*x));
		pixels += y;
	}

	pixels %= k;
	ptr->result = pixels;
	test(rank);
//	printf("threadID: %d, pixels:%d\n", rank, pixels);
	return (void *)0;
}



int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;

	cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	unsigned long long ncpus = CPU_COUNT(&cpuset);
	ncpus += 1;
	
	unsigned long long task = r / ncpus;
	if ( r%ncpus != 0 ){
		++task;
	}
	Cal_info data[ncpus];
	pthread_t threads[ncpus];
	// int *t = new int[ncpus];
	int *t = (int *)malloc(ncpus * sizeof(int));

	// creat child thread calculate
	for (unsigned long long x = 1; x < ncpus; x++) {
		data[x].task = task;
		data[x].threadID = x;
		data[x].r = r;
		data[x].k = k;
		data[x].t = t;
		pthread_create(&threads[x], NULL, cal_func, (void *)&data[x]);
	}

	// main thread calculate
	int rank = 0;
	for ( unsigned long long x = rank*task; x<(rank+1)*task && x<r; x++){
		unsigned long long y = ceil(sqrtl(r*r - x*x));
		pixels += y;
//		printf("threadID: %d, pixels:%d\n", rank, pixels);	
	}

	pixels %= k;
	// wait all child thread
	for( unsigned long long x = 1; x < ncpus; x++){
		pthread_join(threads[x], NULL);
		cout << "rank  " << x << " "<< data[x].t[x] << endl;
	}


	// cummculate sum
	for(int x=1; x<ncpus; ++x){
		pixels += data[x].result;
//		pixels %= k;
	}
	pixels %= k;
	printf("%llu\n", (4 * pixels) % k);
}
