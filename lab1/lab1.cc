#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "must provide exactly 2 arguments!\n");
    return 1;
  }
  int rank, size;
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  unsigned long long r = atoll(argv[1]);
  unsigned long long k = atoll(argv[2]);
  unsigned long long task = r / size;
  if (r % size != 0)
    task += 1;
  //	unsigned long long task = ceil(float(r)/size);

  //	if(rank == 0){
  //		printf("task is %d\n", task);
  //		printf("r is %d\n", r);
  //		printf("k is %d\n",k);
  //	}

  unsigned long long pixels = 0;
  for (unsigned long long x = rank * task; x < (rank + 1) * task && x < r;
       x++) {
    unsigned long long y = ceil(sqrtl(r * r - x * x));
    //		printf("rank is %d, x is %d, y is %d\n", rank, x, y);
    pixels += y;
    pixels %= k;
  }
  unsigned long long global_pixels;

  MPI_Barrier(MPI_COMM_WORLD);
  //	printf("rank is %d ,pixels is %llu\n", rank,pixels);
  MPI_Reduce(&pixels, &global_pixels, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0,
             MPI_COMM_WORLD);
  //	printf("rank is %d, global is %llu\n", rank,global_pixels);

  if (rank == 0) {
    //		printf("task is %llu\n", task);
    printf("%llu\n", (4 * global_pixels) % k);
  }

  MPI_Finalize();
  //	printf("size of long long is %d\n", sizeof(long long));
}
