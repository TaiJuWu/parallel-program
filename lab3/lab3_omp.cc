#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	#pragma omp parallel  for schedule(guided,100000) reduction(+:pixels)
	for (unsigned long long x = 0; x < r; x++) {
		unsigned long long y = ceil(sqrtl(r*r - x*x));
		pixels += y;
//		pixels %= k;
	}
	pixels %= k;
	printf("%llu\n", (4 * pixels) % k);
}
