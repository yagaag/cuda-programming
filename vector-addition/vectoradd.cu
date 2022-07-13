// Cuda programming practice: Vector addition

#include <stdio.h>
#define SIZE 8192

__global__ void VectorAdd(int *a, int *b, int *c, int n)
{
	int i = threadIdx.x;
	if (i<n)
		c[i] = a[i] + b[i];
}

int main()
{
	int *a, *b, *c;
	// Allocate cuda memory
	cudaMallocManaged(&a, SIZE * sizeof(int));
	cudaMallocManaged(&b, SIZE * sizeof(int));
	cudaMallocManaged(&c, SIZE * sizeof(int));
	
	// Initialize the arrays
	for (int i = 0; i < SIZE; ++i)
	{
		a[i] = i;
		b[i] = i;
		c[i] = 0;
	}
	
	// Call addition operation
	VectorAdd <<<1, SIZE>>> (a, b, c, SIZE);

	// Wait for async ops to complete
	cudaDeviceSynchronize();

	for (int i = 0; i < 100; ++i)
		printf("c[%d] = %d\n", i, c[i]);

	// Free GPU memory
	cudaFree(a);
	cudaFree(b);
	cudaFree(c);

	return 0;
}
