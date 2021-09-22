#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include "common.h"


 //My Notes :
 //1. Sahred memory is onchip memory, so it has less latency, but shared memory is very limited.
 //2. A fixed amount of shared memory is allocated to each thread block and it is ahred by all the threads in the thread block/
 //The life time of shared memory is same as the life time of thread block.
 //3.Each request to access Shared memory by a warp is given as one transaction.In worst case, if 32 threads are requested to access shared memory, it will be executed in 32 transactions.
 //4.Shared memory is mainly used when same set of memory neeeds to be accessed again and again ( it is programmed managed cache)
 //5.So instead of locading repeatedly accessed global memory multiple times from DRAM, we can store them in shared memory space and access it from there.


 //This program shows how we allocate shared and dynamically
 //This program shows first to load it from global memory and then put them back to global meory.

/*
*	To execute
	D:\> SharedMemory_Static_and_Dynamic.exe 1
	Dynamic shared memory kernel

	D:\> SharedMemory_Static_and_Dynamic.exe
	Static shared memory kernel
*/


void initialize(int* input, const int array_size)
{
	// Initialize with value 1 to 10
	for (int i = 0; i < array_size; i++)
	{
		input[i] = i % 10;
	}
}

#define SHARED_ARRAY_SIZE 128

__global__ void shared_mem_static_test(int* in, int* out, int size)
{
	// this takes in two gloabl array pointers in and out

	// shared memory access happens with in threadblock
	int tid = threadIdx.x;
	// to access global memeory, we need gid
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	// This access memory from shared memory, not from gloabl memory
	__shared__ int shared_mem[SHARED_ARRAY_SIZE];

	// Each thread block which execute this code will transfer shared memory of SHARED_ARRAY_SIZE = 128 size
	if (gid < size)
	{
		shared_mem[tid] = in[gid];
		out[gid] = shared_mem[tid];
	}
}

__global__ void shared_mem_dynamic_test(int* in, int* out, int size)
{
	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	// Dynamic shared memeory is similar to static,but it tells the kernel that memory allocation is happening somewhere else.
	extern __shared__ int shared_mem[];

	if (gid < size)
	{
		shared_mem[tid] = in[gid];
		out[gid] = shared_mem[tid];
	}
}

int main(int argc, char ** argv)
{
	int size = 1 << 22;
	int block_size = SHARED_ARRAY_SIZE;
	bool dynamic = false;

	if (argc > 1)
	{
		dynamic = atoi(argv[1]);
	}

	size_t NO_BYTES = size * sizeof(int);

	int *h_in, *h_ref, *d_in, *d_out;

	h_in = (int *)malloc(NO_BYTES);
	h_ref = (int *)malloc(NO_BYTES);

	initialize(h_in, size);

	cudaMalloc((int **)&d_in, NO_BYTES);
	cudaMalloc((int **)&d_out, NO_BYTES);

	//kernel launch parameters
	dim3 block(block_size);
	dim3 grid((size / block.x) + 1);

	cudaMemcpy(d_in, h_in, NO_BYTES, cudaMemcpyHostToDevice);

	if (!dynamic)
	{
		printf("Static shared memory kernel \n");
		shared_mem_static_test << <grid, block >> > (d_in, d_out, size);
	}
	else
	{
		printf("Dynamic shared memory kernel \n");
		// Here we are allocating the dynamic memory rather than at the kernel
		shared_mem_dynamic_test << <grid, block, sizeof(int)*  SHARED_ARRAY_SIZE >> > (d_in, d_out, size);
	}
	cudaDeviceSynchronize();

	cudaMemcpy(h_ref, d_out, NO_BYTES, cudaMemcpyDeviceToHost);

	cudaFree(d_in);
	cudaFree(d_out);

	free(h_in);
	free(h_ref);

	cudaDeviceReset();
	return EXIT_SUCCESS;
}