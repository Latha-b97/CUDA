#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

void initialize(int* input, const int array_size)
{
	// initialize with random
	srand(time(NULL));
	for (int i = 0; i < array_size; i++)
	{
		input[i] = rand() % 10;
	}
}

//cpu reduction
int reduction_cpu(int* input, const int size)
{
	int sum = 0;
	for (int i = 0; i < size; i++)
	{
		sum += input[i];
	}
	return sum;
}

//compare results
void compare_results(int gpu_result, int cpu_result)
{
	printf("GPU result : %d , CPU result : %d \n", gpu_result, cpu_result);

	if (gpu_result == cpu_result)
	{
		printf("GPU and CPU results are same \n");
		return;
	}

	printf("GPU and CPU results are different \n");
}

__global__ void reduction_neighbored_pairs(
	int* int_array, int* partialsum_array, int size)
{
	int tid = threadIdx.x;
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	// do the boundary check
	if (gid > size)
		return;

	// We are setting the initial offset to 1 for first iteration ( so T0, T2, T4,T6 and T0 = To +T1; T2= T2+T3; T4 =T4+T5; T6=T6+T7)
	// then the next ierations are multiple of 2 (so T0, T4, and T0 = T0 +T2; T4 =T4+T6;) and so on ( see the notes)
	for (int offset = 1; offset <= blockDim.x / 2; offset *= 2)
	{
		int index = 2 * offset * tid;

		if (tid %(2*offset)== 0)
		{
			int_array[gid] += int_array[gid + offset];
		}

		// here all the threads in the block needs to finish atleast one iteration before any thread starts the next iteration
		__syncthreads();
	}

	// after all the iteration are done, the first element in the block has the summation of the block, assign that block to partial sum array
	if (tid == 0)
	{
		partialsum_array[blockIdx.x] = int_array[gid];
	}
}



__global__ void reduction_neighbored_pairs_improved(
	int* int_array, int* temp_array, int size)
{
	int tid = threadIdx.x;
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	//local data block pointer
	int* i_data = int_array + blockDim.x * blockIdx.x;

	if (gid > size)
		return;

	for (int offset = 1; offset <= blockDim.x / 2; offset *= 2)
	{
		int index = 2 * offset * tid;

		if (index < blockDim.x)
		{
			i_data[index] += i_data[index + offset];
		}

		__syncthreads();
	}

	if (tid == 0)
	{
		temp_array[blockIdx.x] = int_array[gid];
	}
}

int main(int argc, char ** argv)
{
	printf("Running parallel reduction with neighbored pairs kernel \n");

	int size = 1 << 27; // 128Mb of data
	int byte_size = size * sizeof(int);
	int block_size = 128;

	int * h_input, *h_ref;
	h_input = (int*)malloc(byte_size);

 	initialize(h_input, size); // initialize with random

	int cpu_result = reduction_cpu(h_input, size);

	dim3 block(block_size);
	dim3 grid(size / block.x);

	printf("Kernel launch parameters || grid : %d, block : %d \n", grid.x, block.x);

	// allocate memory for partial sum array
	int temp_array_byte_size = sizeof(int)* grid.x;
	h_ref = (int*)malloc(temp_array_byte_size);

	// allocate memory for device pointers
	int * d_input, *d_temp;
	gpuErrchk(cudaMalloc((void**)&d_input, byte_size));
	gpuErrchk(cudaMalloc((void**)&d_temp, temp_array_byte_size));

	// set the initial values to zero
	gpuErrchk(cudaMemset(d_temp, 0, temp_array_byte_size));
	// transfer the input array from host to device 
	gpuErrchk(cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice));


	reduction_neighbored_pairs << < grid, block >> > (d_input, d_temp, size);
	//reduction_neighbored_pairs_improved << < grid, block >> > (d_input, d_temp, size);


	gpuErrchk(cudaDeviceSynchronize());
	// transfer the partial sum array from  device to host
	gpuErrchk(cudaMemcpy(h_ref, d_temp, temp_array_byte_size, cudaMemcpyDeviceToHost));

	// To get the final result, we need to iterate through the partial sum array and add it to global varible
	int gpu_result = 0;
	for (int i = 0; i < grid.x; i++)
	{
		gpu_result += h_ref[i];
	}

	//compare the results
	compare_results(gpu_result, cpu_result);

	gpuErrchk(cudaFree(d_input));
	gpuErrchk(cudaFree(d_temp));
	free(h_input);
	free(h_ref);

	gpuErrchk(cudaDeviceReset());
	return 0;
}