#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//cpu transpose
void mat_transpose_cpu(int* mat, int* transpose, int nx, int ny)
{
	for (int iy = 0; iy < ny; iy++)
	{
		for (int ix = 0; ix < nx; ix++)
		{
			transpose[ix * ny + iy] = mat[iy * nx + ix];
		}
	}
}

//compare arrays
void compare_arrays(int* a, int* b, int size)
{
	for (int i = 0; i < size; i++)
	{
		if (a[i] != b[i])
		{
			printf("Arrays are different \n");
			printf("%d - %d | %d \n", i, a[i], b[i]);
		}
	}
	printf("Arrays are same \n");
}

void initialize(int* input, const int array_size)
{
	for (int i = 0; i < array_size; i++)
	{
		input[i] = i % 10;
	}
}

// Read the elements in row major order(coalesed) and write in stride way
// Read from input matrix in coalesed manner and write to output matrix in stride manner
__global__ void transposematrix_read_rowmajorFormat_write_stride(int* input, int* output, const int nx, const int ny)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        output[ix * ny + iy] = input[iy * nx + ix];
    }
}

// Read the elements in stride order and write in coalesed way
// from input matrix in  stride manner and write to output matrix in coalesed manner
__global__ void transposematrix_read_stride_write_rowmajorFormat(int* input, int* output, const int nx, const int ny)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        output[iy * nx + ix] = input[ix * ny + iy];
    }
}

int main(int argc, char** argv)
{
	//default values for variabless
	int nx = 1024;
	int ny = 1024;
	int block_x = 128;
	int block_y = 8;
	int kernel_num = 0;

	if (argc > 1)
		kernel_num = atoi(argv[1]);

	int size = nx * ny;
	int byte_size = sizeof(int*) * size;

	printf("Matrix transpose for %d X % d matrix with block size %d X %d \n",nx,ny,block_x,block_y);

	int * h_mat_array = (int*)malloc(byte_size);
	int * h_trans_array = (int*)malloc(byte_size);
	int * h_ref = (int*)malloc(byte_size);

	//initialize matrix with integers between one and ten
	initialize(h_mat_array,size);

	//matirx transpose in CPU
	mat_transpose_cpu(h_mat_array, h_trans_array, nx, ny);

	int * d_mat_array, *d_trans_array;
	
	cudaMalloc((void**)&d_mat_array, byte_size);
	cudaMalloc((void**)&d_trans_array, byte_size);

	cudaMemcpy(d_mat_array, h_mat_array, byte_size, cudaMemcpyHostToDevice);

	//Each threadblock is 128*8, and so 1024/128 threadblocks in x dirction and 1024/8 thread blocks in y dimension
	dim3 blocks(block_x, block_y);
	dim3 grid(nx/block_x, ny/block_y);


	//transposematrix_read_rowmajorFormat_write_stride <<< grid, blocks>> > (d_mat_array, d_trans_array,nx, ny);
	transposematrix_read_stride_write_rowmajorFormat << < grid, blocks >> > (d_mat_array, d_trans_array, nx, ny);

	cudaDeviceSynchronize();

	//copy the transpose memroy back to cpu; h_ref is the GPU out
	cudaMemcpy(h_ref, d_trans_array, byte_size, cudaMemcpyDeviceToHost);

	//compare the CPU and GPU transpose matrix for validity
	compare_arrays(h_ref, h_trans_array, size);

	cudaDeviceReset();
	return EXIT_SUCCESS;
}



