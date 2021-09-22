
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__global__ void sum_array_gpu(int* a, int* b, int* c, int size)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < size)
    {
        c[gid] = a[gid] + b[gid];
    }

}

int main()
{

    int size = 10000;
    int block_size = 128;

    int NO_BYTES = size * sizeof(int);

    int* h_a, * h_b, * gpu_results;

    h_a = (int*)malloc(NO_BYTES);
    h_b = (int*)malloc(NO_BYTES);
    gpu_results = (int*)malloc(NO_BYTES);

    time_t t;
    // we are assining random sequence of numbers
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++)
    {
        h_a[i] = (int)rand() & (0xff); // assigning random values between 0 and 255 to each element
    }

    for (int i = 0; i < size; i++)
    {
        h_b[i] = (int)rand() & (0xff); // assigning random values between 0 and 255 to each element
    }

    memset(gpu_results, 0, NO_BYTES);


    // now we need to allocate memory in the device.
    int* d_a, * d_b, * d_c;
    cudaMalloc((void**)&d_a, NO_BYTES);
    cudaMalloc((void**)&d_b, NO_BYTES);
    cudaMalloc((void**)&d_c, NO_BYTES);

    // we need to 
    cudaMemcpy(d_a, h_a, NO_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, NO_BYTES, cudaMemcpyHostToDevice);


    dim3 block(block_size);
    // our array size is 10000, but our block size is 128, we are going to have 10000 threads, it is not fully divided by 128 blocksize. 
    // in these scenario, we add 1 as gridsize. This one will guarentees that we are going to have more threads than array size.
    dim3 grid((size / block.x) + 1);

    // launch kernel
    sum_array_gpu << < grid, block >> > (d_a, d_b, d_c, size);

    // cuda device synchronize, wait for the host to finish
    cudaDeviceSynchronize();

    // the results will be in device memory, we need to transfer that to host
    // gpu_results is the destination CPU pointer
    cudaMemcpy(gpu_results, d_c, NO_BYTES, cudaMemcpyHostToDevice);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(gpu_results);
    free(h_a);
    free(h_b);

    return 0;
}



