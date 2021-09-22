/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */

/*
My Notes:
Reference Book: Cuda By Example

*/


#include "cuda.h"

#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

static void HandleError(cudaError_t err,
    const char* file,
    int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
            file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define imin(a,b) (a<b?a:b)

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid =
imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);


__global__ void dot(float* a, float* b, float* c) 
{
    // this shared buffer will store the running sum.
    // specify threadsPerBlock, so that each threas in the block has a place to store temporary result
    __shared__ float cache[threadsPerBlock];

    //The offset into our shared memory cache is our thread index
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    // Each thread computes a running sum of product of corresponding entries in a and b.
    // after reaching the end of the array, each thread stores its temporary sum to shared buffer
    float   temp = 0;
    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    // set the cache values
    cache[cacheIndex] = temp;

    // Here all the threads wait until all the threads finished its execution as there are some threads needs to access the stored values, we don't want them to read before the threads are completed
    // synchronize threads in this block
    __syncthreads();

    // we have now the temporary cache, we can sum the values in them. That produces smaller array called reduction
    // Here we take two values in cache and store the result in cache. Since it combines two entries, we start with only half entries and next step we do the remaining half.
    // for reductions, threadsPerBlock must be a power of 2
    // because of the following code
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    // At the end the result will be in the beginning entry.
    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}


int main(void) {
    float* a, * b, c, * partial_c;
    float* dev_a, * dev_b, * dev_partial_c;

    // allocate memory on the cpu side
    a = (float*)malloc(N * sizeof(float));
    b = (float*)malloc(N * sizeof(float));
    partial_c = (float*)malloc(blocksPerGrid * sizeof(float));

    // allocate the memory on the GPU
    HANDLE_ERROR(cudaMalloc((void**)&dev_a,
        N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b,
        N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c,
        blocksPerGrid * sizeof(float)));

    // fill in the host memory with data
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // copy the arrays 'a' and 'b' to the GPU
    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(float),
        cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(float),
        cudaMemcpyHostToDevice));

    dot << <blocksPerGrid, threadsPerBlock >> > (dev_a, dev_b,
        dev_partial_c);

    // copy the array 'c' back from the GPU to the CPU
    HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c,
        blocksPerGrid * sizeof(float),
        cudaMemcpyDeviceToHost));

    // finish up on the CPU side
    c = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        c += partial_c[i];
    }

#define sum_squares(x)  (x*(x+1)*(2*x+1)/6)
    printf("Does GPU value %.6g = %.6g?\n", c,
        2 * sum_squares((float)(N - 1)));

    // free memory on the gpu side
    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_partial_c));

    // free memory on the cpu side
    free(a);
    free(b);
    free(partial_c);
}
