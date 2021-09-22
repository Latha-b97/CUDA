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
 1.This is an example to generate pictures using GPU computing power. It uses OpenGL.
 2. Pass function pointer "generate_frame" to bitmap.anim_and_exit function. generate_frame gets 
 called everytime a new frame gets called.
 3. See the notes in generate_frame
    // blocks represnets number of blocks launched in our Grid
    // threads represensts number of threads in each block
    // because we generate an image the threads are two dimensional to represent each pixel. each (x,y) index 
    // corresponds to a pixel in the image
 4. // If the image is DIM X DIM, if we want each pixel to be represented by   a thread, then we need DIM/16, DIM/16 blocks
 5. // In the kernel function, pass in GPU device memory to hold the output pixels and ticks is for animation time
 */

#include "cuda.h"
#include "cpu_anim.h"

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



#define DIM 1024
#define PI 3.1415926535897932f

__global__ void kernel(unsigned char* ptr, int ticks) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    // now calculate the value at that position
    float fx = x - DIM / 2;
    float fy = y - DIM / 2;
    float d = sqrtf(fx * fx + fy * fy);
    unsigned char grey = (unsigned char)(128.0f + 127.0f *
        cos(d / 10.0f - ticks / 7.0f) /
        (d / 10.0f + 1.0f));
    ptr[offset * 4 + 0] = grey;
    ptr[offset * 4 + 1] = grey;
    ptr[offset * 4 + 2] = grey;
    ptr[offset * 4 + 3] = 255;
}

struct DataBlock {
    unsigned char* dev_bitmap;
    CPUAnimBitmap* bitmap;
};

void generate_frame(DataBlock* d, int ticks) 
{
    // blocks represnets number of blocks launched in our Grid
    // threads represensts number of threads in each block
    // because we generate an image the threads are two dimensional to represent each pixel. each (x,y) index corresponds to 
    // a pixel in the image

    dim3    blocks(DIM / 16, DIM / 16);
    dim3    threads(16, 16);
    kernel << <blocks, threads >> > (d->dev_bitmap, ticks); // pass in GPU device memory to hold the output, ticks is for animation time

    HANDLE_ERROR(cudaMemcpy(d->bitmap->get_ptr(),
        d->dev_bitmap,
        d->bitmap->image_size(),
        cudaMemcpyDeviceToHost));
}

// clean up memory allocated on the GPU
void cleanup(DataBlock* d) {
    HANDLE_ERROR(cudaFree(d->dev_bitmap));
}

int main(void) {
    DataBlock   data;
    CPUAnimBitmap  bitmap(DIM, DIM, &data);
    data.bitmap = &bitmap;

    HANDLE_ERROR(cudaMalloc((void**)&data.dev_bitmap,
        bitmap.image_size()));

    bitmap.anim_and_exit((void (*)(void*, int))generate_frame,
        (void (*)(void*))cleanup);
}
