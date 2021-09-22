
// My Notes: This is a simple Raytracing example from Cuda by Examples Book 
// This is a simple raytracer and camera is restricted to only z-axis facing origin, no lighting effects as well. Instead it will assign a sphere a color and shade them with precomputed function.
// The RayTracer will fire a ray from each pixel and keep track of which rays hit which sphere.When ray passes through multiple spheres, only the sphere closest to the camera can be seen.


#include "cuda.h"
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bitmap.h"

#define DIM 1024

#define rnd( x ) (x * rand() / RAND_MAX)
#define INF     2e10f

void CHECKERROR(cudaError_t error)
{
	if (error != cudaSuccess)
	{
		printf("Error : %s : %d, ", __FILE__, __LINE__);
		printf("code : %d, reason: %s \n", error, cudaGetErrorString(error));
		exit(1);
	}
}

struct Sphere {
    float   r, b, g;
    float   radius;
    float   x, y, z;

    //this method computes whether the ray intersects the sphere, the method computes the distance from the camera wher the ray hits the sphere.
    // Given a ray shot from the pixel at (ox,oy)this method computes the distance from the camera where the ray hits the sphere. If the ray hit more than one sphere, only the closest will be seen
    __device__ float hit(float ox, float oy, float* n) {
        float dx = ox - x;
        float dy = oy - y;
        if (dx * dx + dy * dy < radius * radius) {
            float dz = sqrtf(radius * radius - dx * dx - dy * dy);
            *n = dz / sqrtf(radius * radius);
            return dz + z;
        }
        return -INF;
    }
};
#define SPHERES 20

__constant__ Sphere s[SPHERES];

__global__ void kernel(unsigned char* ptr) 
{

    // Each thread is generating one pixel for our output image, so we compute x and y coordinates for the thread ans linearlized offset into output buffer.
    // We will also shift our(x, y) image coordinates by DIM / 2 so that the z-axis runs through the center of the image.
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    float   ox = (x - DIM / 2);
    float   oy = (y - DIM / 2);

    float   r = 0, g = 0, b = 0;
    float   maxz = -INF;

    // Here we iterate through each of the inout sphere and call its hit() to see whether the ray fron our pixels see the sphere.
    // If the ray hit the current sphere, we determine whether the hit is closer to the camers than the last sphere we hit.If it is closer, we store the depth of the closest sphere.
    // In addiiton we store the color associated with this sphere so that when the loop has terminated, the thread knows the color of the sphere that is closest to camera
    for (int i = 0; i < SPHERES; i++) {
        float   n;
        float   t = s[i].hit(ox, oy, &n);
        if (t > maxz) {
            float fscale = n;
            r = s[i].r * fscale;
            g = s[i].g * fscale;
            b = s[i].b * fscale;
            maxz = t;
        }
    }

    ptr[offset * 4 + 0] = (int)(r * 255);
    ptr[offset * 4 + 1] = (int)(g * 255);
    ptr[offset * 4 + 2] = (int)(b * 255);
    ptr[offset * 4 + 3] = 255;
}

// globals needed by the update routine
struct DataBlock {
    unsigned char* dev_bitmap;
};

int main(void) {
    DataBlock   data;
    // capture the start time
    cudaEvent_t     start, stop;
    CHECKERROR(cudaEventCreate(&start));
    CHECKERROR(cudaEventCreate(&stop));
    CHECKERROR(cudaEventRecord(start, 0));

    Bitmap bitmap(DIM, DIM, &data);
    unsigned char* dev_bitmap;

    // allocate memory on the GPU for the output bitmap. It will be filled with output pixel data as we trace our spheres on the GPU.
    CHECKERROR(cudaMalloc((void**)&dev_bitmap,
        bitmap.image_size()));

    // allocate temp memory, initialize it, copy to constant
    // memory on the GPU, then free our temp memory.
    // Here we randomly generate the center, coordinates, color and radius of the spheres.
    Sphere* temp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);
    for (int i = 0; i < SPHERES; i++) {
        temp_s[i].r = rnd(1.0f);
        temp_s[i].g = rnd(1.0f);
        temp_s[i].b = rnd(1.0f);
        temp_s[i].x = rnd(1000.0f) - 500;
        temp_s[i].y = rnd(1000.0f) - 500;
        temp_s[i].z = rnd(1000.0f) - 500;
        temp_s[i].radius = rnd(100.0f) + 20;
    }

    // This will generate a random array of 20 spheres, specified by SPHERE
    CHECKERROR(cudaMemcpyToSymbol(s, temp_s,
        sizeof(Sphere) * SPHERES));
    free(temp_s);

    // generate a bitmap from our sphere data.  Here the ray traces and generates pixel datafor the inout scene of spheres.
    dim3    grids(DIM / 16, DIM / 16);
    dim3    threads(16, 16);
    kernel << <grids, threads >> > (dev_bitmap);


    // copy our bitmap back from the GPU for display
    CHECKERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap,
        bitmap.image_size(),
        cudaMemcpyDeviceToHost));



    // get stop time, and display the timing results
    CHECKERROR(cudaEventRecord(stop, 0));
    CHECKERROR(cudaEventSynchronize(stop));
    float   elapsedTime;
    CHECKERROR(cudaEventElapsedTime(&elapsedTime,
        start, stop));
    printf("Time to generate:  %3.1f ms\n", elapsedTime);

    CHECKERROR(cudaEventDestroy(start));
    CHECKERROR(cudaEventDestroy(stop));

    CHECKERROR(cudaFree(dev_bitmap));

    // display
    bitmap.display_and_exit();
}

