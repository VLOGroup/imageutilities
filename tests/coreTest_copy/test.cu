#ifdef WIN32
// necessary for Intellisense
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

#include "../../src/iucore.h"

// Device code
__global__ void test_kernel(iu::ImageGpu_32f_C1::KernelData img)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x < img.width_ && y < img.height_)
    {
        img(x,y) = x+y;
    }
}

// Interface functions

namespace cuda
{

void test(iu::ImageGpu_32f_C1& img)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid(iu::divUp(img.width(), dimBlock.x), img.height(), dimBlock.y);

    test_kernel <<< dimGrid, dimBlock >>> (img);
}

}
