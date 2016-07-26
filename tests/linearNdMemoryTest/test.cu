#ifdef WIN32
// necessary for Intellisense
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

#include "../../src/iucore.h"

// Device code
__global__ void test_kernel(iu::LinearDeviceMemory<float, 1>::KernelData img)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;

    if (x < img.size_[0])
    {
        img(x) = img.size_[0];
    }
}

__global__ void test_kernel(iu::LinearDeviceMemory<float, 2>::KernelData img)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x < img.size_[0] && y < img.size_[1])
    {
        img(x,y) = img.size_[0]*img.size_[1];
    }
}

// Interface functions

namespace cuda
{

void test(iu::LinearDeviceMemory<float, 1>& img)
{
    dim3 dimBlock(16, 1);
    dim3 dimGrid(iu::divUp(img.size()[0], dimBlock.x));

    test_kernel <<< dimGrid, dimBlock >>> (img);
}

void test(iu::LinearDeviceMemory<float, 2>& img)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid(iu::divUp(img.size()[0], dimBlock.x),
                 iu::divUp(img.size()[1], dimBlock.y));

    test_kernel <<< dimGrid, dimBlock >>> (img);
}

}
