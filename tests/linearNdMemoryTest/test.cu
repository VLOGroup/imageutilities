#ifdef WIN32
// necessary for Intellisense
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

#include "../../src/iucore.h"

// Device code
__global__ void test_kernel(iu::LinearDeviceMemory<float, 1>::KernelData img)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x < img.size_[0])
  {
    img(x) = x;
  }
}

__global__ void test_kernel(iu::LinearDeviceMemory<float, 2>::KernelData img)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < img.size_[0] && y < img.size_[1])
  {
    img(x, y) = img.getLinearIndex(x, y);
  }
}

__global__ void test_kernel(iu::LinearDeviceMemory<float, 4>::KernelData img)
{
  unsigned int lin_idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int x, y, z, t;
  img.getPosition(lin_idx, x, y, z, t);
  if (x < img.size_[0] && y < img.size_[1] && z < img.size_[2]
      && t < img.size_[3])
  {
    img(x, y, z, t) = img.getLinearIndex(x, y, z, t);
  }
}

__global__ void test_kernel2(iu::LinearDeviceMemory<float, 4>::KernelData img)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
  unsigned int c = z / img.size_[2];
  unsigned int n = z % img.size_[2];

  if (x < img.size_[0] && y < img.size_[1] && c < img.size_[2]
      && n < img.size_[3])
  {
    img(x, y, c, n) = img.getLinearIndex(x, y, c, n);
  }
}

// Interface functions

namespace cuda {

void test(iu::LinearDeviceMemory<float, 1>& img)
{
  dim3 dimBlock(16, 1);
  dim3 dimGrid(iu::divUp(img.size()[0], dimBlock.x));

  test_kernel<<<dimGrid, dimBlock>>>(img);
  IU_CUDA_CHECK;
}

void test(iu::LinearDeviceMemory<float, 2>& img)
{
  dim3 dimBlock(16, 16);
  dim3 dimGrid(iu::divUp(img.size()[0], dimBlock.x),
               iu::divUp(img.size()[1], dimBlock.y));

  test_kernel<<<dimGrid, dimBlock>>>(img);
  IU_CUDA_CHECK;
}

void test(iu::LinearDeviceMemory<float, 4>& img)
{
  dim3 dimBlock(16, 1);
  dim3 dimGrid(iu::divUp(img.numel(), dimBlock.x));

  test_kernel<<<dimGrid, dimBlock>>>(img);
  IU_CUDA_CHECK;
}

void test2(iu::LinearDeviceMemory<float, 4>& img)
{
  dim3 dimBlock(16, 8, 8);
  dim3 dimGrid(iu::divUp(img.size()[0], dimBlock.x),
               iu::divUp(img.size()[1], dimBlock.y),
               iu::divUp(img.size()[2] * img.size()[3], dimBlock.z));

  test_kernel2<<<dimGrid, dimBlock>>>(img);
  IU_CUDA_CHECK;}
}
