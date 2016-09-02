#include "../iudefs.h"

template<typename PixelType, class Allocator>
__global__ void ifftshift2_kernel(
    typename iu::ImageGpu<PixelType, Allocator>::KernelData src,
    typename iu::ImageGpu<PixelType, Allocator>::KernelData dst)
{
  unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

  unsigned int x_mid = (src.width_ + 1.f) / 2.f;
  unsigned int y_mid = (src.height_ + 1.f) / 2.f;

  if (x < src.width_ && y < src.height_)
  {
    unsigned int x_dst = (x + x_mid) % src.width_;
    unsigned int y_dst = (y + y_mid) % src.height_;

    dst(x_dst, y_dst) = src(x, y);
  }
}

template<typename PixelType, class Allocator>
__global__ void ifftshift2_kernel(
    typename iu::VolumeGpu<PixelType, Allocator>::KernelData src,
    typename iu::VolumeGpu<PixelType, Allocator>::KernelData dst)
{
  unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int z = threadIdx.z + blockIdx.z * blockDim.z;

  unsigned int x_mid = (src.width_ + 1.f) / 2.f;
  unsigned int y_mid = (src.height_ + 1.f) / 2.f;

  if (x < src.width_ && y < src.height_ && z < src.depth_)
  {
    unsigned int x_dst = (x + x_mid) % src.width_;
    unsigned int y_dst = (y + y_mid) % src.height_;

    dst(x_dst, y_dst, z) = src(x, y, z);
  }
}

template<typename PixelType>
__global__ void ifftshift2_kernel(
    typename iu::LinearDeviceMemory<PixelType, 2>::KernelData src,
    typename iu::LinearDeviceMemory<PixelType, 2>::KernelData dst)
{
  unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

  unsigned int x_mid = (src.size_[0] + 1.f) / 2.f;
  unsigned int y_mid = (src.size_[1] + 1.f) / 2.f;

  if (x < src.size_[0] && y < src.size_[1])
  {
    unsigned int x_dst = (x + x_mid) % src.size_[0];
    unsigned int y_dst = (y + y_mid) % src.size_[1];

    dst(x_dst, y_dst) = src(x, y);
  }
}

template<typename PixelType, unsigned int Ndim>
__global__ void ifftshift2_kernel(
    typename iu::LinearDeviceMemory<PixelType, Ndim>::KernelData src,
    typename iu::LinearDeviceMemory<PixelType, Ndim>::KernelData dst)
{
  unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int z = threadIdx.z + blockIdx.z * blockDim.z;
  unsigned int batch = src.numel_ / (src.size_[0] * src.size_[1]);

  unsigned int x_mid = (src.size_[0] + 1.f) / 2.f;
  unsigned int y_mid = (src.size_[1] + 1.f) / 2.f;

  if (x < src.size_[0] && y < src.size_[1] && z < batch)
  {
    unsigned int x_dst = (x + x_mid) % src.size_[0];
    unsigned int y_dst = (y + y_mid) % src.size_[1];

    dst(x_dst, y_dst, z) = src(x, y, z);
  }
}

template<typename PixelType, class Allocator>
__global__ void fftshift2_kernel(
    typename iu::ImageGpu<PixelType, Allocator>::KernelData src,
    typename iu::ImageGpu<PixelType, Allocator>::KernelData dst)
{
  unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

  int x_mid = src.width_ / 2.f;
  int y_mid = src.height_ / 2.f;

  if (x < src.width_ && y < src.height_)
  {
    unsigned int x_dst = (x + x_mid) % src.width_;
    unsigned int y_dst = (y + y_mid) % src.height_;
    dst(x_dst, y_dst) = src(x, y);
  }
}

template<typename PixelType, class Allocator>
__global__ void fftshift2_kernel(
    typename iu::VolumeGpu<PixelType, Allocator>::KernelData src,
    typename iu::VolumeGpu<PixelType, Allocator>::KernelData dst)
{
  unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int z = threadIdx.z + blockIdx.z * blockDim.z;

  int x_mid = src.width_ / 2.f;
  int y_mid = src.height_ / 2.f;

  if (x < src.width_ && y < src.height_ && z < src.depth_)
  {
    unsigned int x_dst = (x + x_mid) % src.width_;
    unsigned int y_dst = (y + y_mid) % src.height_;
    dst(x_dst, y_dst, z) = src(x, y, z);
  }
}


template<typename PixelType>
__global__ void fftshift2_kernel(
    typename iu::LinearDeviceMemory<PixelType, 2>::KernelData src,
    typename iu::LinearDeviceMemory<PixelType, 2>::KernelData dst)
{
  unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

  int x_mid = src.size_[0] / 2.f;
  int y_mid = src.size_[1] / 2.f;

  if (x < src.size_[0] && y < src.size_[1])
  {
    unsigned int x_dst = (x + x_mid) % src.size_[0];
    unsigned int y_dst = (y + y_mid) % src.size_[1];
    dst(x_dst, y_dst) = src(x, y);
  }
}

template<typename PixelType, unsigned int Ndim>
__global__ void fftshift2_kernel(
    typename iu::LinearDeviceMemory<PixelType, Ndim>::KernelData src,
    typename iu::LinearDeviceMemory<PixelType, Ndim>::KernelData dst)
{
  unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int z = threadIdx.z + blockIdx.z * blockDim.z;
  unsigned int batch = src.numel_ / (src.size_[0] * src.size_[1]);

  int x_mid = src.size_[0] / 2.f;
  int y_mid = src.size_[1] / 2.f;

  if (x < src.size_[0] && y < src.size_[1] && z < batch)
  {
    unsigned int x_dst = (x + x_mid) % src.size_[0];
    unsigned int y_dst = (y + y_mid) % src.size_[1];
    dst(x_dst, y_dst, z) = src(x, y, z);
  }
}

////////////////////////////////////////////////////////////////////////////////////////
// ifftshift3
template<typename PixelType, class Allocator>
__global__ void ifftshift3_kernel(
    typename iu::VolumeGpu<PixelType, Allocator>::KernelData src,
    typename iu::VolumeGpu<PixelType, Allocator>::KernelData dst)
{
  unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int z = threadIdx.z + blockIdx.z * blockDim.z;

  unsigned int x_mid = (src.width_ + 1.f) / 2.f;
  unsigned int y_mid = (src.height_ + 1.f) / 2.f;
  unsigned int z_mid = (src.depth + 1.f) / 2.f;

  if (x < src.width_ && y < src.height_ && z < src.depth_)
  {
    unsigned int x_dst = (x + x_mid) % src.width_;
    unsigned int y_dst = (y + y_mid) % src.height_;
    unsigned int z_dst = (z + z_mid) % src.depth_;

    dst(x_dst, y_dst, z_dst) = src(x, y, z);
  }
}

template<typename PixelType>
__global__ void ifftshift3_kernel(
    typename iu::LinearDeviceMemory<PixelType, 3>::KernelData src,
    typename iu::LinearDeviceMemory<PixelType, 3>::KernelData dst)
{
  unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int z = threadIdx.z + blockIdx.z * blockDim.z;

  unsigned int x_mid = (src.size_[0] + 1.f) / 2.f;
  unsigned int y_mid = (src.size_[1] + 1.f) / 2.f;
  unsigned int z_mid = (src.size_[2] + 1.f) / 2.f;

  if (x < src.size_[0] && y < src.size_[1] && z < src.size_[2])
  {
    unsigned int x_dst = (x + x_mid) % src.size_[0];
    unsigned int y_dst = (y + y_mid) % src.size_[1];
    unsigned int z_dst = (z + z_mid) % src.size_[2];

    dst(x_dst, y_dst, z_dst) = src(x, y, z);
  }
}


template<typename PixelType, unsigned int Ndim>
__global__ void ifftshift3_kernel(
    typename iu::LinearDeviceMemory<PixelType, Ndim>::KernelData src,
    typename iu::LinearDeviceMemory<PixelType, Ndim>::KernelData dst)
{
  unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int z = threadIdx.z + blockIdx.z * blockDim.z;
  unsigned int batch = src.numel_ / (src.size_[0] * src.size_[1] * src.size_[2]);

  unsigned int x_mid = (src.size_[0] + 1.f) / 2.f;
  unsigned int y_mid = (src.size_[1] + 1.f) / 2.f;
  unsigned int z_mid = (src.size_[2] + 1.f) / 2.f;

  if (x < src.size_[0] && y < src.size_[1] && z < src.size_[2])
  {
    unsigned int x_dst = (x + x_mid) % src.size_[0];
    unsigned int y_dst = (y + y_mid) % src.size_[1];
    unsigned int z_dst = (z + z_mid) % src.size_[2];

    for (unsigned int s = 0; s < batch; s++)
      dst(x_dst, y_dst, z_dst, s) = src(x, y, z, s);
  }
}
