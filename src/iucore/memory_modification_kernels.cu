/*
 * Copyright (c) ICG. All rights reserved.
 *
 * Institute for Computer Graphics and Vision
 * Graz University of Technology / Austria
 *
 *
 * This software is distributed WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 * PURPOSE.  See the above copyright notices for more information.
 *
 *
 * Project     : ImageUtilities
 * Module      : Core
 * Class       : none
 * Language    : C/CUDA
 * Description : CUDA kernels for core functions
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUCORE_MEMORY_MODIFICATION_KERNELS_CU
#define IUCORE_MEMORY_MODIFICATION_KERNELS_CU

#include <cutil_math.h>
#include "iutextures.cuh"

namespace iuprivate {

///////////////////////////////////////////////////////////////////////////////

// kernel: 1D set values; 1D
template<class T>
__global__ void cuSetValueKernel(T value, T* dst, int length)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if(x<length)
  {
    dst[x] = value;
  }
}

// kernel: 2D set values; multi-channel
template<class T>
__global__ void cuSetValueKernel(T value, T* dst, size_t stride, unsigned int num_channels,
                                 int xoff, int yoff, int width, int height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  const int c =  y*stride+x*num_channels;

  if(x+xoff>=0 && y+yoff>=0 && x<width && y<height)
  {
    for(unsigned int channel = 0; channel<num_channels; ++channel)
      dst[c+channel] = value;
  }
}

// kernel: 3D set values; multi-channel
template<class T>
__global__ void cuSetValueKernel(T value, T* dst, size_t stride, size_t slice_stride,
                                 int xoff, int yoff, int zoff, int width, int height, int depth)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  const int c =  y*stride+x;

  if(x+xoff>=0 && y+yoff>=0 && x<width && y<height)
  {
    for(int z = -min(0,zoff); z<depth; ++z)
      dst[c+z*slice_stride] = value;
  }
}

///////////////////////////////////////////////////////////////////////////////
// thresholding/clamping kernels

// kernel: 1D set values; 1D
template<class T>
__global__ void cuClampKernel(T min, T max, T* dst, int length)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if(x<length)
  {
    dst[x] = clamp(min, max, dst[x]);
  }
}

// kernel: 2D set values;
template<class T>
__global__ void cuClampKernel_32f_C1(T min, T max, T* dst, size_t stride,
                                     int xoff, int yoff, int width, int height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  const int c =  y*stride+x;

  float xx = x + xoff + 0.5f;
  float yy = y + yoff + 0.5f;

  if(x>=0 && y>=0 && x<width && y<height)
  {
    dst[c] = clamp(tex2D(tex1_32f_C1__, xx, yy).x, min, max);
  }
}

///////////////////////////////////////////////////////////////////////////////

// convert 32f_C3 -> 32f_C4 (float3 -> float4)
__global__ void cuConvertC3ToC4Kernel(const float* src, size_t src_stride, int src_width, int src_height,
                                      float* dst, size_t dst_stride, int dst_width, int dst_height)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;
  int src_c = y*src_stride + x*3;
  int dst_c = y*dst_stride + x*4;

  if (x<src_width && y<src_height && x<dst_width && y<dst_height)
  {
    dst[dst_c] = src[src_c];
    dst[dst_c+1] = src[src_c+1];
    dst[dst_c+2] = src[src_c+2];
    dst[dst_c+3] = 1.0f; // make_float4(r, g, b, 1.0f);
  }
}

// convert 32f_C4 -> 32f_C3 (float4 -> float3)
__global__ void cuConvertC4ToC3Kernel(const float* src, size_t src_stride, int src_width, int src_height,
                                      float* dst, size_t dst_stride, int dst_width, int dst_height)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;
  int src_c = y*src_stride + x*4;
  int dst_c = y*dst_stride + x*3;

  if (x<src_width && y<src_height && x<dst_width && y<dst_height)
  {
    dst[dst_c] = src[src_c];
    dst[dst_c+1] = src[src_c+1];
    dst[dst_c+2] = src[src_c+2];
  }
}

} // namespace iuprivate

#endif // IUCORE_MEMORY_MODIFICATION_KERNELS_CU
