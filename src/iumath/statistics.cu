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
 * Module      : Math
 * Class       : none
 * Language    : C++
 * Description : Implementation of Cuda wrappers for statistics functions
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUMATH_STATISTICS_CU
#define IUMATH_STATISTICS_CU

#include <iucore/copy.h>
#include <iucore/setvalue.h>
#include <iucutil.h>
#include <iucore/iutextures.cuh>
#include "statistics.cuh"


#ifdef CUDA_NO_SM12_ATOMIC_INTRINSICS
#error Compilation target does not support shared-memory atomics
#endif

namespace iuprivate {

////////////////////////////////////////////////////////////////////////////////
__device__ inline void histogramAtomicAdd(float* address, float value)
{
  float old = value;
  while ((old = atomicExch(address, atomicExch(address, 0.0f)+old))!=0.0f);
}


/******************************************************************************
    CUDA KERNELS
*******************************************************************************/

/*
  KERNELS FOR MIN/MAX
*/

// kernel; find min/max; 8u_C1
__global__ void cuMinMaxXKernel_8u_C1(unsigned char* min, unsigned char* max,
                                      int xoff, int yoff, int width, int height)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;

  float xx = x+xoff+0.5f;
  float yy = y+yoff+0.5f;

  unsigned char cur_min = tex2D(tex1_8u_C1__, xx, yy);
  unsigned char cur_max = tex2D(tex1_8u_C1__, xx, yy);

  // find minima of columns
  if (x<width)
  {
    unsigned char val;
    for(int y = 1; y < height; ++y)
    {
      yy = y+yoff+0.5f;
      val = tex2D(tex1_8u_C1__, xx, yy);
      if(val < cur_min) cur_min = val;
      if(val > cur_max) cur_max = val;
    }

    min[x] = cur_min;
    max[x] = cur_max;
  }
}

__global__ void cuMinMaxXKernel2_8u_C1(unsigned char* min, unsigned char* max,
                                       int xoff, int yoff, int width, int height,
                                       const unsigned char* img, size_t stride)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;


  unsigned char cur_min = img[y*stride+x];
  unsigned char cur_max = cur_min;

  // find minima of columns
  if (x<width)
  {
    unsigned char val;
    for(int y = 1; y < height; ++y)
    {
      val = img[y*stride+x];
      if(val < cur_min) cur_min = val;
      if(val > cur_max) cur_max = val;
    }

    min[x] = cur_min;
    max[x] = cur_max;
  }
}


// kernel; find min/max; 8u_C4
__global__ void cuMinMaxXKernel_8u_C4(uchar4* min, uchar4* max,
                                      int xoff, int yoff, int width, int height)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;

  float xx = x+xoff+0.5f;
  float yy = y+yoff+0.5f;

  uchar4 cur_min = tex2D(tex1_8u_C4__, xx, yy);
  uchar4 cur_max = tex2D(tex1_8u_C4__, xx, yy);

  // find minima of columns
  if (x<width)
  {
    uchar4 val;
    for(int y = 1; y < height; ++y)
    {
      yy = y+yoff+0.5f;
      val = tex2D(tex1_8u_C4__, xx, yy);
      if(val.x < cur_min.x) cur_min.x = val.x;
      if(val.y < cur_min.y) cur_min.y = val.y;
      if(val.z < cur_min.z) cur_min.z = val.z;
      if(val.w < cur_min.w) cur_min.w = val.w;

      if(val.x > cur_max.x) cur_max.x = val.x;
      if(val.y > cur_max.y) cur_max.y = val.y;
      if(val.z > cur_max.z) cur_max.z = val.z;
      if(val.w > cur_max.w) cur_max.w = val.w;
    }

    min[x] = cur_min;
    max[x] = cur_max;
  }
}

// kernel; find min/max; 32f_C1
__global__ void cuMinMaxXKernel_32f_C1(float* min, float* max,
                                       int xoff, int yoff, int width, int height)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;

  // find minima of columns
  if (x<width)
  {
    float xx = x+xoff+0.5f;
    float yy = y+yoff+0.5f;

    float cur_min = tex2D(tex1_32f_C1__, xx, yy);
    float cur_max = tex2D(tex1_32f_C1__, xx, yy);

    float val;
    for(int y = 1; y < height; ++y)
    {
      yy = y+yoff+0.5f;
      val = tex2D(tex1_32f_C1__, xx, yy);
      if(val < cur_min) cur_min = val;
      if(val > cur_max) cur_max = val;
    }

    min[x] = cur_min;
    max[x] = cur_max;
  }
}

// kernel; find min/max; 32f_C2
__global__ void cuMinMaxXKernel_32f_C2(float2* min, float2* max,
                                       int xoff, int yoff, int width, int height)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;

  float xx = x+xoff+0.5f;
  float yy = y+yoff+0.5f;

  float2 cur_min = tex2D(tex1_32f_C2__, xx, yy);
  float2 cur_max = tex2D(tex1_32f_C2__, xx, yy);

  // find minima of columns
  if (x<width)
  {
    float2 val;
    for(int y = 1; y < height; ++y)
    {
      yy = y+yoff+0.5f;
      val = tex2D(tex1_32f_C2__, xx, yy);
      if(val.x < cur_min.x) cur_min.x = val.x;
      if(val.y < cur_min.y) cur_min.y = val.y;

      if(val.x > cur_max.x) cur_max.x = val.x;
      if(val.y > cur_max.y) cur_max.y = val.y;
    }

    min[x] = cur_min;
    max[x] = cur_max;
  }
}

// kernel; find min/max; 32f_C4
__global__ void cuMinMaxXKernel_32f_C4(float4* min, float4* max,
                                       int xoff, int yoff, int width, int height)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;

  float xx = x+xoff+0.5f;
  float yy = y+yoff+0.5f;

  float4 cur_min = tex2D(tex1_32f_C4__, xx, yy);
  float4 cur_max = tex2D(tex1_32f_C4__, xx, yy);

  // find minima of columns
  if (x<width)
  {
    float4 val;
    for(int y = 0; y < height; ++y)
    {
      yy = y+yoff+0.5f;
      val = tex2D(tex1_32f_C4__, xx, yy);
      if(val.x < cur_min.x) cur_min.x = val.x;
      if(val.y < cur_min.y) cur_min.y = val.y;
      if(val.z < cur_min.z) cur_min.z = val.z;
      if(val.w < cur_min.w) cur_min.w = val.w;

      if(val.x > cur_max.x) cur_max.x = val.x;
      if(val.y > cur_max.y) cur_max.y = val.y;
      if(val.z > cur_max.z) cur_max.z = val.z;
      if(val.w > cur_max.w) cur_max.w = val.w;
    }

    min[x] = cur_min;
    max[x] = cur_max;
  }
}

/*
  KERNELS FOR min + min COORDS
*/

// kernel; find min + min idx; 32f_C1
__global__ void cuMinXKernel_32f_C1(float* min, unsigned short* min_col_idx,
                                    int xoff, int yoff, int width, int height)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;

  // find minima of columns
  if (x<width)
  {
    float xx = x+xoff+0.5f;
    float yy = y+yoff+0.5f;

    unsigned short cur_min_col_idx = 0;
    float cur_min = tex2D(tex1_32f_C1__, xx, yy);

    float val;
    for(unsigned short y = 1; y < height; ++y)
    {
      yy = y+yoff+0.5f;
      val = tex2D(tex1_32f_C1__, xx, yy);
      if(val < cur_min)
      {
        cur_min_col_idx = y;
        cur_min = val;
      }
    }

    min_col_idx[x] = cur_min_col_idx;
    min[x] = cur_min;
  }
}

/*
  KERNELS FOR MAX + MAX COORDS
*/

// kernel; find max + max idx; 32f_C1
__global__ void cuMaxXKernel_32f_C1(float* max, float* max_col_idx,
                                    int xoff, int yoff, int width, int height)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;

  float xx = x+xoff+0.5f;
  float yy = y+yoff+0.5f;

  float cur_max_col_idx = 0.0f;
  float cur_max = tex2D(tex1_32f_C1__, xx, yy);

  // find minima of columns
  if (x<width)
  {
    float val;
    for(int y = 0; y < height; ++y)
    {
      yy = y+yoff+0.5f;
      val = tex2D(tex1_32f_C1__, xx, yy);
      if(val > cur_max)
      {
        cur_max_col_idx = (float)y;
        cur_max = val;
      }
    }

    max_col_idx[x] = cur_max_col_idx;
    max[x] = cur_max;
  }
}

/*
  KERNELS FOR SUM
*/

// kernel; compute sum; 8u_C1
__global__ void cuSumColKernel_8u_C1(unsigned char* sum, int xoff, int yoff, int width, int height)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;

  float xx = x+xoff+0.5f;
  float yy = y+yoff+0.5f;

  float cur_sum = 0.0f;

  // compute sum of each column
  if (xx<width+0.5f)
  {
    for(int y = yoff; y < height; ++y)
    {
      yy = y+0.5f;
      cur_sum += tex2D(tex1_8u_C1__, xx, yy);
    }
    sum[x] = cur_sum;
  }
}

// kernel; compute sum; 32f_C1
__global__ void cuSumColKernel_32f_C1(float* sum, int xoff, int yoff, int width, int height)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;

  float xx = x+xoff+0.5f;
  float yy = y+yoff+0.5f;

  float cur_sum = 0.0f;

  // compute sum of each column
  if ((x+xoff)<width)
  {
    for(int y = yoff; y < height; ++y)
    {
      yy = y+0.5f;
      cur_sum += tex2D(tex1_32f_C1__, xx, yy);
    }
    sum[x] = cur_sum;
  }
}

/*
  KERNELS for NORM OF DIFFERENCES
*/

// kernel: compute L1 norm; |image1-image2|;
__global__ void  cuNormDiffL1Kernel_32f_C1(float* dst, size_t stride,
                                           int xoff, int yoff, int width, int height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int oc = y*stride+x;

  x += xoff;
  y += yoff;

  float xx = x+0.5f;
  float yy = y+0.5f;

  if(x>=0 && y>= 0 && x<width && y<height)
  {
    dst[oc] = fabs(tex2D(tex1_32f_C1__, xx, yy) - tex2D(tex2_32f_C1__, xx, yy));
  }
}

// kernel: compute L1 norm; |image-value|;
__global__ void  cuNormDiffValueL1Kernel_32f_C1(float value, float* dst, size_t stride,
                                                int xoff, int yoff, int width, int height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int oc = y*stride+x;

  x += xoff;
  y += yoff;

  float xx = x+0.5f;
  float yy = y+0.5f;

  if(x>=0 && y>= 0 && x<width && y<height)
  {
    dst[oc] = fabs(tex2D(tex1_32f_C1__, xx, yy) - value);
  }
}

// kernel: compute L2 norm; ||image1-image2||;
__global__ void  cuNormDiffL2Kernel_32f_C1(float* dst, size_t stride,
                                           int xoff, int yoff, int width, int height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int oc = y*stride+x;

  x += xoff;
  y += yoff;

  float xx = x+0.5f;
  float yy = y+0.5f;

  if(x>=0 && y>= 0 && x<width && y<height)
  {
    dst[oc] = iu::sqr(tex2D(tex1_32f_C1__, xx, yy) - tex2D(tex2_32f_C1__, xx, yy));
  }
}

// kernel: compute L2 norm; ||image-value||;
__global__ void  cuNormDiffValueL2Kernel_32f_C1(float value, float* dst, size_t stride,
                                                int xoff, int yoff, int width, int height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int oc = y*stride+x;

  x += xoff;
  y += yoff;

  float xx = x+0.5f;
  float yy = y+0.5f;

  if(x>=0 && y>= 0 && x<width && y<height)
  {
    dst[oc] = iu::sqr(tex2D(tex1_32f_C1__, xx, yy) - value);
  }
}

/******************************************************************************
  CUDA INTERFACES
*******************************************************************************/

/*
  WRAPPERS FOR MIN/MAX
*/

// wrapper: find min/max; 8u_C1
IuStatus cuMinMax(const iu::ImageGpu_8u_C1 *src, const IuRect &roi,
                  unsigned char& min_C1, unsigned char& max_C1)
{
  // prepare and bind texture
  tex1_8u_C1__.filterMode = cudaFilterModePoint;
  tex1_8u_C1__.addressMode[0] = cudaAddressModeClamp;
  tex1_8u_C1__.addressMode[1] = cudaAddressModeClamp;
  tex1_8u_C1__.normalized = false;
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar1>();
//  printf("bind texture\n");
  cudaBindTexture2D(0, &tex1_8u_C1__, src->data(), &channel_desc,
                    src->width(), src->height(), src->pitch());

  // fragmentation
  const unsigned int block_width = 512;
  dim3 dimBlock(block_width, 1, 1);
  dim3 dimGridX(iu::divUp(roi.width, block_width), 1);

  // temporary memory for row sums on the host
  int num_row_sums = roi.width;
  iu::LinearDeviceMemory_8u_C1 row_mins(num_row_sums);
  iu::LinearDeviceMemory_8u_C1 row_maxs(num_row_sums);

#if 1
  cuMinMaxXKernel_8u_C1 <<< dimGridX, dimBlock >>> (
                                                    row_mins.data(), row_maxs.data(), roi.x, roi.y, roi.width, roi.height);
#else
  cuMinMaxXKernel2_8u_C1  <<< dimGridX, dimBlock >>> (
                                                      row_mins.data(), row_maxs.data(), roi.x, roi.y, roi.width, roi.height,
                                                      src->data(), src->stride());
#endif
  iu::LinearHostMemory_8u_C1 h_row_mins(num_row_sums);
  iu::LinearHostMemory_8u_C1 h_row_maxs(num_row_sums);
  iuprivate::copy(&row_mins, &h_row_mins);
  iuprivate::copy(&row_maxs, &h_row_maxs);

  min_C1 = *h_row_mins.data();
  max_C1 = *h_row_maxs.data();

  for (int i = 0; i < num_row_sums; ++i)
  {
//    printf("#%d: %d / %d\n", i, h_row_mins.data(i)[0], h_row_maxs.data(i)[0]);
    min_C1 = IUMIN(min_C1, *h_row_mins.data(i));
    max_C1 = IUMAX(max_C1, *h_row_maxs.data(i));
  }

  cudaUnbindTexture(&tex1_8u_C1__);
//  printf("min/max=%d/%d\n", min_C1, max_C1);
  return iu::checkCudaErrorState();
}

// wrapper: find min/max; 8u_C4
IuStatus cuMinMax(const iu::ImageGpu_8u_C4 *src, const IuRect &roi, uchar4& min_C4, uchar4& max_C4)
{
  // prepare and bind texture
  tex1_8u_C4__.filterMode = cudaFilterModePoint;
  tex1_8u_C4__.addressMode[0] = cudaAddressModeClamp;
  tex1_8u_C4__.addressMode[1] = cudaAddressModeClamp;
  tex1_8u_C4__.normalized = false;
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();
  cudaBindTexture2D(0, &tex1_8u_C4__, src->data(), &channel_desc,
                    src->width(), src->height(), src->pitch());

  // fragmentation
  const unsigned int block_width = 512;
  dim3 dimBlock(block_width, 1, 1);
  dim3 dimGridX(iu::divUp(roi.width, block_width), 1);

  // temporary memory for row sums on the host
  int num_row_sums = roi.width;
  iu::LinearDeviceMemory_8u_C4 row_mins(num_row_sums);
  iu::LinearDeviceMemory_8u_C4 row_maxs(num_row_sums);

  cuMinMaxXKernel_8u_C4 <<< dimGridX, dimBlock >>> (
                                                    row_mins.data(), row_maxs.data(),
                                                    roi.x, roi.y, roi.width, roi.height);

  iu::LinearHostMemory_8u_C4 h_row_mins(num_row_sums);
  iu::LinearHostMemory_8u_C4 h_row_maxs(num_row_sums);
  iuprivate::copy(&row_mins, &h_row_mins);
  iuprivate::copy(&row_maxs, &h_row_maxs);

  min_C4 = *h_row_mins.data(0);
  max_C4 = *h_row_maxs.data(0);

  for (int i = 1; i < num_row_sums; ++i)
  {
    min_C4.x = IUMIN(min_C4.x, h_row_mins.data(i)->x);
    min_C4.y = IUMIN(min_C4.y, h_row_mins.data(i)->y);
    min_C4.z = IUMIN(min_C4.z, h_row_mins.data(i)->z);
    min_C4.w = IUMIN(min_C4.w, h_row_mins.data(i)->w);

    max_C4.x = IUMAX(max_C4.x, h_row_maxs.data(i)->x);
    max_C4.y = IUMAX(max_C4.y, h_row_maxs.data(i)->y);
    max_C4.z = IUMAX(max_C4.z, h_row_maxs.data(i)->z);
    max_C4.w = IUMAX(max_C4.w, h_row_maxs.data(i)->w);
  }

  cudaUnbindTexture(&tex1_8u_C4__);
  return iu::checkCudaErrorState();
}

// wrapper: find min/max; 32f_C1
IuStatus cuMinMax(const iu::ImageGpu_32f_C1 *src, const IuRect &roi, float& min_C1, float& max_C1)
{
  // prepare and bind texture
  tex1_32f_C1__.filterMode = cudaFilterModePoint;
  tex1_32f_C1__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C1__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C1__.normalized = false;
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float1>();
  cudaBindTexture2D(0, &tex1_32f_C1__, src->data(), &channel_desc,
                    src->width(), src->height(), src->pitch());

  // fragmentation
  const unsigned int block_width = 512;
  dim3 dimBlock(block_width, 1, 1);
  dim3 dimGridX(iu::divUp(roi.width, block_width), 1);

  // temporary memory for row sums on the host
  int num_row_sums = roi.width;
  iu::LinearDeviceMemory_32f_C1 row_mins(num_row_sums);
  iu::LinearDeviceMemory_32f_C1 row_maxs(num_row_sums);

  cuMinMaxXKernel_32f_C1 <<< dimGridX, dimBlock >>> (
                                                     row_mins.data(), row_maxs.data(), roi.x, roi.y, roi.width, roi.height);

  iu::LinearHostMemory_32f_C1 h_row_mins(num_row_sums);
  iu::LinearHostMemory_32f_C1 h_row_maxs(num_row_sums);
  iuprivate::copy(&row_mins, &h_row_mins);
  iuprivate::copy(&row_maxs, &h_row_maxs);

  min_C1 = *h_row_mins.data(0);
  max_C1 = *h_row_maxs.data(0);

  for (int i = 1; i < num_row_sums; ++i)
  {
    min_C1 = IUMIN(min_C1, *h_row_mins.data(i));
    max_C1 = IUMAX(max_C1, *h_row_maxs.data(i));
  }

  cudaUnbindTexture(&tex1_32f_C1__);
  return iu::checkCudaErrorState();
}


// kernel; find min/max; 32f_C1
__global__ void cuMinMaxXYKernel_32f_C1(float* minim, float* maxim,
                                        int width, int height, int m_stride,
                                        float* data, int depth, int d_stride,
                                        int d_slice_stride)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;

  // find minima of columns
  if (x<width && y<height)
  {
    float cur_min = data[y*d_stride + x];
    float cur_max = cur_min;

    float val;
    for(int z = 1; z < depth; z++)
    {
      val = data[z*d_slice_stride + y*d_stride + x];
      if(val < cur_min) cur_min = val;
      if(val > cur_max) cur_max = val;
    }

    minim[y*m_stride + x] = cur_min;
    maxim[y*m_stride + x] = cur_max;
  }
}

// kernel; find min; 32f_C1
__global__ void cuMinXKernel_32f_C1(float* minim, int width, int height)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;

  // find minima of columns
  if (x<width)
  {
    float xx = x+0.5f;
    float yy = y+0.5f;

    float cur_min = tex2D(tex1_32f_C1__, xx, yy);

    float val;
    for(int y = 1; y < height; ++y)
    {
      yy = y+0.5f;
      val = tex2D(tex1_32f_C1__, xx, yy);
      if(val < cur_min) cur_min = val;
    }

    minim[x] = cur_min;
  }
}

// kernel; find max; 32f_C1
__global__ void cuMaxXKernel_32f_C1(float* maxim, int width, int height)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;

  // find minima of columns
  if (x<width)
  {
    float xx = x+0.5f;
    float yy = y+0.5f;

    float cur_max = tex2D(tex1_32f_C1__, xx, yy);

    float val;
    for(int y = 1; y < height; ++y)
    {
      yy = y+0.5f;
      val = tex2D(tex1_32f_C1__, xx, yy);
      if(val > cur_max) cur_max = val;
    }

    maxim[x] = cur_max;
  }
}


// wrapper: find min/max; 32f_C1
IuStatus cuMinMax(iu::VolumeGpu_32f_C1 *src, float& min_C1, float& max_C1)
{
  iu::ImageGpu_32f_C1 minim(src->width(), src->height());
  iu::ImageGpu_32f_C1 maxim(src->width(), src->height());

  const unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(src->width(), dimBlock.x), iu::divUp(src->height(), dimBlock.y));

  cuMinMaxXYKernel_32f_C1<<< dimGrid,dimBlock >>>(minim.data(), maxim.data(),
                                                  minim.width(), minim.height(), minim.stride(),
                                                  src->data(), src->depth(),
                                                  src->stride(), src->slice_stride());

  // prepare and bind texture
  tex1_32f_C1__.filterMode = cudaFilterModePoint;
  tex1_32f_C1__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C1__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C1__.normalized = false;
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float1>();

  // fragmentation
  const unsigned int block_width = 512;
  dim3 dimBlockX(block_width, 1, 1);
  dim3 dimGridX(iu::divUp(minim.width(), block_width), 1);

  // find minimum
  cudaBindTexture2D(0, &tex1_32f_C1__, minim.data(), &channel_desc,
                    minim.width(), minim.height(), minim.pitch());
  int num_row_sums = minim.width();
  iu::LinearDeviceMemory_32f_C1 row_mins(num_row_sums);
  cuMinXKernel_32f_C1<<<dimGridX, dimBlockX>>>(row_mins.data(), minim.width(), minim.height());
  iu::LinearHostMemory_32f_C1 h_row_mins(num_row_sums);
  iuprivate::copy(&row_mins, &h_row_mins);
  min_C1 = *h_row_mins.data(0);
  for (int i = 1; i < num_row_sums; ++i)
    min_C1 = IUMIN(min_C1, *h_row_mins.data(i));
  cudaUnbindTexture(&tex1_32f_C1__);

  // find maximum
  cudaBindTexture2D(0, &tex1_32f_C1__, maxim.data(), &channel_desc,
                    maxim.width(), maxim.height(), maxim.pitch());
  iu::LinearDeviceMemory_32f_C1 row_maxs(num_row_sums);
  cuMaxXKernel_32f_C1<<<dimGridX, dimBlockX>>>(row_maxs.data(), maxim.width(), maxim.height());
  iu::LinearHostMemory_32f_C1 h_row_maxs(num_row_sums);
  iuprivate::copy(&row_maxs, &h_row_maxs);
  max_C1 = *h_row_maxs.data(0);
  for (int i = 1; i < num_row_sums; ++i)
    max_C1 = IUMAX(max_C1, *h_row_maxs.data(i));
  cudaUnbindTexture(&tex1_32f_C1__);

  return iu::checkCudaErrorState();
}

// wrapper: find min/max; 32f_C2
IuStatus cuMinMax(const iu::ImageGpu_32f_C2 *src, const IuRect &roi, float2& min_C2, float2& max_C2)
{
  // prepare and bind texture
  tex1_32f_C2__.filterMode = cudaFilterModePoint;
  tex1_32f_C2__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C2__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C2__.normalized = false;
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float2>();
  cudaBindTexture2D(0, &tex1_32f_C2__, src->data(), &channel_desc,
                    src->width(), src->height(), src->pitch());

  // fragmentation
  const unsigned int block_width = 512;
  dim3 dimBlock(block_width, 1, 1);
  dim3 dimGridX(iu::divUp(roi.width, block_width), 1);

  // temporary memory for row sums on the host
  int num_row_sums = roi.width;
  iu::LinearDeviceMemory_32f_C2 row_mins(num_row_sums);
  iu::LinearDeviceMemory_32f_C2 row_maxs(num_row_sums);

  cuMinMaxXKernel_32f_C2 <<< dimGridX, dimBlock >>> (
                                                     row_mins.data(), row_maxs.data(),
                                                     roi.x, roi.y, roi.width, roi.height);

  iu::LinearHostMemory_32f_C2 h_row_mins(num_row_sums);
  iu::LinearHostMemory_32f_C2 h_row_maxs(num_row_sums);
  iuprivate::copy(&row_mins, &h_row_mins);
  iuprivate::copy(&row_maxs, &h_row_maxs);

  min_C2 = *h_row_mins.data(0);
  max_C2 = *h_row_maxs.data(0);

  for (int i = 1; i < num_row_sums; ++i)
  {
    min_C2.x = IUMIN(min_C2.x, h_row_mins.data(i)->x);
    min_C2.y = IUMIN(min_C2.y, h_row_mins.data(i)->y);

    max_C2.x = IUMAX(max_C2.x, h_row_maxs.data(i)->x);
    max_C2.y = IUMAX(max_C2.y, h_row_maxs.data(i)->y);
  }

  cudaUnbindTexture(&tex1_32f_C2__);
  return iu::checkCudaErrorState();
}

// wrapper: find min/max; 32f_C4
IuStatus cuMinMax(const iu::ImageGpu_32f_C4 *src, const IuRect &roi, float4& min_C4, float4& max_C4)
{
  // prepare and bind texture
  tex1_32f_C4__.filterMode = cudaFilterModePoint;
  tex1_32f_C4__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C4__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C4__.normalized = false;
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
  cudaBindTexture2D(0, &tex1_32f_C4__, src->data(), &channel_desc,
                    src->width(), src->height(), src->pitch());

  // fragmentation
  const unsigned int block_width = 512;
  dim3 dimBlock(block_width, 1, 1);
  dim3 dimGridX(iu::divUp(roi.width, block_width), 1);

  // temporary memory for row sums on the host
  int num_cols = roi.width;
  iu::LinearDeviceMemory_32f_C4 row_mins(num_cols);
  iu::LinearDeviceMemory_32f_C4 row_maxs(num_cols);

  cuMinMaxXKernel_32f_C4 <<< dimGridX, dimBlock >>> (
                                                     row_mins.data(), row_maxs.data(),
                                                     roi.x, roi.y, roi.width, roi.height);

  iu::LinearHostMemory_32f_C4 h_row_mins(num_cols);
  iu::LinearHostMemory_32f_C4 h_row_maxs(num_cols);
  iuprivate::copy(&row_mins, &h_row_mins);
  iuprivate::copy(&row_maxs, &h_row_maxs);

  min_C4 = *h_row_mins.data(0);
  max_C4 = *h_row_maxs.data(0);

  for (int i = 1; i < num_cols; ++i)
  {
    min_C4.x = IUMIN(min_C4.x, h_row_mins.data(i)->x);
    min_C4.y = IUMIN(min_C4.y, h_row_mins.data(i)->y);
    min_C4.z = IUMIN(min_C4.z, h_row_mins.data(i)->z);
    min_C4.w = IUMIN(min_C4.w, h_row_mins.data(i)->w);

    max_C4.x = IUMAX(max_C4.x, h_row_maxs.data(i)->x);
    max_C4.y = IUMAX(max_C4.y, h_row_maxs.data(i)->y);
    max_C4.z = IUMAX(max_C4.z, h_row_maxs.data(i)->z);
    max_C4.w = IUMAX(max_C4.w, h_row_maxs.data(i)->w);
  }

  cudaUnbindTexture(&tex1_32f_C4__);
  return iu::checkCudaErrorState();
}

/*
  WRAPPERS FOR MIN + MIN COORDINATES
*/

// wrapper: find min + min idx; 32f_C1
IuStatus cuMin(const iu::ImageGpu_32f_C1 *src, const IuRect &roi,
               float& min, int& min_x, int& min_y)
{
  // prepare and bind texture
  tex1_32f_C1__.filterMode = cudaFilterModePoint;
  tex1_32f_C1__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C1__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C1__.normalized = false;
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float1>();
  cudaBindTexture2D(0, &tex1_32f_C1__, src->data(), &channel_desc,
                    src->width(), src->height(), src->pitch());

  // fragmentation
  const unsigned int block_width = 512;
  dim3 dimBlock(block_width, 1, 1);
  dim3 dimGridX(iu::divUp(roi.width, block_width), 1);

  // temporary memory for row sums on the host
  int num_cols = roi.width;
  iu::LinearDeviceMemory_32f_C1 col_mins(num_cols);
  iu::LinearDeviceMemory_16u_C1 col_min_idxs(num_cols);

  cuMinXKernel_32f_C1
      <<< dimGridX, dimBlock >>> (col_mins.data(), col_min_idxs.data(),
                                  roi.x, roi.y, roi.width, roi.height);

  iu::LinearHostMemory_32f_C1 h_col_mins(num_cols);
  iuprivate::copy(&col_mins, &h_col_mins);
  iu::LinearHostMemory_16u_C1 h_col_min_idxs(num_cols);
  iuprivate::copy(&col_min_idxs, &h_col_min_idxs);

  min_x = roi.x;
  min_y = (int)(roi.y + *h_col_min_idxs.data(0));
  min = *h_col_mins.data(0);

  for (int i = 1; i < num_cols; ++i)
  {
    if(min > *h_col_mins.data(i))
    {
      min = *h_col_mins.data(i);
      min_x = roi.x + i;
      min_y = (int)(roi.y + *h_col_min_idxs.data(i));
    }
  }

  cudaUnbindTexture(&tex1_32f_C1__);
  return iu::checkCudaErrorState();
}

/*
  WRAPPERS FOR MAX + MAX COORDINATES
*/

// wrapper: find max + max idx; 32f_C1
IuStatus cuMax(const iu::ImageGpu_32f_C1 *src, const IuRect &roi,
               float& max, int& max_x, int& max_y)
{
  // prepare and bind texture
  tex1_32f_C1__.filterMode = cudaFilterModePoint;
  tex1_32f_C1__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C1__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C1__.normalized = false;
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float1>();
  cudaBindTexture2D(0, &tex1_32f_C1__, src->data(), &channel_desc,
                    src->width(), src->height(), src->pitch());

  // fragmentation
  const unsigned int block_width = 512;
  dim3 dimBlock(block_width, 1, 1);
  dim3 dimGridX(iu::divUp(roi.width, block_width), 1);

  // temporary memory for row sums on the host
  int num_cols = roi.width;
  iu::LinearDeviceMemory_32f_C1 col_maxs(num_cols);
  iu::LinearDeviceMemory_32f_C1 col_max_idxs(num_cols);

  cuMaxXKernel_32f_C1
      <<< dimGridX, dimBlock >>> (col_maxs.data(), col_max_idxs.data(),
                                  roi.x, roi.y, roi.width, roi.height);

  iu::LinearHostMemory_32f_C1 h_col_max_idxs(num_cols);
  iu::LinearHostMemory_32f_C1 h_col_maxs(num_cols);
  iuprivate::copy(&col_max_idxs, &h_col_max_idxs);
  iuprivate::copy(&col_maxs, &h_col_maxs);

  max_y = (int)(roi.y + *h_col_max_idxs.data(0));
  max = *h_col_maxs.data(0);

  for (int i = 1; i < num_cols; ++i)
  {
    if(max < *h_col_maxs.data(i))
    {
      max = *h_col_maxs.data(i);
      max_x = roi.x + i;
      max_y = (int)(roi.y + *h_col_max_idxs.data(i));
    }
  }

  cudaUnbindTexture(&tex1_32f_C1__);
  return iu::checkCudaErrorState();
}



/*
  WRAPPERS FOR SUM
*/

// wrapper: compute sum; 8u_C1
IuStatus cuSummation(const iu::ImageGpu_8u_C1 *src, const IuRect &roi, long& sum)
{
  // prepare and bind texture
  tex1_8u_C1__.filterMode = cudaFilterModePoint;
  tex1_8u_C1__.addressMode[0] = cudaAddressModeClamp;
  tex1_8u_C1__.addressMode[1] = cudaAddressModeClamp;
  tex1_8u_C1__.normalized = false;
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar1>();
  cudaBindTexture2D(0, &tex1_8u_C1__, src->data(), &channel_desc, src->width(), src->height(), src->pitch());

  // fragmentation
  const unsigned int block_width = 512;
  dim3 dimBlock(block_width, 1, 1);
  dim3 dimGridX(iu::divUp(roi.width, block_width), 1);

  // temporary memory for row sums on the host
  int num_col_sums = roi.width;
  iu::LinearDeviceMemory_8u_C1 col_sums(num_col_sums);

  cuSumColKernel_8u_C1 <<< dimGridX, dimBlock >>> (
                                                   col_sums.data(), roi.x, roi.y, roi.width, roi.height);

  // :TODO: 32f vs 32u?
  iu::LinearHostMemory_8u_C1 h_col_sums(num_col_sums);
  iuprivate::copy(&col_sums, &h_col_sums);

  sum = 0;
  for (int i = 0; i < num_col_sums; ++i)
  {
    sum += static_cast<unsigned int>(*h_col_sums.data(i));
  }

  cudaUnbindTexture(&tex1_8u_C1__);
  return iu::checkCudaErrorState();
}

// wrapper: compute sum; 32f_C1
IuStatus cuSummation(const iu::ImageGpu_32f_C1 *src, const IuRect &roi, double& sum)
{
  // prepare and bind texture
  tex1_32f_C1__.filterMode = cudaFilterModePoint;
  tex1_32f_C1__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C1__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C1__.normalized = false;
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float1>();
  cudaBindTexture2D(0, &tex1_32f_C1__, src->data(), &channel_desc, src->width(), src->height(), src->pitch());

  // fragmentation
  const unsigned int block_width = 512;
  dim3 dimBlock(block_width, 1, 1);
  dim3 dimGridX(iu::divUp(roi.width, block_width), 1);

  // temporary memory for row sums on the host
  int num_col_sums = roi.width;
  iu::LinearDeviceMemory_32f_C1 col_sums(num_col_sums);

  cuSumColKernel_32f_C1 <<< dimGridX, dimBlock >>> (
                                                    col_sums.data(), roi.x, roi.y, roi.width, roi.height);

  iu::LinearHostMemory_32f_C1 h_col_sums(num_col_sums);
  iuprivate::copy(&col_sums, &h_col_sums);

  sum = 0.0;
  for (int i = 0; i < num_col_sums; ++i)
  {
    sum += *h_col_sums.data(i);
  }

  cudaUnbindTexture(&tex1_32f_C1__);
  return iu::checkCudaErrorState();
}

/*
  WRAPPERS for NORM OF DIFFERENCES
*/

// wrapper: compute L1 norm; |image1-image2|;
IuStatus cuNormDiffL1(const iu::ImageGpu_32f_C1* src1, const iu::ImageGpu_32f_C1* src2, const IuRect& roi, double& norm)
{
  // prepare and bind texture
  tex1_32f_C1__.filterMode = cudaFilterModePoint;
  tex1_32f_C1__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C1__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C1__.normalized = false;
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float1>();
  cudaBindTexture2D(0, &tex1_32f_C1__, src1->data(), &channel_desc, src1->width(), src1->height(), src1->pitch());

  // prepare and bind texture
  tex2_32f_C1__.filterMode = cudaFilterModePoint;
  tex2_32f_C1__.addressMode[0] = cudaAddressModeClamp;
  tex2_32f_C1__.addressMode[1] = cudaAddressModeClamp;
  tex2_32f_C1__.normalized = false;
  cudaBindTexture2D(0, &tex2_32f_C1__, src2->data(), &channel_desc, src2->width(), src2->height(), src2->pitch());

  const unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(roi.width - roi.x, dimBlock.x),
               iu::divUp(roi.height - roi.y, dimBlock.y));

  iu::ImageGpu_32f_C1 squared_deviances(roi.width, roi.height);

  cuNormDiffL1Kernel_32f_C1 <<< dimGrid, dimBlock >>> (
                                                       squared_deviances.data(roi.x, roi.y), squared_deviances.stride(), roi.x, roi.y, roi.width, roi.height);

  double sum_squared = 0.0;
  iuprivate::cuSummation(&squared_deviances, roi, sum_squared);
  norm = sqrtf(sum_squared);

  return iu::checkCudaErrorState();
}

// wrapper: compute L1 norm; |image1-value|;
IuStatus cuNormDiffL1(const iu::ImageGpu_32f_C1* src, const float& value, const IuRect& roi, double& norm)
{
  // prepare and bind texture
  tex1_32f_C1__.filterMode = cudaFilterModePoint;
  tex1_32f_C1__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C1__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C1__.normalized = false;
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float1>();
  cudaBindTexture2D(0, &tex1_32f_C1__, src->data(), &channel_desc, src->width(), src->height(), src->pitch());

  const unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(roi.width - roi.x, dimBlock.x),
               iu::divUp(roi.height - roi.y, dimBlock.y));

  iu::ImageGpu_32f_C1 squared_deviances(roi.width, roi.height);

  cuNormDiffValueL1Kernel_32f_C1 <<< dimGrid, dimBlock >>> (
                                                            value, squared_deviances.data(roi.x, roi.y), squared_deviances.stride(),
                                                            roi.x, roi.y, roi.width, roi.height);

  double sum_squared = 0.0;
  iuprivate::cuSummation(&squared_deviances, roi, sum_squared);
  norm = sqrtf(sum_squared);

  return iu::checkCudaErrorState();
}

// wrapper: compute L2 norm; ||image1-image2||;
IuStatus cuNormDiffL2(const iu::ImageGpu_32f_C1* src1, const iu::ImageGpu_32f_C1* src2, const IuRect& roi, double& norm)
{
  // prepare and bind texture
  tex1_32f_C1__.filterMode = cudaFilterModePoint;
  tex1_32f_C1__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C1__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C1__.normalized = false;
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float1>();
  cudaBindTexture2D(0, &tex1_32f_C1__, src1->data(), &channel_desc, src1->width(), src1->height(), src1->pitch());

  // prepare and bind texture
  tex2_32f_C1__.filterMode = cudaFilterModePoint;
  tex2_32f_C1__.addressMode[0] = cudaAddressModeClamp;
  tex2_32f_C1__.addressMode[1] = cudaAddressModeClamp;
  tex2_32f_C1__.normalized = false;
  cudaBindTexture2D(0, &tex2_32f_C1__, src2->data(), &channel_desc, src2->width(), src2->height(), src2->pitch());

  const unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(roi.width - roi.x, dimBlock.x),
               iu::divUp(roi.height - roi.y, dimBlock.y));

  iu::ImageGpu_32f_C1 squared_deviances(roi.width, roi.height);

  cuNormDiffL2Kernel_32f_C1 <<< dimGrid, dimBlock >>> (
                                                       squared_deviances.data(roi.x, roi.y), squared_deviances.stride(), roi.x, roi.y, roi.width, roi.height);

  double sum_squared = 0.0;
  iuprivate::cuSummation(&squared_deviances, roi, sum_squared);
  norm = sqrtf(sum_squared);

  return iu::checkCudaErrorState();
}

// wrapper: compute L2 norm; ||image1-value||;
IuStatus cuNormDiffL2(const iu::ImageGpu_32f_C1* src, const float& value, const IuRect& roi, double& norm)
{
  // prepare and bind texture
  tex1_32f_C1__.filterMode = cudaFilterModePoint;
  tex1_32f_C1__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C1__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C1__.normalized = false;
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float1>();
  cudaBindTexture2D(0, &tex1_32f_C1__, src->data(), &channel_desc, src->width(), src->height(), src->pitch());

  const unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(roi.width - roi.x, dimBlock.x),
               iu::divUp(roi.height - roi.y, dimBlock.y));

  iu::ImageGpu_32f_C1 squared_deviances(roi.width, roi.height);

  cuNormDiffValueL2Kernel_32f_C1 <<< dimGrid, dimBlock >>> (
                                                            value, squared_deviances.data(roi.x, roi.y), squared_deviances.stride(),
                                                            roi.x, roi.y, roi.width, roi.height);

  double sum_squared = 0.0;
  iuprivate::cuSummation(&squared_deviances, roi, sum_squared);
  norm = sqrtf(sum_squared);

  return iu::checkCudaErrorState();
}

/*
  WRAPPERS for ERROR MEASUREMENTS
*/


// kernel: compute MSE
__global__ void cuMseKernel(float* dst, size_t stride, int xoff, int yoff, int width, int height)
{
  // calculate absolute texture coordinates
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;

  if ((x < width) && (y < height))
  {
    float diff = tex2D(tex1_32f_C1__, x+xoff+0.5f, y+yoff+0.5f) - tex2D(tex2_32f_C1__, x+xoff+0.5f, y+yoff+0.5f);
    dst[y*stride + x] = diff*diff;
  }
}


// wrapper: compute MSE
IuStatus cuMse(const iu::ImageGpu_32f_C1* src, const iu::ImageGpu_32f_C1* reference, const IuRect& roi, double& mse)
{
  tex1_32f_C1__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C1__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C1__.filterMode = cudaFilterModeLinear;
  tex1_32f_C1__.normalized = false;

  tex2_32f_C1__.addressMode[0] = cudaAddressModeClamp;
  tex2_32f_C1__.addressMode[1] = cudaAddressModeClamp;
  tex2_32f_C1__.filterMode = cudaFilterModeLinear;
  tex2_32f_C1__.normalized = false;

  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float1>();
  cudaBindTexture2D(0, &tex1_32f_C1__, src->data(), &channel_desc, src->width(), src->height(), src->pitch());
  cudaBindTexture2D(0, &tex2_32f_C1__, reference->data(), &channel_desc, reference->width(), reference->height(), reference->pitch());

  const unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(roi.width - roi.x, dimBlock.x),
               iu::divUp(roi.height - roi.y, dimBlock.y));

  iu::ImageGpu_32f_C1 tmp(roi.width, roi.height);
  iuprivate::setValue(0.0f, &tmp, tmp.roi());

  cuMseKernel <<< dimGrid,dimBlock >>> (
                                        tmp.data(), tmp.stride(), roi.x, roi.y, roi.width, roi.height);

  double sum = 0.0;
  cuSummation(&tmp, tmp.roi(), sum);
  mse = sum/(static_cast<float>(roi.width*roi.height));

  return iu::checkCudaErrorState();
}


// kernel: compute SSIM
__global__ void cuSsimKernel(float c1, float c2, float* dst, size_t stride, int xoff, int yoff, int width, int height)
{
  // calculate absolute texture coordinates
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;

  if ((x < width) && (y < height))
  {
    int hkl = -4;
    int hkr = 4;

    // Calc means
    float mu_in = 0.0f;
    float mu_ref = 0.0f;
    float n = 0.0f;
    for (int dx=hkl; dx<=hkr; dx++)
    {
      for (int dy=hkl; dy<=hkr; dy++)
      {
        mu_in += tex2D(tex1_32f_C1__, x+dx+0.5f, y+dy+0.5f);
        mu_ref += tex2D(tex2_32f_C1__, x+dx+0.5f, y+dy+0.5f);
        n++;
      }
    }
    mu_in /= n;
    mu_ref /= n;

    // Calc variance and covariance
    float sigma_in = 0.0f;
    float sigma_ref = 0.0f;
    float cov = 0.0f;
    for (int dx=hkl; dx<=hkr; dx++)
    {
      for (int dy=hkl; dy<=hkr; dy++)
      {
        float in = tex2D(tex1_32f_C1__, x+dx+0.5f, y+dy+0.5f) - mu_in;
        float ref = tex2D(tex2_32f_C1__, x+dx+0.5f, y+dy+0.5f) - mu_ref;

        sigma_in  += in*in;
        sigma_ref += ref*ref;
        cov       += in*ref;
      }
    }
    sigma_in /= n-1.0f;
    sigma_ref /= n-1.0f;
    cov /= n-1.0f;

    // Calculate Structural similarity
    dst[y*stride + x] = (2.0f*mu_in*mu_ref + c1)*(2.0f*cov + c2)/((mu_in*mu_in + mu_ref*mu_ref + c1)*(sigma_in + sigma_ref + c2));
  }
}

// wrapper: compute SSIM
IuStatus cuSsim(const iu::ImageGpu_32f_C1* src, const iu::ImageGpu_32f_C1* reference, const IuRect& roi, double& ssim)
{
  tex1_32f_C1__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C1__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C1__.filterMode = cudaFilterModeLinear;
  tex1_32f_C1__.normalized = false;

  tex2_32f_C1__.addressMode[0] = cudaAddressModeClamp;
  tex2_32f_C1__.addressMode[1] = cudaAddressModeClamp;
  tex2_32f_C1__.filterMode = cudaFilterModeLinear;
  tex2_32f_C1__.normalized = false;

  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float1>();
  cudaBindTexture2D(0, &tex1_32f_C1__, src->data(), &channel_desc, src->width(), src->height(), src->pitch());
  cudaBindTexture2D(0, &tex2_32f_C1__, reference->data(), &channel_desc, reference->width(), reference->height(), reference->pitch());

  const unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(roi.width - roi.x, dimBlock.x),
               iu::divUp(roi.height - roi.y, dimBlock.y));

  iu::ImageGpu_32f_C1 tmp(roi.width, roi.height);
  iuprivate::setValue(0.0f, &tmp, tmp.roi());

  float k1 = 0.01f;
  float k2 = 0.03f;
  float dynamic_range = 1.0f;
  float c1 = (k1*dynamic_range)*(k1*dynamic_range);
  float c2 = (k2*dynamic_range)*(k2*dynamic_range);

  cuSsimKernel <<< dimGrid,dimBlock >>> (
                                         c1, c2, tmp.data(), tmp.stride(), roi.x, roi.y, roi.width, roi.height);

  double sum = 0.0;
  cuSummation(&tmp, tmp.roi(), sum);
  ssim = ssim/(static_cast<float>(roi.width*roi.height));

  return iu::checkCudaErrorState();
}


// kernel: color histogram
__global__ void  cuColorHistogramKernel(float* hist, int width, int height,
                                        int hstrideX, int hstrideXY, unsigned char mask_val)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  if(x<width && y<height)
  {
    if (tex2D(tex1_8u_C1__, x+0.5f, y+0.5f) == mask_val)
    {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 200
      uchar4 bins = tex2D(tex1_8u_C4__, x+0.5f, y+0.5f);
      int hc = bins.x + bins.y*hstrideX + bins.z*hstrideXY;
      atomicAdd(&hist[hc], 1.0f);
#else
  #if __CUDA_ARCH__ >= 120
        uchar4 bins = tex2D(tex1_8u_C4__, x+0.5f, y+0.5f);
        int hc = bins.x + bins.y*hstrideX + bins.z*hstrideXY;
        histogramAtomicAdd(&hist[hc], 1.0f);
  #else
    #if !WIN32
      #warning Color Histograms will not work: >= sm_12 needed!
	#endif
  #endif
#endif
    }
  }
}


// wrapper: color histogram
IuStatus cuColorHistogram(const iu::ImageGpu_8u_C4* binned_image, const iu::ImageGpu_8u_C1* mask,
                          iu::VolumeGpu_32f_C1* hist, unsigned char mask_val)
{
  tex1_8u_C4__.addressMode[0] = cudaAddressModeClamp;
  tex1_8u_C4__.addressMode[1] = cudaAddressModeClamp;
  tex1_8u_C4__.filterMode = cudaFilterModePoint;
  tex1_8u_C4__.normalized = false;

  tex1_8u_C1__.addressMode[0] = cudaAddressModeClamp;
  tex1_8u_C1__.addressMode[1] = cudaAddressModeClamp;
  tex1_8u_C1__.filterMode = cudaFilterModePoint;
  tex1_8u_C1__.normalized = false;

  setValue(0.0f, hist, hist->roi());

  cudaChannelFormatDesc channel_desc_c4 = cudaCreateChannelDesc<uchar4>();
  cudaBindTexture2D(0, &tex1_8u_C4__, binned_image->data(), &channel_desc_c4,
                    binned_image->width(), binned_image->height(), binned_image->pitch());
  cudaChannelFormatDesc channel_desc_c1 = cudaCreateChannelDesc<unsigned char>();
  cudaBindTexture2D(0, &tex1_8u_C1__, mask->data(), &channel_desc_c1, mask->width(),
                    mask->height(), mask->pitch());

  const unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(binned_image->width(), dimBlock.x),
               iu::divUp(binned_image->height(), dimBlock.y));

  cuColorHistogramKernel<<<dimGrid,dimBlock>>>(hist->data(), binned_image->width(),
                                               binned_image->height(), hist->stride(),
                                               hist->slice_stride(), mask_val);

  return iu::checkCudaErrorState();
}



} // namespace iuprivate

#endif // IUMATH_STATISTICS_CU

