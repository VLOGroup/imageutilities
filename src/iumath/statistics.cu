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
#include <iucore/memory_modification.h>
#include <iucutil.h>
#include <iucore/iutextures.cuh>
#include "statistics.cuh"


namespace iuprivate {

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
   for(int y = 0; y < height; ++y)
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
    for(int y = 0; y < height; ++y)
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

 float xx = x+xoff+0.5f;
 float yy = y+yoff+0.5f;

 float cur_min = tex2D(tex1_32f_C1__, xx, yy);
 float cur_max = tex2D(tex1_32f_C1__, xx, yy);

 // find minima of columns
 if (x<width)
 {
   float val;
   for(int y = 0; y < height; ++y)
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
    for(int y = 0; y < height; ++y)
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
 if ((x+xoff)<width)
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

  cuMinMaxXKernel_8u_C1 <<< dimGridX, dimBlock >>> (
      row_mins.data(), row_maxs.data(), roi.x, roi.y, roi.width, roi.height);

  IU_CHECK_AND_RETURN_CUDA_ERRORS();

  iu::LinearHostMemory_8u_C1 h_row_mins(num_row_sums);
  iu::LinearHostMemory_8u_C1 h_row_maxs(num_row_sums);
  iuprivate::copy(&row_mins, &h_row_mins);
  iuprivate::copy(&row_maxs, &h_row_maxs);

  min_C1 = *h_row_mins.data();
  max_C1 = *h_row_maxs.data();

  for (int i = 1; i < num_row_sums; ++i)
  {
    min_C1 = min(min_C1, *h_row_mins.data(i));
    max_C1 = max(max_C1, *h_row_maxs.data(i));
  }

  cudaUnbindTexture(&tex1_8u_C1__);
  IU_CHECK_AND_RETURN_CUDA_ERRORS();
  return IU_SUCCESS;
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

  IU_CHECK_AND_RETURN_CUDA_ERRORS();

  iu::LinearHostMemory_8u_C4 h_row_mins(num_row_sums);
  iu::LinearHostMemory_8u_C4 h_row_maxs(num_row_sums);
  iuprivate::copy(&row_mins, &h_row_mins);
  iuprivate::copy(&row_maxs, &h_row_maxs);

  min_C4 = *h_row_mins.data(0);
  max_C4 = *h_row_maxs.data(0);

  for (int i = 1; i < num_row_sums; ++i)
  {
    min_C4.x = min(min_C4.x, h_row_mins.data(i)->x);
    min_C4.y = min(min_C4.y, h_row_mins.data(i)->y);
    min_C4.z = min(min_C4.z, h_row_mins.data(i)->z);
    min_C4.w = min(min_C4.w, h_row_mins.data(i)->w);

    max_C4.x = max(max_C4.x, h_row_maxs.data(i)->x);
    max_C4.y = max(max_C4.y, h_row_maxs.data(i)->y);
    max_C4.z = max(max_C4.z, h_row_maxs.data(i)->z);
    max_C4.w = max(max_C4.w, h_row_maxs.data(i)->w);
  }

  cudaUnbindTexture(&tex1_8u_C4__);
  IU_CHECK_AND_RETURN_CUDA_ERRORS();
  return IU_SUCCESS;
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

  IU_CHECK_AND_RETURN_CUDA_ERRORS();

  iu::LinearHostMemory_32f_C1 h_row_mins(num_row_sums);
  iu::LinearHostMemory_32f_C1 h_row_maxs(num_row_sums);
  iuprivate::copy(&row_mins, &h_row_mins);
  iuprivate::copy(&row_maxs, &h_row_maxs);

  min_C1 = *h_row_mins.data(0);
  max_C1 = *h_row_maxs.data(0);

  for (int i = 1; i < num_row_sums; ++i)
  {
    min_C1 = min(min_C1, *h_row_mins.data(i));
    max_C1 = max(max_C1, *h_row_maxs.data(i));
  }

  cudaUnbindTexture(&tex1_32f_C1__);
  IU_CHECK_AND_RETURN_CUDA_ERRORS();
  return IU_SUCCESS;
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

  IU_CHECK_AND_RETURN_CUDA_ERRORS();

  iu::LinearHostMemory_32f_C2 h_row_mins(num_row_sums);
  iu::LinearHostMemory_32f_C2 h_row_maxs(num_row_sums);
  iuprivate::copy(&row_mins, &h_row_mins);
  iuprivate::copy(&row_maxs, &h_row_maxs);

  min_C2 = *h_row_mins.data(0);
  max_C2 = *h_row_maxs.data(0);

  for (int i = 1; i < num_row_sums; ++i)
  {
    min_C2.x = min(min_C2.x, h_row_mins.data(i)->x);
    min_C2.y = min(min_C2.y, h_row_mins.data(i)->y);

    max_C2.x = max(max_C2.x, h_row_maxs.data(i)->x);
    max_C2.y = max(max_C2.y, h_row_maxs.data(i)->y);
  }

  cudaUnbindTexture(&tex1_32f_C2__);
  IU_CHECK_AND_RETURN_CUDA_ERRORS();
  return IU_SUCCESS;
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

  IU_CHECK_AND_RETURN_CUDA_ERRORS();

  iu::LinearHostMemory_32f_C4 h_row_mins(num_cols);
  iu::LinearHostMemory_32f_C4 h_row_maxs(num_cols);
  iuprivate::copy(&row_mins, &h_row_mins);
  iuprivate::copy(&row_maxs, &h_row_maxs);

  min_C4 = *h_row_mins.data(0);
  max_C4 = *h_row_maxs.data(0);

  for (int i = 1; i < num_cols; ++i)
  {
    min_C4.x = min(min_C4.x, h_row_mins.data(i)->x);
    min_C4.y = min(min_C4.y, h_row_mins.data(i)->y);
    min_C4.z = min(min_C4.z, h_row_mins.data(i)->z);
    min_C4.w = min(min_C4.w, h_row_mins.data(i)->w);

    max_C4.x = max(max_C4.x, h_row_maxs.data(i)->x);
    max_C4.y = max(max_C4.y, h_row_maxs.data(i)->y);
    max_C4.z = max(max_C4.z, h_row_maxs.data(i)->z);
    max_C4.w = max(max_C4.w, h_row_maxs.data(i)->w);
  }

  cudaUnbindTexture(&tex1_32f_C4__);
  IU_CHECK_AND_RETURN_CUDA_ERRORS();
  return IU_SUCCESS;
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
  IU_CHECK_AND_RETURN_CUDA_ERRORS();
  return IU_SUCCESS;
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
  IU_CHECK_AND_RETURN_CUDA_ERRORS();
  return IU_SUCCESS;
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
  norm = sqrt(sum_squared);

  IU_CHECK_AND_RETURN_CUDA_ERRORS();
  return IU_SUCCESS;
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
  norm = sqrt(sum_squared);

  IU_CHECK_AND_RETURN_CUDA_ERRORS();
  return IU_SUCCESS;
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
  norm = sqrt(sum_squared);

  IU_CHECK_AND_RETURN_CUDA_ERRORS();
  return IU_SUCCESS;
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
  norm = sqrt(sum_squared);

  IU_CHECK_AND_RETURN_CUDA_ERRORS();
  return IU_SUCCESS;
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

  IU_CHECK_AND_RETURN_CUDA_ERRORS();
  return IU_SUCCESS;
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

  IU_CHECK_AND_RETURN_CUDA_ERRORS();
  return IU_SUCCESS;
}

} // namespace iuprivate

#endif // IUMATH_STATISTICS_CU

