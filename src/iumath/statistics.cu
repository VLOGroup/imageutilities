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
__global__ void cuMinMaxXKernel_8u_C1(Npp8u min[], Npp8u max[],
                                      int xoff, int yoff, int width, int height)
{
 const int x = blockIdx.x*blockDim.x + threadIdx.x;
 const int y = blockIdx.y*blockDim.y + threadIdx.y;

 float xx = x+xoff+0.5f;
 float yy = y+yoff+0.5f;

 Npp8u cur_min = tex2D(tex1_8u_C1__, xx, yy).x;
 Npp8u cur_max = tex2D(tex1_8u_C1__, xx, yy).x;

 // find minima of columns
 if (x<width)
 {
   Npp8u val;
   for(int y = 0; y < height; ++y)
   {
     yy = y+yoff+0.5f;
     val = tex2D(tex1_8u_C1__, xx, yy).x;
     if(val < cur_min) cur_min = val;
     if(val > cur_max) cur_max = val;
   }

   min[x] = cur_min;
   max[x] = cur_max;
 }
}

// kernel; find min/max; 8u_C4
__global__ void cuMinMaxXKernel_8u_C4(uchar4 min[], uchar4 max[],
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
__global__ void cuMinMaxXKernel_32f_C1(float min[], float max[],
                                       int xoff, int yoff, int width, int height)
{
 const int x = blockIdx.x*blockDim.x + threadIdx.x;
 const int y = blockIdx.y*blockDim.y + threadIdx.y;

 float xx = x+xoff+0.5f;
 float yy = y+yoff+0.5f;

 float cur_min = tex2D(tex1_32f_C1__, xx, yy).x;
 float cur_max = tex2D(tex1_32f_C1__, xx, yy).x;

 // find minima of columns
 if (x<width)
 {
   float val;
   for(int y = 0; y < height; ++y)
   {
     yy = y+yoff+0.5f;
     val = tex2D(tex1_32f_C1__, xx, yy).x;
     if(val < cur_min) cur_min = val;
     if(val > cur_max) cur_max = val;
   }

   min[x] = cur_min;
   max[x] = cur_max;
 }
}

// kernel; find min/max; 32f_C2
__global__ void cuMinMaxXKernel_32f_C2(float2 min[], float2 max[],
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
__global__ void cuMinMaxXKernel_32f_C4(float4 min[], float4 max[],
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
__global__ void cuSumColKernel_8u_C1(float sum[], int xoff, int yoff, int width, int height)
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
     cur_sum += tex2D(tex1_8u_C1__, xx, yy).x;
   }
   sum[x] = cur_sum;
 }
}

// kernel; compute sum; 32f_C1
__global__ void cuSumColKernel_32f_C1(float sum[], int xoff, int yoff, int width, int height)
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
     cur_sum += tex2D(tex1_32f_C1__, xx, yy).x;
   }
   sum[x] = cur_sum;
 }
}

/*
  KERNELS for NORM OF DIFFERENCES
*/

// kernel: compute L1 norm; |image1-image2|;
__global__ void  cuNormDiffL1Kernel(Npp32f* dst, size_t stride, int xoff, int yoff, int width, int height) {
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int oc = y*stride+x;

  x += xoff;
  y += yoff;

  float xx = x+0.5f;
  float yy = y+0.5f;

  if(x>=0 && y>= 0 && x<width && y<height)
  {
    dst[oc] = fabs(tex2D(tex1_32f_C1__, xx, yy).x - tex2D(tex2_32f_C1__, xx, yy).x);
  }
}

// kernel: compute L1 norm; |image-value|;
__global__ void  cuNormDiffValueL1Kernel(Npp32f value, Npp32f* dst, size_t stride, int xoff, int yoff, int width, int height)
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
    dst[oc] = fabs(tex2D(tex1_32f_C1__, xx, yy).x - value);
  }
}

// kernel: compute L2 norm; ||image1-image2||;
__global__ void  cuNormDiffL2Kernel(Npp32f* dst, size_t stride, int xoff, int yoff, int width, int height)
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
    dst[oc] = iu::sqr(tex2D(tex1_32f_C1__, xx, yy).x - tex2D(tex2_32f_C1__, xx, yy).x);
  }
}

// kernel: compute L2 norm; ||image-value||;
__global__ void  cuNormDiffValueL2Kernel(Npp32f value, Npp32f* dst, size_t stride, int xoff, int yoff, int width, int height)
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
    dst[oc] = iu::sqr(tex2D(tex1_32f_C1__, xx, yy).x - value);
  }
}

/******************************************************************************
  CUDA INTERFACES
*******************************************************************************/

/*
  WRAPPERS FOR MIN/MAX
*/

// wrapper: find min/max; 8u_C1
NppStatus cuMinMax(const iu::ImageNpp_8u_C1 *src, const IuRect &roi, Npp8u& min_C1, Npp8u& max_C1)
{
  NppStatus status;
  NppiSize npp_roi = {roi.width, roi.height};
  status = nppiMinMax_8u_C1R(src->data(roi.x, roi.y), static_cast<int>(src->pitch()), npp_roi, &min_C1, &max_C1 );
  return status;
/* TODO check timing between min/max kernels and npp stuff
  // prepare and bind texture
  tex1_8u_C1__.filterMode = cudaFilterModePoint;
  tex1_8u_C1__.addressMode[0] = cudaAddressModeClamp;
  tex1_8u_C1__.addressMode[1] = cudaAddressModeClamp;
  tex1_8u_C1__.normalized = false;
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar1>();
  cudaBindTexture2D(0, &tex1_8u_C1__, (uchar1*)src->data(), &channel_desc,
                    src->width(), src->height(), src->pitch());

  // fragmentation
  const unsigned int block_width = 512;
  dim3 dimBlock(block_width, 1, 1);
  dim3 dimGridX(iu::divUp(roi.width-roi.x, block_width), 1);

  // temporary memory for row sums on the host
  int num_row_sums = roi.width-roi.x;
  iu::ImageNpp_8u_C1 row_mins(num_row_sums, 1);
  iu::ImageNpp_8u_C1 row_maxs(num_row_sums, 1);

  cuMinMaxXKernel_8u_C1 <<< dimGridX, dimBlock >>> (
      row_mins.data(), row_maxs.data(), roi.x, roi.y, roi.width, roi.height);

  IU_CHECK_CUDA_ERRORS();

  iu::ImageCpu_8u_C1 h_row_mins(num_row_sums, 1);
  iu::ImageCpu_8u_C1 h_row_maxs(num_row_sums, 1);
  iuprivate::copy(&row_mins, row_mins.roi(), &h_row_mins, h_row_mins.roi());
  iuprivate::copy(&row_maxs, row_maxs.roi(), &h_row_maxs, h_row_maxs.roi());

  Npp8u* hb_row_mins = h_row_mins.data();
  Npp8u* hb_row_maxs= h_row_maxs.data();

  min_C1 = hb_row_mins[0];
  max_C1 = hb_row_maxs[0];

  for (int i = 1; i < num_row_sums; ++i)
  {
    min_C1 = min(min_C1, hb_row_mins[i]);
    max_C1 = max(max_C1, hb_row_maxs[i]);
  }

  cudaUnbindTexture(&tex1_8u_C1__);
  IU_CHECK_CUDA_ERRORS();
  return NPP_SUCCESS;
  */
}


// wrapper: find min/max; 8u_C4
NppStatus cuMinMax(const iu::ImageNpp_8u_C4 *src, const IuRect &roi, Npp8u min_C4[4], Npp8u max_C4[4])
{
  // nppiMinMax_8u_C4R seems to have a bug
//  NppStatus status;
//  NppiSize npp_roi = {roi.width, roi.height};
//  status = nppiMinMax_8u_C4R(src->data(roi.x, roi.y), src->pitch(), npp_roi, min_C4, max_C4 );
//  return status;

  // prepare and bind texture
  tex1_8u_C4__.filterMode = cudaFilterModePoint;
  tex1_8u_C4__.addressMode[0] = cudaAddressModeClamp;
  tex1_8u_C4__.addressMode[1] = cudaAddressModeClamp;
  tex1_8u_C4__.normalized = false;
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();
  cudaBindTexture2D(0, &tex1_8u_C4__, (uchar4*)src->data(), &channel_desc,
                    src->width(), src->height(), src->pitch());

  // fragmentation
  const unsigned int block_width = 512;
  dim3 dimBlock(block_width, 1, 1);
  dim3 dimGridX(iu::divUp(roi.width-roi.x, block_width), 1);

  // temporary memory for row sums on the host
  int num_row_sums = roi.width-roi.x;
  iu::ImageNpp_8u_C4 row_mins(num_row_sums, 1);
  iu::ImageNpp_8u_C4 row_maxs(num_row_sums, 1);

  cuMinMaxXKernel_8u_C4 <<< dimGridX, dimBlock >>> (
      (uchar4*)row_mins.data(), (uchar4*)row_maxs.data(),
      roi.x, roi.y, roi.width, roi.height);

  IU_CHECK_CUDA_ERRORS();

  iu::ImageCpu_8u_C4 h_row_mins(num_row_sums, 1);
  iu::ImageCpu_8u_C4 h_row_maxs(num_row_sums, 1);
  iuprivate::copy(&row_mins, &h_row_mins);
  iuprivate::copy(&row_maxs, &h_row_maxs);

  Npp8u* hb_row_mins = h_row_mins.data();
  Npp8u* hb_row_maxs= h_row_maxs.data();

  min_C4[0] = hb_row_mins[0];
  min_C4[1] = hb_row_mins[1];
  min_C4[2] = hb_row_mins[2];
  min_C4[3] = hb_row_mins[3];

  max_C4[0] = hb_row_maxs[0];
  max_C4[1] = hb_row_maxs[1];
  max_C4[2] = hb_row_maxs[2];
  max_C4[3] = hb_row_maxs[3];

  for (int i = 1; i < num_row_sums; ++i)
  {
    min_C4[0] = min(min_C4[0], hb_row_mins[i*4]);
    min_C4[1] = min(min_C4[0], hb_row_mins[i*4+1]);
    min_C4[2] = min(min_C4[0], hb_row_mins[i*4+2]);
    min_C4[3] = min(min_C4[0], hb_row_mins[i*4+3]);

    max_C4[0] = max(max_C4[0], hb_row_maxs[i*4]);
    max_C4[1] = max(max_C4[0], hb_row_maxs[i*4+1]);
    max_C4[2] = max(max_C4[0], hb_row_maxs[i*4+2]);
    max_C4[3] = max(max_C4[0], hb_row_maxs[i*4+3]);
  }

  cudaUnbindTexture(&tex1_8u_C4__);
  IU_CHECK_CUDA_ERRORS();
  return NPP_SUCCESS;
}

// wrapper: find min/max; 32f_C1
NppStatus cuMinMax(const iu::ImageNpp_32f_C1 *src, const IuRect &roi, Npp32f& min_C1, Npp32f& max_C1)
{
  // prepare and bind texture
  tex1_32f_C1__.filterMode = cudaFilterModePoint;
  tex1_32f_C1__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C1__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C1__.normalized = false;
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float1>();
  cudaBindTexture2D(0, &tex1_32f_C1__, (float1*)src->data(), &channel_desc,
                    src->width(), src->height(), src->pitch());

  // fragmentation
  const unsigned int block_width = 512;
  dim3 dimBlock(block_width, 1, 1);
  dim3 dimGridX(iu::divUp(roi.width-roi.x, block_width), 1);

  // temporary memory for row sums on the host
  int num_row_sums = roi.width-roi.x;
  iu::ImageNpp_32f_C1 row_mins(num_row_sums, 1);
  iu::ImageNpp_32f_C1 row_maxs(num_row_sums, 1);

  cuMinMaxXKernel_32f_C1 <<< dimGridX, dimBlock >>> (
      row_mins.data(), row_maxs.data(), roi.x, roi.y, roi.width, roi.height);

  IU_CHECK_CUDA_ERRORS();

  iu::ImageCpu_32f_C1 h_row_mins(num_row_sums, 1);
  iu::ImageCpu_32f_C1 h_row_maxs(num_row_sums, 1);
  iuprivate::copy(&row_mins, &h_row_mins);
  iuprivate::copy(&row_maxs, &h_row_maxs);

  float* hb_row_mins = h_row_mins.data();
  float* hb_row_maxs= h_row_maxs.data();

  min_C1 = hb_row_mins[0];
  max_C1 = hb_row_maxs[0];

  for (int i = 1; i < num_row_sums; ++i)
  {
    min_C1 = min(min_C1, hb_row_mins[i]);
    max_C1 = max(max_C1, hb_row_maxs[i]);
  }

  cudaUnbindTexture(&tex1_32f_C1__);
  IU_CHECK_CUDA_ERRORS();
  return NPP_SUCCESS;
}

// wrapper: find min/max; 32f_C2
NppStatus cuMinMax(const iu::ImageNpp_32f_C2 *src, const IuRect &roi, Npp32f min_C2[2], Npp32f max_C2[2])
{
  // prepare and bind texture
  tex1_32f_C2__.filterMode = cudaFilterModePoint;
  tex1_32f_C2__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C2__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C2__.normalized = false;
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float2>();
  cudaBindTexture2D(0, &tex1_32f_C2__, (float2*)src->data(), &channel_desc,
                    src->width(), src->height(), src->pitch());

  // fragmentation
  const unsigned int block_width = 512;
  dim3 dimBlock(block_width, 1, 1);
  dim3 dimGridX(iu::divUp(roi.width-roi.x, block_width), 1);

  // temporary memory for row sums on the host
  int num_row_sums = roi.width-roi.x;
  iu::ImageNpp_32f_C2 row_mins(num_row_sums, 1);
  iu::ImageNpp_32f_C2 row_maxs(num_row_sums, 1);

  cuMinMaxXKernel_32f_C2 <<< dimGridX, dimBlock >>> (
      (float2*)row_mins.data(), (float2*)row_maxs.data(),
      roi.x, roi.y, roi.width, roi.height);

  IU_CHECK_CUDA_ERRORS();

  iu::ImageCpu_32f_C2 h_row_mins(num_row_sums, 1);
  iu::ImageCpu_32f_C2 h_row_maxs(num_row_sums, 1);
  iuprivate::copy(&row_mins, &h_row_mins);
  iuprivate::copy(&row_maxs, &h_row_maxs);

  float* hb_row_mins = h_row_mins.data();
  float* hb_row_maxs= h_row_maxs.data();

  min_C2[0] = hb_row_mins[0];
  min_C2[1] = hb_row_mins[1];

  max_C2[0] = hb_row_maxs[0];
  max_C2[1] = hb_row_maxs[1];

  for (int i = 1; i < num_row_sums; ++i)
  {
    min_C2[0] = min(min_C2[0], hb_row_mins[i*2]);
    min_C2[1] = min(min_C2[0], hb_row_mins[i*2+1]);

    max_C2[0] = max(max_C2[0], hb_row_maxs[i*2]);
    max_C2[1] = max(max_C2[0], hb_row_maxs[i*2+1]);
  }

  cudaUnbindTexture(&tex1_32f_C2__);
  IU_CHECK_CUDA_ERRORS();
  return NPP_SUCCESS;
}

// wrapper: find min/max; 32f_C4
NppStatus cuMinMax(const iu::ImageNpp_32f_C4 *src, const IuRect &roi, Npp32f min_C4[4], Npp32f max_C4[4])
{
  // prepare and bind texture
  tex1_32f_C4__.filterMode = cudaFilterModePoint;
  tex1_32f_C4__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C4__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C4__.normalized = false;
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
  cudaBindTexture2D(0, &tex1_32f_C4__, (float4*)src->data(), &channel_desc,
                    src->width(), src->height(), src->pitch());

  // fragmentation
  const unsigned int block_width = 512;
  dim3 dimBlock(block_width, 1, 1);
  dim3 dimGridX(iu::divUp(roi.width-roi.x, block_width), 1);

  // temporary memory for row sums on the host
  int num_row_sums = roi.width-roi.x;
  iu::ImageNpp_32f_C4 row_mins(num_row_sums, 1);
  iu::ImageNpp_32f_C4 row_maxs(num_row_sums, 1);

  cuMinMaxXKernel_32f_C4 <<< dimGridX, dimBlock >>> (
      (float4*)row_mins.data(), (float4*)row_maxs.data(),
      roi.x, roi.y, roi.width, roi.height);

  IU_CHECK_CUDA_ERRORS();

  iu::ImageCpu_32f_C4 h_row_mins(num_row_sums, 1);
  iu::ImageCpu_32f_C4 h_row_maxs(num_row_sums, 1);
  iuprivate::copy(&row_mins, &h_row_mins);
  iuprivate::copy(&row_maxs, &h_row_maxs);

  float* hb_row_mins = h_row_mins.data();
  float* hb_row_maxs= h_row_maxs.data();

  min_C4[0] = hb_row_mins[0];
  min_C4[1] = hb_row_mins[1];
  min_C4[2] = hb_row_mins[2];
  min_C4[3] = hb_row_mins[3];

  max_C4[0] = hb_row_maxs[0];
  max_C4[1] = hb_row_maxs[1];
  max_C4[2] = hb_row_maxs[2];
  max_C4[3] = hb_row_maxs[3];

  for (int i = 1; i < num_row_sums; ++i)
  {
    min_C4[0] = min(min_C4[0], hb_row_mins[i*4]);
    min_C4[1] = min(min_C4[0], hb_row_mins[i*4+1]);
    min_C4[2] = min(min_C4[0], hb_row_mins[i*4+2]);
    min_C4[3] = min(min_C4[0], hb_row_mins[i*4+3]);

    max_C4[0] = max(max_C4[0], hb_row_maxs[i*4]);
    max_C4[1] = max(max_C4[0], hb_row_maxs[i*4+1]);
    max_C4[2] = max(max_C4[0], hb_row_maxs[i*4+2]);
    max_C4[3] = max(max_C4[0], hb_row_maxs[i*4+3]);
  }

  cudaUnbindTexture(&tex1_32f_C4__);
  IU_CHECK_CUDA_ERRORS();
  return NPP_SUCCESS;
}

/*
  WRAPPERS FOR SUM
*/

// wrapper: compute sum; 8u_C1
NppStatus cuSummation(const iu::ImageNpp_8u_C1 *src, const IuRect &roi, Npp64s& sum)
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
  dim3 dimGridX(iu::divUp(roi.width-roi.x, block_width), 1);

  // temporary memory for row sums on the host
  int num_col_sums = roi.width-roi.x;
  iu::ImageNpp_32f_C1 col_sums(num_col_sums, 1);

  cuSumColKernel_8u_C1 <<< dimGridX, dimBlock >>> (
      col_sums.data(), roi.x, roi.y, roi.width, roi.height);

  // :TODO: 32f vs 32u?
  iu::ImageCpu_32f_C1 h_col_sums(num_col_sums, 1);
  iuprivate::copy(&col_sums, &h_col_sums);

  float* hb_col_sums = h_col_sums.data();
  sum = 0;
  for (int i = 0; i < num_col_sums; ++i)
  {
    sum += static_cast<Npp32u>(hb_col_sums[i]);
  }

  cudaUnbindTexture(&tex1_8u_C1__);
  IU_CHECK_CUDA_ERRORS();
  return NPP_SUCCESS;
}

// wrapper: compute sum; 32f_C1
NppStatus cuSummation(const iu::ImageNpp_32f_C1 *src, const IuRect &roi, Npp64f& sum)
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
  dim3 dimGridX(iu::divUp(roi.width-roi.x, block_width), 1);

  // temporary memory for row sums on the host
  int num_col_sums = roi.width-roi.x;
  iu::ImageNpp_32f_C1 col_sums(num_col_sums, 1);

  cuSumColKernel_32f_C1 <<< dimGridX, dimBlock >>> (
      col_sums.data(), roi.x, roi.y, roi.width, roi.height);

  iu::ImageCpu_32f_C1 h_col_sums(num_col_sums, 1);
  iuprivate::copy(&col_sums, &h_col_sums);

  float* hb_col_sums = h_col_sums.data();
  sum = 0.0;
  for (int i = 0; i < num_col_sums; ++i)
  {
    sum += hb_col_sums[i];
  }

  cudaUnbindTexture(&tex1_32f_C1__);
  IU_CHECK_CUDA_ERRORS();
  return NPP_SUCCESS;
}

/*
  WRAPPERS for NORM OF DIFFERENCES
*/

// wrapper: compute L1 norm; |image1-image2|;
NppStatus cuNormDiffL1(const iu::ImageNpp_32f_C1* src1, const iu::ImageNpp_32f_C1* src2, const IuRect& roi, Npp64f& norm)
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

  iu::ImageNpp_32f_C1 squared_deviances(roi.width, roi.height);

  cuNormDiffL1Kernel <<< dimGrid, dimBlock >>> (
      squared_deviances.data(roi.x, roi.y), squared_deviances.stride(), roi.x, roi.y, roi.width, roi.height);

  Npp64f sum_squared = 0.0;
  iuprivate::cuSummation(&squared_deviances, roi, sum_squared);
  norm = sqrt(sum_squared);

  IU_CHECK_CUDA_ERRORS();
  return NPP_SUCCESS;
}

// wrapper: compute L1 norm; |image1-value|;
NppStatus cuNormDiffL1(const iu::ImageNpp_32f_C1* src, const Npp32f& value, const IuRect& roi, Npp64f& norm)
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

  iu::ImageNpp_32f_C1 squared_deviances(roi.width, roi.height);

  cuNormDiffValueL1Kernel <<< dimGrid, dimBlock >>> (
      value, squared_deviances.data(roi.x, roi.y), squared_deviances.stride(),
      roi.x, roi.y, roi.width, roi.height);

  Npp64f sum_squared = 0.0;
  iuprivate::cuSummation(&squared_deviances, roi, sum_squared);
  norm = sqrt(sum_squared);

  IU_CHECK_CUDA_ERRORS();
  return NPP_SUCCESS;
}

// wrapper: compute L2 norm; ||image1-image2||;
NppStatus cuNormDiffL2(const iu::ImageNpp_32f_C1* src1, const iu::ImageNpp_32f_C1* src2, const IuRect& roi, Npp64f& norm)
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

  iu::ImageNpp_32f_C1 squared_deviances(roi.width, roi.height);

  cuNormDiffL2Kernel <<< dimGrid, dimBlock >>> (
      squared_deviances.data(roi.x, roi.y), squared_deviances.stride(), roi.x, roi.y, roi.width, roi.height);

  Npp64f sum_squared = 0.0;
  iuprivate::cuSummation(&squared_deviances, roi, sum_squared);
  norm = sqrt(sum_squared);

  IU_CHECK_CUDA_ERRORS();
  return NPP_SUCCESS;
}

// wrapper: compute L2 norm; ||image1-value||;
NppStatus cuNormDiffL2(const iu::ImageNpp_32f_C1* src, const Npp32f& value, const IuRect& roi, Npp64f& norm)
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

  iu::ImageNpp_32f_C1 squared_deviances(roi.width, roi.height);

  cuNormDiffValueL2Kernel <<< dimGrid, dimBlock >>> (
      value, squared_deviances.data(roi.x, roi.y), squared_deviances.stride(),
      roi.x, roi.y, roi.width, roi.height);

  Npp64f sum_squared = 0.0;
  iuprivate::cuSummation(&squared_deviances, roi, sum_squared);
  norm = sqrt(sum_squared);

  IU_CHECK_CUDA_ERRORS();
  return NPP_SUCCESS;
}

/*
  WRAPPERS for ERROR MEASUREMENTS
*/


// kernel: compute MSE
__global__ void cuMseKernel(Npp32f* dst, size_t stride, int xoff, int yoff, int width, int height)
{
  // calculate absolute texture coordinates
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;

  if ((x < width) && (y < height))
  {
    float diff = tex2D(tex1_32f_C1__, x+xoff+0.5f, y+yoff+0.5f).x - tex2D(tex2_32f_C1__, x+xoff+0.5f, y+yoff+0.5f).x;
    dst[y*stride + x] = diff*diff;
  }
}


// wrapper: compute MSE
NppStatus cuMse(const iu::ImageNpp_32f_C1* src, const iu::ImageNpp_32f_C1* reference, const IuRect& roi, Npp64f& mse)
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

  iu::ImageNpp_32f_C1 tmp(roi.width, roi.height);
  iuprivate::setValue(0.0f, &tmp, tmp.roi());

  cuMseKernel <<< dimGrid,dimBlock >>> (
      tmp.data(), tmp.stride(), roi.x, roi.y, roi.width, roi.height);

  Npp64f sum = 0.0;
  cuSummation(&tmp, tmp.roi(), sum);
  mse = sum/(static_cast<Npp32f>(roi.width*roi.height));

  IU_CHECK_CUDA_ERRORS();
  return NPP_SUCCESS;
}


// kernel: compute SSIM
__global__ void cuSsimKernel(float c1, float c2, Npp32f* dst, size_t stride, int xoff, int yoff, int width, int height)
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
        mu_in += tex2D(tex1_32f_C1__, x+dx+0.5f, y+dy+0.5f).x;
        mu_ref += tex2D(tex2_32f_C1__, x+dx+0.5f, y+dy+0.5f).x;
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
        float in = tex2D(tex1_32f_C1__, x+dx+0.5f, y+dy+0.5f).x - mu_in;
        float ref = tex2D(tex2_32f_C1__, x+dx+0.5f, y+dy+0.5f).x - mu_ref;

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
NppStatus cuSsim(const iu::ImageNpp_32f_C1* src, const iu::ImageNpp_32f_C1* reference, const IuRect& roi, Npp64f& ssim)
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

  iu::ImageNpp_32f_C1 tmp(roi.width, roi.height);
  iuprivate::setValue(0.0f, &tmp, tmp.roi());

  float k1 = 0.01f;
  float k2 = 0.03f;
  float dynamic_range = 1.0f;
  float c1 = (k1*dynamic_range)*(k1*dynamic_range);
  float c2 = (k2*dynamic_range)*(k2*dynamic_range);

  cuSsimKernel <<< dimGrid,dimBlock >>> (
      c1, c2, tmp.data(), tmp.stride(), roi.x, roi.y, roi.width, roi.height);

  Npp64f sum = 0.0;
  cuSummation(&tmp, tmp.roi(), sum);
  ssim = ssim/(static_cast<Npp32f>(roi.width*roi.height));

  IU_CHECK_CUDA_ERRORS();
  return NPP_SUCCESS;
}

} // namespace iuprivate

#endif // IUMATH_STATISTICS_CU

