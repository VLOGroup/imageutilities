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
 * Language    : CUDA
 * Description : Implementation of Cuda wrappers for arithmetic functions
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUMATH_ARITHMETIC_CU
#define IUMATH_ARITHMETIC_CU

#include <iucore/iutextures.cuh>
#include <iucutil.h>
#include "arithmetic.cuh"

namespace iuprivate {

/* ****************************************************************************
 *  weighted add
 * ****************************************************************************/

// kernel: weighted add; 32-bit;
__global__ void cuAddWeightedKernel_32f_C1(
    const float weight1, const float weight2, float* dst, const size_t stride,
    const int xoff, const int yoff, const int width, const int height)
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
    dst[oc] = weight1*tex2D(tex1_32f_C1__, xx, yy) +
              weight2*tex2D(tex2_32f_C1__, xx, yy);
  }
}

// wrapper: weighted add; 32-bit;
IuStatus cuAddWeighted(const iu::ImageGpu_32f_C1* src1, const float& weight1,
                       const iu::ImageGpu_32f_C1* src2, const float& weight2,
                       iu::ImageGpu_32f_C1* dst, const IuRect& roi)
{
  // bind textures
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
  cudaBindTexture2D(0, &tex1_32f_C1__, src1->data(), &channel_desc, src1->width(), src1->height(), src1->pitch());
  cudaBindTexture2D(0, &tex2_32f_C1__, src2->data(), &channel_desc, src2->width(), src2->height(), src2->pitch());

  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(dst->width(), dimBlock.x),
               iu::divUp(dst->height(), dimBlock.y));

  cuAddWeightedKernel_32f_C1 <<< dimGrid, dimBlock >>> (
      weight1, weight2, dst->data(roi.x, roi.y), dst->stride(),
      roi.x, roi.y, roi.width, roi.height);

  // unbind textures
  cudaUnbindTexture(&tex1_32f_C1__);
  cudaUnbindTexture(&tex2_32f_C1__);

  // error check
  IU_CHECK_AND_RETURN_CUDA_ERRORS();
}

/******************************************************************************
  multiplication with factor
*******************************************************************************/

// kernel: multiplication with factor; 8-bit; 1-channel
__global__ void  cuMulCKernel(const unsigned char factor, unsigned char* dst, const size_t stride,
                              const int xoff, const int yoff,
                              const int width, const int height)
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
    unsigned char val = tex2D(tex1_8u_C1__, xx, yy);
    dst[oc] = val * factor;
  }
}

// wrapper: multiplication with factor; 8-bit; 1-channel
IuStatus cuMulC(const iu::ImageGpu_8u_C1* src, const unsigned char& factor, iu::ImageGpu_8u_C1* dst, const IuRect& roi)
{
  // bind textures
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar1>();
  cudaBindTexture2D(0, &tex1_8u_C1__, src->data(), &channel_desc, src->width(), src->height(), src->pitch());

  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(dst->width(), dimBlock.x),
               iu::divUp(dst->height(), dimBlock.y));

  cuMulCKernel <<< dimGrid, dimBlock >>> (
    factor, dst->data(roi.x, roi.y), dst->stride(), roi.x, roi.y, roi.width, roi.height);

  // unbind textures
  cudaUnbindTexture(&tex1_8u_C1__);

  // error check
  IU_CHECK_AND_RETURN_CUDA_ERRORS();
}

// kernel: multiplication with factor; 8-bit; 4-channel
__global__ void  cuMulCKernel(const uchar4 factor, uchar4* dst, const size_t stride,
                              const int xoff, const int yoff,
                              const int width, const int height)
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
    uchar4 val = tex2D(tex1_8u_C4__, xx, yy);
    dst[oc] = val * factor;
  }
}

// wrapper: multiplication with factor; 8-bit; 4-channel
IuStatus cuMulC(const iu::ImageGpu_8u_C4* src, const uchar4& factor, iu::ImageGpu_8u_C4* dst, const IuRect& roi)
{
  // bind textures
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();
  cudaBindTexture2D(0, &tex1_8u_C4__, (uchar4*)src->data(), &channel_desc, src->width(), src->height(), src->pitch());

  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(dst->width(), dimBlock.x),
               iu::divUp(dst->height(), dimBlock.y));

  cuMulCKernel <<< dimGrid, dimBlock >>> (
    factor, dst->data(roi.x, roi.y), dst->stride(),
    roi.x, roi.y, roi.width, roi.height);


  // unbind textures
  cudaUnbindTexture(&tex1_8u_C4__);

  // error check
  IU_CHECK_AND_RETURN_CUDA_ERRORS();
}

// kernel: multiplication with factor; 32-bit; 1-channel
__global__ void  cuMulCKernel(const float factor, float* dst, const size_t stride,
                              const int xoff, const int yoff,
                              const int width, const int height)
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
    float val = tex2D(tex1_32f_C1__, xx, yy);
    dst[oc] = val * factor;
  }
}

// wrapper: multiplication with factor; 32-bit; 1-channel
IuStatus cuMulC(const iu::ImageGpu_32f_C1* src, const float& factor, iu::ImageGpu_32f_C1* dst, const IuRect& roi)
{
  // bind textures
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
  cudaBindTexture2D(0, &tex1_32f_C1__, src->data(), &channel_desc, src->width(), src->height(), src->pitch());

  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(dst->width(), dimBlock.x),
               iu::divUp(dst->height(), dimBlock.y));

  cuMulCKernel <<< dimGrid, dimBlock >>> (
    factor, dst->data(roi.x, roi.y), dst->stride(),
    roi.x, roi.y, roi.width, roi.height);


  // unbind textures
  cudaUnbindTexture(&tex1_32f_C1__);

  // error check
  IU_CHECK_AND_RETURN_CUDA_ERRORS();
}

// kernel: volume multiplication with factor; 32-bit; 1-channel
__global__ void  cuVolMulCKernel(const float factor, float* dst, const float*src,
                                 const size_t stride, const size_t slice_stride,
                                 const int width, const int height, const int depth)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int oc = y*stride+x;

  if(x<width && y<height)
  {
    for (int z=0; z<depth; z++)
    {
      int vc = oc + z*slice_stride;
      dst[vc] = src[vc] * factor;
    }
  }
}

// wrapper: volume multiplication with factor; 32-bit; 1-channel
IuStatus cuMulC(const iu::VolumeGpu_32f_C1* src, const float& factor,
                iu::VolumeGpu_32f_C1* dst)
{
  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(dst->width(), dimBlock.x),
               iu::divUp(dst->height(), dimBlock.y));

  cuVolMulCKernel<<<dimGrid, dimBlock>>>(factor, dst->data(), src->data(),
    dst->stride(), dst->slice_stride(), dst->width(), dst->height(), dst->depth());

  // error check
  IU_CHECK_AND_RETURN_CUDA_ERRORS();
}

// kernel: multiplication with factor; 32-bit; 2-channel
__global__ void  cuMulCKernel(const float2 factor, float2* dst, const size_t stride,
                              const int xoff, const int yoff,
                              const int width, const int height)
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
    float2 val = tex2D(tex1_32f_C2__, xx, yy);
    dst[oc] = val * factor;
  }
}

// wrapper: multiplication with factor; 32-bit; 4-channel
IuStatus cuMulC(const iu::ImageGpu_32f_C2* src, const float2& factor, iu::ImageGpu_32f_C2* dst, const IuRect& roi)
{
  // bind textures
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float2>();
  cudaBindTexture2D(0, &tex1_32f_C2__, src->data(), &channel_desc, src->width(), src->height(), src->pitch());

  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(dst->width(), dimBlock.x),
               iu::divUp(dst->height(), dimBlock.y));

  cuMulCKernel <<< dimGrid, dimBlock >>> (
    factor, dst->data(roi.x, roi.y), dst->stride(),
    roi.x, roi.y, roi.width, roi.height);


  // unbind textures
  cudaUnbindTexture(&tex1_32f_C2__);

  // error check
  IU_CHECK_AND_RETURN_CUDA_ERRORS();
}


// kernel: multiplication with factor; 32-bit; 4-channel
__global__ void  cuMulCKernel(const float4 factor, float4* dst, const size_t stride,
                              const int xoff, const int yoff,
                              const int width, const int height)
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
    float4 val = tex2D(tex1_32f_C4__, xx, yy);
    dst[oc] = val * factor;
  }
}

// wrapper: multiplication with factor; 32-bit; 4-channel
IuStatus cuMulC(const iu::ImageGpu_32f_C4* src, const float4& factor, iu::ImageGpu_32f_C4* dst, const IuRect& roi)
{
  // bind textures
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
  cudaBindTexture2D(0, &tex1_32f_C4__, (float4*)src->data(), &channel_desc, src->width(), src->height(), src->pitch());

  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(dst->width(), dimBlock.x),
               iu::divUp(dst->height(), dimBlock.y));

  cuMulCKernel <<< dimGrid, dimBlock >>> (
    factor, dst->data(roi.x, roi.y), dst->stride(),
    roi.x, roi.y, roi.width, roi.height);


  // unbind textures
  cudaUnbindTexture(&tex1_32f_C4__);

  // error check
  IU_CHECK_AND_RETURN_CUDA_ERRORS();
}


/******************************************************************************
  add val
*******************************************************************************/

// kernel: add val; 8-bit; 1-channel
__global__ void  cuAddCKernel(const unsigned char val, unsigned char* dst, const size_t stride,
                              const int xoff, const int yoff,
                              const int width, const int height)
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
    dst[oc] = val + tex2D(tex1_8u_C1__, xx, yy);
  }
}

// wrapper: add val; 8-bit; 1-channel
IuStatus cuAddC(const iu::ImageGpu_8u_C1* src, const unsigned char& val, iu::ImageGpu_8u_C1* dst, const IuRect& roi)
{
  // bind textures
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar1>();
  cudaBindTexture2D(0, &tex1_8u_C1__, src->data(), &channel_desc, src->width(), src->height(), src->pitch());

  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(dst->width(), dimBlock.x),
               iu::divUp(dst->height(), dimBlock.y));

  cuAddCKernel <<< dimGrid, dimBlock >>> (
    val, dst->data(roi.x, roi.y), dst->stride(), roi.x, roi.y, roi.width, roi.height);

  // unbind textures
  cudaUnbindTexture(&tex1_8u_C1__);

  // error check
  IU_CHECK_AND_RETURN_CUDA_ERRORS();
}

// kernel: add val; 8-bit; 4-channel
__global__ void  cuAddCKernel(const uchar4 val, uchar4* dst, const size_t stride,
                              const int xoff, const int yoff,
                              const int width, const int height)
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
    uchar4 value = tex2D(tex1_8u_C4__, xx, yy);
    value.x = value.x + val.x;
    value.y = value.y + val.y;
    value.z = value.z + val.z;
    value.w = value.w + val.w;
    dst[oc] = value;
  }
}

// wrapper: add val; 8-bit; 4-channel
IuStatus cuAddC(const iu::ImageGpu_8u_C4* src, const uchar4& val, iu::ImageGpu_8u_C4* dst, const IuRect& roi)
{
  // bind textures
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();
  cudaBindTexture2D(0, &tex1_8u_C4__, (uchar4*)src->data(), &channel_desc, src->width(), src->height(), src->pitch());

  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(dst->width(), dimBlock.x),
               iu::divUp(dst->height(), dimBlock.y));

  cuAddCKernel <<< dimGrid, dimBlock >>> (
    val, dst->data(roi.x, roi.y), dst->stride(),
    roi.x, roi.y, roi.width, roi.height);


  // unbind textures
  cudaUnbindTexture(&tex1_8u_C4__);

  // error check
  IU_CHECK_AND_RETURN_CUDA_ERRORS();
}

// kernel: add val; 32-bit; 1-channel
__global__ void  cuAddCKernel(const float val, float* dst, const size_t stride,
                              const int xoff, const int yoff,
                              const int width, const int height)
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
    dst[oc] = val + tex2D(tex1_32f_C1__, xx, yy);
  }
}

// wrapper: add val; 32-bit; 1-channel
IuStatus cuAddC(const iu::ImageGpu_32f_C1* src, const float& val, iu::ImageGpu_32f_C1* dst, const IuRect& roi)
{
  // bind textures
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
  cudaBindTexture2D(0, &tex1_32f_C1__, src->data(), &channel_desc, src->width(), src->height(), src->pitch());

  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(dst->width(), dimBlock.x),
               iu::divUp(dst->height(), dimBlock.y));

  cuAddCKernel <<< dimGrid, dimBlock >>> (
    val, dst->data(roi.x, roi.y), dst->stride(),
    roi.x, roi.y, roi.width, roi.height);


  // unbind textures
  cudaUnbindTexture(&tex1_32f_C1__);

  // error check
  IU_CHECK_AND_RETURN_CUDA_ERRORS();
}

// kernel: add val; 32-bit; 2-channel
__global__ void  cuAddCKernel(const float2 val, float2* dst, const size_t stride,
                              const int xoff, const int yoff,
                              const int width, const int height)
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
    dst[oc] = val + tex2D(tex1_32f_C2__, xx, yy);
  }
}

// wrapper: add val; 32-bit; 4-channel
IuStatus cuAddC(const iu::ImageGpu_32f_C2* src, const float2& val, iu::ImageGpu_32f_C2* dst, const IuRect& roi)
{
  // bind textures
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float2>();
  cudaBindTexture2D(0, &tex1_32f_C2__, src->data(), &channel_desc, src->width(), src->height(), src->pitch());

  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(dst->width(), dimBlock.x),
               iu::divUp(dst->height(), dimBlock.y));

  cuAddCKernel <<< dimGrid, dimBlock >>> (
    val, dst->data(roi.x, roi.y), dst->stride(),
    roi.x, roi.y, roi.width, roi.height);


  // unbind textures
  cudaUnbindTexture(&tex1_32f_C2__);

  // error check
  IU_CHECK_AND_RETURN_CUDA_ERRORS();
}


// kernel: add val; 32-bit; 1-channel
__global__ void  cuAddCKernel(const float4 val, float4* dst, const size_t stride,
                              const int xoff, const int yoff,
                              const int width, const int height)
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
    dst[oc] = val + tex2D(tex1_32f_C4__, xx, yy);
  }
}

// wrapper: add val; 32-bit; 4-channel
IuStatus cuAddC(const iu::ImageGpu_32f_C4* src, const float4& val, iu::ImageGpu_32f_C4* dst, const IuRect& roi)
{
  // bind textures
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
  cudaBindTexture2D(0, &tex1_32f_C4__, (float4*)src->data(), &channel_desc, src->width(), src->height(), src->pitch());

  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(dst->width(), dimBlock.x),
               iu::divUp(dst->height(), dimBlock.y));

  cuAddCKernel <<< dimGrid, dimBlock >>> (
    val, dst->data(roi.x, roi.y), dst->stride(),
    roi.x, roi.y, roi.width, roi.height);


  // unbind textures
  cudaUnbindTexture(&tex1_32f_C4__);

  // error check
  IU_CHECK_AND_RETURN_CUDA_ERRORS();
}

} // namespace iuprivate

#endif // IUMATH_ARITHMETIC_CU

