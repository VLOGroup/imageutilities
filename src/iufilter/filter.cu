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
 * Module      : Filter
 * Class       : none
 * Language    : CUDA
 * Description : Definition of CUDA wrappers for filter functions on Npp images
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUPRIVATE_FILTER_CU
#define IUPRIVATE_FILTER_CU

#include <float.h>
#include <iucutil.h>
#include <iucore/iutextures.cuh>
#include <iucore/copy.h>
#include <iucore/setvalue.h>

#include "filterbspline_kernels.cu"

#include "filter.cuh"

namespace iuprivate {

// ----------------------------------------------------------------------------
// kernel: median filter; 32-bit; 1-channel
__global__ void  cuFilterMedian3x3Kernel_32f_C1(float* dst, const size_t stride,
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

  // shared stuff
  const int tx = threadIdx.x+1;
  const int ty = threadIdx.y+1;
  // we have a 3x3 kernel, so our width of the shared memory (shp) is blockDim.x + 2!
  const int shp = blockDim.x + 2;
  const int shc = (threadIdx.y+1) * shp + (threadIdx.x+1);
  extern __shared__ float sh_in[];

  if(x>=0 && y>= 0 && x<width && y<height)
  {
    // Load input 3x3 block into shared memory
    {
      // for each thread: copy the data of the current input position to shared mem
      sh_in[shc] = tex2D(tex1_32f_C1__, xx, yy);

      // Note: the FLT_MAX prevents us from overemphasizing the border pixels if they are outliers!

      /////////////////////////////////////////////////////////////////////////////
      // boundary conditions
      /////////////////////////////////////////////////////////////////////////////
      if (x == 0) // at left image border
      {
        if (y == 0)
          sh_in[shc-shp-1] = FLT_MAX; // left-upper corner (image)
        else if (ty == 1)
          sh_in[shc-shp-1] = tex2D(tex1_32f_C1__, xx, yy-1.0f); // left-upper corner (block)

        sh_in[shc-1] = sh_in[shc];     // left border (image)

        if (y == height-1)
          sh_in[shc+shp-1] = FLT_MAX; // left-lower corner (image)
        else if (ty == blockDim.y)
          sh_in[shc+shp-1] = tex2D(tex1_32f_C1__, xx, yy+1.0f); // left-lower corner (block)
      }
      else if (tx == 1) // at left block border (inside image w.r.t x)
      {
        if (y == 0)
          sh_in[shc-shp-1] = tex2D(tex1_32f_C1__, xx-1.0f, yy); // left-upper corner (block, outside)
        else if (ty == 1)
          sh_in[shc-shp-1] = tex2D(tex1_32f_C1__, xx-1.0f, yy-1.0f); // left-upper corner (block, inside)

        sh_in[shc-1] = tex2D(tex1_32f_C1__, xx-1.0f, yy); // left border (block)

        if (y == height-1)
          sh_in[shc+shp-1] = tex2D(tex1_32f_C1__, xx-1.0f, yy); // left-lower corner (block, outside)
        else if (ty == blockDim.y)
          sh_in[shc+shp-1] = tex2D(tex1_32f_C1__, xx-1.0f, yy+1.0f); // left-lower corner (block, inside)
      }


      if (x == width-1) // at right image border
      {
        if (y == 0)
          sh_in[shc-shp+1] = FLT_MAX; // right-upper corner (image)
        else if (ty == 1)
          sh_in[shc-shp+1] = tex2D(tex1_32f_C1__, xx, yy-1.0f); // right-upper corner (block)

        sh_in[shc+1] = sh_in[shc]; // right border (image)

        if (y == height-1)
          sh_in[shc+shp+1] = FLT_MAX; // right-lower corner (image)
        else if (ty == blockDim.y)
          sh_in[shc+shp+1] = tex2D(tex1_32f_C1__, xx, yy+1.0f); // right-lower corner (block)
      }
      else if (tx == blockDim.x) // at right block border (inside image w.r.t x)
      {
        if (y == 0)
          sh_in[shc-shp+1] = tex2D(tex1_32f_C1__, xx+1.0f, yy); // right-upper corner (block, outside)
        else if (ty == 1)
          sh_in[shc-shp+1] = tex2D(tex1_32f_C1__, xx+1.0f, yy-1.0f); // right-upper corner (block, inside)

        sh_in[shc+1] = tex2D(tex1_32f_C1__, xx+1.0f, yy); // right border (block)

        if (y == height-1)
          sh_in[shc+shp+1] = tex2D(tex1_32f_C1__, xx+1.0f, yy); // right-lower corner (block, outside)
        else if (ty == blockDim.y)
          sh_in[shc+shp+1] = tex2D(tex1_32f_C1__, xx+1.0f, yy+1.0f); // right-lower corner (block, inside)
      }

      if (y == 0)
        sh_in[shc-shp] = sh_in[shc]; // upper border (image)
      else if (ty == 1)
        sh_in[shc-shp] = tex2D(tex1_32f_C1__, xx, yy-1.0f); // upper border (block)

      if (y == height-1)
        sh_in[shc+shp] = sh_in[shc]; // lower border (image)
      else if (ty == blockDim.y)
        sh_in[shc+shp] = tex2D(tex1_32f_C1__, xx, yy+1.0f); // lower border (block)

      __syncthreads();
    }

    // in a sequence of nine elements, we have to remove four times the maximum from the sequence and need
    // a fifth calculated maximum which is the median!

    float maximum;
    {
      float vals[8];

      // first 'loop'
      vals[0] = fmin(sh_in[shc-shp-1], sh_in[shc-shp]);
      maximum = fmax(sh_in[shc-shp-1], sh_in[shc-shp]);
      vals[1] = fmin(maximum, sh_in[shc-shp+1]);
      maximum = fmax(maximum, sh_in[shc-shp+1]);
      vals[2] = fmin(maximum, sh_in[shc-1]);
      maximum = fmax(maximum, sh_in[shc-1]);
      vals[3] = fmin(maximum, sh_in[shc]);
      maximum = fmax(maximum, sh_in[shc]);
      vals[4] = fmin(maximum, sh_in[shc+1]);
      maximum = fmax(maximum, sh_in[shc+1]);
      vals[5] = fmin(maximum, sh_in[shc+shp-1]);
      maximum = fmax(maximum, sh_in[shc+shp-1]);
      vals[6] = fmin(maximum, sh_in[shc+shp]);
      maximum = fmax(maximum, sh_in[shc+shp]);
      vals[7] = fmin(maximum, sh_in[shc+shp+1]);
      maximum = fmax(maximum, sh_in[shc+shp+1]);

      // second 'loop'
      maximum = fmax(vals[0], vals[1]);
      vals[0] = fmin(vals[0], vals[1]);
      vals[1] = maximum;
      maximum = fmax(vals[1], vals[2]);
      vals[1] = fmin(vals[1], vals[2]);
      vals[2] = maximum;
      maximum = fmax(vals[2], vals[3]);
      vals[2] = fmin(vals[2], vals[3]);
      vals[3] = maximum;
      maximum = fmax(vals[3], vals[4]);
      vals[3] = fmin(vals[3], vals[4]);
      vals[4] = maximum;
      maximum = fmax(vals[4], vals[5]);
      vals[4] = fmin(vals[4], vals[5]);
      vals[5] = maximum;
      maximum = fmax(vals[5], vals[6]);
      vals[5] = fmin(vals[5], vals[6]);
      vals[6] = fmin(maximum, vals[7]);

      // third 'loop'
      maximum = fmax(vals[0], vals[1]);
      vals[0] = fmin(vals[0], vals[1]);
      vals[1] = maximum;
      maximum = fmax(vals[1], vals[2]);
      vals[1] = fmin(vals[1], vals[2]);
      vals[2] = maximum;
      maximum = fmax(vals[2], vals[3]);
      vals[2] = fmin(vals[2], vals[3]);
      vals[3] = maximum;
      maximum = fmax(vals[3], vals[4]);
      vals[3] = fmin(vals[3], vals[4]);
      vals[4] = maximum;
      maximum = fmax(vals[4], vals[5]);
      vals[4] = fmin(vals[4], vals[5]);
      vals[5] = fmin(maximum, vals[6]);

      // 4th 'loop'
      maximum = fmax(vals[0], vals[1]);
      vals[0] = fmin(vals[0], vals[1]);
      vals[1] = maximum;
      maximum = fmax(vals[1], vals[2]);
      vals[1] = fmin(vals[1], vals[2]);
      vals[2] = maximum;
      maximum = fmax(vals[2], vals[3]);
      vals[2] = fmin(vals[2], vals[3]);
      vals[3] = maximum;
      maximum = fmax(vals[3], vals[4]);
      vals[3] = fmin(vals[3], vals[4]);
      vals[4] = fmin(maximum, vals[5]);

      // 5th 'loop'
      maximum = fmax(vals[0], vals[1]);
      maximum = fmax(maximum, vals[2]);
      maximum = fmax(maximum, vals[3]);
      maximum = fmax(maximum, vals[4]);
    }
    dst[oc] = maximum;
  }
}

// ----------------------------------------------------------------------------
// wrapper: median filter; 32-bit; 1-channel
IuStatus cuFilterMedian3x3(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst, const IuRect& roi)
{
  // bind textures
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
  tex1_32f_C1__.filterMode = cudaFilterModeLinear;
  tex1_32f_C1__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C1__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C1__.normalized = false;
  cudaBindTexture2D(0, &tex1_32f_C1__, src->data(), &channel_desc, src->width(), src->height(), src->pitch());

  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(roi.width, dimBlock.x), iu::divUp(roi.height, dimBlock.y));

  size_t shared_size = (block_size+2)*(block_size+2)*sizeof(float);

  cuFilterMedian3x3Kernel_32f_C1 <<< dimGrid, dimBlock, shared_size >>> (
                                                                         dst->data(roi.x, roi.y), dst->stride(), roi.x, roi.y, roi.width, roi.height);

  // unbind textures
  cudaUnbindTexture(&tex1_32f_C1__);

  // error check
  return iu::checkCudaErrorState();
}

// ----------------------------------------------------------------------------
// kernel: Gaussian filter; 32-bit; 1-channel
/** Perform a convolution with an gaussian smoothing kernel
 * @param dst          pointer to output image (linear memory)
 * @param stride       length of image row [pixels]
 * @param xoff         x-coordinate offset where to start the region [pixels]
 * @param yoff         y-coordinate offset where to start the region [pixels]
 * @param width        width of region [pixels]
 * @param height       height of region [pixels]
 * @param sigma        sigma of the smoothing kernel
 * @param kernel_size  lenght of the smoothing kernel [pixels]
 * @param horizontal   defines the direction of convolution
 */
__global__ void cuFilterGaussKernel_32f_C1(float* dst, const size_t stride,
                                           const int xoff, const int yoff,
                                           const int width, const int height,
                                           float sigma, int kernel_size, bool horizontal=true)
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
    float sum = 0.0f;
    int half_kernel_elements = (kernel_size - 1) / 2;

    if (horizontal)
    {
      // convolve horizontally
      float g0 = 1.0f / (sqrtf(2.0f * 3.141592653589793f) * sigma);
      float g1 = exp(-0.5f / (sigma * sigma));
      float g2 = g1 * g1;
      sum = g0 * tex2D(tex1_32f_C1__, xx, yy);
      float sum_coeff = g0;
      for (int i = 1; i <= half_kernel_elements; i++)
      {
        g0 *= g1;
        g1 *= g2;
        float cur_xx = IUMAX(0.5f, IUMIN(width-0.5f, xx + i));
        sum += g0 * tex2D(tex1_32f_C1__, cur_xx, yy);
        cur_xx = IUMAX(0.5f, IUMIN(width-0.5f, xx-i));
        sum += g0 * tex2D(tex1_32f_C1__, cur_xx, yy);
        sum_coeff += 2.0f*g0;
      }
      dst[oc] = sum/sum_coeff;
    }
    else
    {
      // convolve vertically
      float g0 = 1.0f / (sqrtf(2.0f * 3.141592653589793f) * sigma);
      float g1 = exp(-0.5f / (sigma * sigma));
      float g2 = g1 * g1;
      sum = g0 * tex2D(tex1_32f_C1__, xx, yy);
      float sum_coeff = g0;
      for (int j = 1; j <= half_kernel_elements; j++)
      {
        g0 *= g1;
        g1 *= g2;
        float cur_yy = IUMAX(0.5f, IUMIN(height-0.5f, yy+j));
        sum += g0 * tex2D(tex1_32f_C1__, xx, cur_yy);
        cur_yy = IUMAX(0.5f, IUMIN(height-0.5f, yy-j));
        sum += g0 *  tex2D(tex1_32f_C1__, xx, cur_yy);
        sum_coeff += 2.0f*g0;
      }
      dst[oc] = sum/sum_coeff;
    }
  }
}


// ----------------------------------------------------------------------------
// kernel: Gaussian filter; 32-bit; 1-channel
/** Perform a convolution with an gaussian smoothing kernel
 * @param dst          pointer to output image (linear memory)
 * @param stride       length of image row [pixels]
 * @param xoff         x-coordinate offset where to start the region [pixels]
 * @param yoff         y-coordinate offset where to start the region [pixels]
 * @param width        width of region [pixels]
 * @param height       height of region [pixels]
 * @param sigma        sigma of the smoothing kernel
 * @param kernel_size  lenght of the smoothing kernel [pixels]
 * @param horizontal   defines the direction of convolution
 */
__global__ void cuFilterGaussZKernel_32f_C1(float* dst, float* src,
                                            const int y,
                                            const int width, const int depth,
                                            const size_t stride, const size_t slice_stride,
                                            float sigma, int kernel_size)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int z = blockIdx.y*blockDim.y + threadIdx.y;

  if(x>=0 && z>= 0 && x<width && z<depth)
  {
    float sum = 0.0f;
    int half_kernel_elements = (kernel_size - 1) / 2;

    // convolve horizontally
    float g0 = 1.0f / (sqrtf(2.0f * 3.141592653589793f) * sigma);
    float g1 = exp(-0.5f / (sigma * sigma));
    float g2 = g1 * g1;
    sum = g0 * src[z*slice_stride + y*stride + x];
    float sum_coeff = g0;
    for (int i = 1; i <= half_kernel_elements; i++)
    {
      g0 *= g1;
      g1 *= g2;
      int cur_z = IUMAX(0, IUMIN(depth-1, z + i));
      sum += g0 * src[cur_z*slice_stride + y*stride + x];
      cur_z = IUMAX(0, IUMIN(depth-1, z - i));
      sum += g0 * src[cur_z*slice_stride + y*stride + x];
      sum_coeff += 2.0f*g0;
    }
    dst[z*slice_stride + y*stride + x] = sum/sum_coeff;
  }
}

// ----------------------------------------------------------------------------
// kernel: Gaussian filter; 32-bit; 4-channel
/** Perform a convolution with an gaussian smoothing kernel
 * @param dst          pointer to output image (linear memory)
 * @param stride       length of image row [pixels]
 * @param xoff         x-coordinate offset where to start the region [pixels]
 * @param yoff         y-coordinate offset where to start the region [pixels]
 * @param width        width of region [pixels]
 * @param height       height of region [pixels]
 * @param sigma        sigma of the smoothing kernel
 * @param kernel_size  lenght of the smoothing kernel [pixels]
 * @param horizontal   defines the direction of convolution
 */
__global__ void cuFilterGaussKernel_32f_C4(float4* dst, const size_t stride,
                                           const int xoff, const int yoff,
                                           const int width, const int height,
                                           float sigma, int kernel_size, bool horizontal=true)
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
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    int half_kernel_elements = (kernel_size - 1) / 2;

    if (horizontal)
    {
      // convolve horizontally
      float g0 = 1.0f / (sqrtf(2.0f * 3.141592653589793f) * sigma);
      float g1 = exp(-0.5f / (sigma * sigma));
      float g2 = g1 * g1;
      sum = g0 * tex2D(tex1_32f_C4__, xx, yy);
      float sum_coeff = g0;
      for (int i = 1; i <= half_kernel_elements; i++)
      {
        g0 *= g1;
        g1 *= g2;
        float cur_xx = IUMAX(0.5f, IUMIN(width-0.5f, xx + i));
        sum += g0 * tex2D(tex1_32f_C4__, cur_xx, yy);
        cur_xx = IUMAX(0.5f, IUMIN(width-0.5f, xx-i));
        sum += g0 * tex2D(tex1_32f_C4__, cur_xx, yy);
        sum_coeff += 2.0f*g0;
      }
      dst[oc] = sum/sum_coeff;
    }
    else
    {
      // convolve vertically
      float g0 = 1.0f / (sqrtf(2.0f * 3.141592653589793f) * sigma);
      float g1 = exp(-0.5f / (sigma * sigma));
      float g2 = g1 * g1;
      sum = g0 * tex2D(tex1_32f_C4__, xx, yy);
      float sum_coeff = g0;
      for (int j = 1; j <= half_kernel_elements; j++)
      {
        g0 *= g1;
        g1 *= g2;
        float cur_yy = IUMAX(0.5f, IUMIN(height-0.5f, yy+j));
        sum += g0 * tex2D(tex1_32f_C4__, xx, cur_yy);
        cur_yy = IUMAX(0.5f, IUMIN(height-0.5f, yy-j));
        sum += g0 *  tex2D(tex1_32f_C4__, xx, cur_yy);
        sum_coeff += 2.0f*g0;
      }
      dst[oc] = sum/sum_coeff;
    }
  }
}


// ----------------------------------------------------------------------------
// wrapper: Gaussian filter; 32-bit; 1-channel
IuStatus cuFilterGauss(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst, const IuRect& roi, float sigma, int kernel_size)
{
  if (kernel_size == 0)
    kernel_size = max(5, (unsigned int)ceil(sigma*  3)*  2 + 1);
  if (kernel_size%2 == 0)
    ++kernel_size;

  // temporary variable for filtering (separabed kernel!)
  iu::ImageGpu_32f_C1 tmp(src->size());

  // textures
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
  tex1_32f_C1__.filterMode = cudaFilterModeLinear;
  tex1_32f_C1__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C1__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C1__.normalized = false;

  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(roi.width, dimBlock.x), iu::divUp(roi.height, dimBlock.y));

  // Convolve horizontally
  cudaBindTexture2D(0, &tex1_32f_C1__, src->data(), &channel_desc, src->width(), src->height(), src->pitch());
  cuFilterGaussKernel_32f_C1 <<< dimGrid, dimBlock >>> (tmp.data(roi.x, roi.y), tmp.stride(),
                                                        roi.x, roi.y, tmp.width(), tmp.height(),
                                                        sigma, kernel_size, false);

  // Convolve vertically
  cudaBindTexture2D(0, &tex1_32f_C1__, tmp.data(), &channel_desc, tmp.width(), tmp.height(), tmp.pitch());
  cuFilterGaussKernel_32f_C1 <<< dimGrid, dimBlock >>> (dst->data(roi.x, roi.y), dst->stride(),
                                                        roi.x, roi.y, dst->width(), dst->height(),
                                                        sigma, kernel_size, true);

  // unbind textures
  cudaUnbindTexture(&tex1_32f_C1__);

  // error check
  return iu::checkCudaErrorState();
}

// ----------------------------------------------------------------------------
// wrapper: Gaussian filter; Volume; 32-bit; 1-channel
IuStatus cuFilterGauss(const iu::VolumeGpu_32f_C1* src, iu::VolumeGpu_32f_C1* dst, float sigma, int kernel_size)
{
  if (kernel_size == 0)
    kernel_size = max(5, (unsigned int)ceil(sigma*  3)*  2 + 1);
  if (kernel_size%2 == 0)
    ++kernel_size;

  // temporary variable for filtering (separabel kernel!)
  iu::VolumeGpu_32f_C1 tmpVol(src->size());


  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(src->width(), dimBlock.x), iu::divUp(src->height(), dimBlock.y));

  // filter slices
  for (int z=0; z<src->depth(); z++)
  {
    // temporary variable for filtering (separabed kernel!)
    iu::ImageGpu_32f_C1 tmp(src->width(), src->height());

    // textures
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
    tex1_32f_C1__.filterMode = cudaFilterModeLinear;
    tex1_32f_C1__.addressMode[0] = cudaAddressModeClamp;
    tex1_32f_C1__.addressMode[1] = cudaAddressModeClamp;
    tex1_32f_C1__.normalized = false;


    // Convolve horizontally
    cudaBindTexture2D(0, &tex1_32f_C1__, src->data(0,0,z), &channel_desc, src->width(), src->height(), src->pitch());
    cuFilterGaussKernel_32f_C1 <<< dimGrid, dimBlock >>> (tmp.data(), tmp.stride(),
                                                          0, 0, tmp.width(), tmp.height(),
                                                          sigma, kernel_size, false);

    // Convolve vertically
    cudaBindTexture2D(0, &tex1_32f_C1__, tmp.data(), &channel_desc, tmp.width(), tmp.height(), tmp.pitch());
    cuFilterGaussKernel_32f_C1 <<< dimGrid, dimBlock >>> (tmpVol.data(0,0,z), tmpVol.stride(),
                                                          0, 0, tmpVol.width(), tmpVol.height(),
                                                          sigma, kernel_size, true);

    // unbind textures
    cudaUnbindTexture(&tex1_32f_C1__);
  }

  cudaThreadSynchronize();

  dim3 dimGridZ(iu::divUp(src->width(), dimBlock.x), iu::divUp(src->depth(), dimBlock.y));

  // filter slices
  for (int y=0; y<src->height(); y++)
  {
    cuFilterGaussZKernel_32f_C1 <<< dimGridZ, dimBlock >>> (dst->data(), tmpVol.data(),
                                                            y, dst->width(), dst->depth(),
                                                            dst->stride(), dst->slice_stride(),
                                                            sigma, kernel_size);
  }


  // error check
  return iu::checkCudaErrorState();
}

// ----------------------------------------------------------------------------
// wrapper: Gaussian filter; 32-bit; 4-channel
IuStatus cuFilterGauss(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst, const IuRect& roi, float sigma, int kernel_size)
{
  if (kernel_size == 0)
    kernel_size = max(5, (unsigned int)ceil(sigma*  3)*  2 + 1);
  if (kernel_size%2 == 0)
    ++kernel_size;

  // temporary variable for filtering (separabed kernel!)
  iu::ImageGpu_32f_C4 tmp(src->size());

  // textures
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
  tex1_32f_C4__.filterMode = cudaFilterModeLinear;
  tex1_32f_C4__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C4__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C4__.normalized = false;

  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(roi.width, dimBlock.x), iu::divUp(roi.height, dimBlock.y));

  // Convolve horizontally
  cudaBindTexture2D(0, &tex1_32f_C4__, src->data(), &channel_desc, src->width(), src->height(), src->pitch());
  cuFilterGaussKernel_32f_C4 <<< dimGrid, dimBlock >>> (tmp.data(roi.x, roi.y), tmp.stride(),
                                                        roi.x, roi.y, tmp.width(), tmp.height(),
                                                        sigma, kernel_size, false);
  cudaUnbindTexture(tex1_32f_C4__);

  // Convolve vertically
  cudaBindTexture2D(0, &tex1_32f_C4__, tmp.data(), &channel_desc, tmp.width(), tmp.height(), tmp.pitch());
  cuFilterGaussKernel_32f_C4 <<< dimGrid, dimBlock >>> (dst->data(roi.x, roi.y), dst->stride(),
                                                        roi.x, roi.y, dst->width(), dst->height(),
                                                        sigma, kernel_size, true);
  cudaUnbindTexture(&tex1_32f_C4__);

  // error check
  return iu::checkCudaErrorState();
}

//-----------------------------------------------------------------------------
// wrapper: cubic bspline coefficients prefilter.
IuStatus cuCubicBSplinePrefilter_32f_C1I(iu::ImageGpu_32f_C1 *input)
{
  const unsigned int block_size = 64;
  const unsigned int width  = input->width();
  const unsigned int height = input->height();

  dim3 dimBlockX(block_size,1,1);
  dim3 dimGridX(iu::divUp(height, block_size),1,1);
  cuSamplesToCoefficients2DX<float> <<< dimGridX, dimBlockX >>> (input->data(), width, height, input->stride());

  dim3 dimBlockY(block_size,1,1);
  dim3 dimGridY(iu::divUp(width, block_size),1,1);
  cuSamplesToCoefficients2DY<float> <<< dimGridY, dimBlockY >>> (input->data(), width, height, input->stride());

  return iu::checkCudaErrorState();
}


// -- C1 -> C2 ---------------------------------------------------------------
// kernel: edge filter; 32-bit; 1-channel
__global__ void  cuFilterEdgeKernel_32f_C1(float2* dst, const size_t stride,
                                           const int xoff, const int yoff,
                                           const int width, const int height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  x += xoff;
  y += yoff;

  float xx = x+0.5f;
  float yy = y+0.5f;

  if(x>=0 && y>= 0 && x<width && y<height)
  {
    dst[y*stride+x] = make_float2(tex2D(tex1_32f_C1__, xx+1.0f, yy) - tex2D(tex1_32f_C1__, xx, yy),
                                  tex2D(tex1_32f_C1__, xx, yy+1.0f) - tex2D(tex1_32f_C1__, xx, yy) );
  }
}

// ----------------------------------------------------------------------------
// wrapper: edge filter
IuStatus cuFilterEdge(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C2* dst, const IuRect& roi)
{
  // bind textures
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
  tex1_32f_C1__.filterMode = cudaFilterModeLinear;
  tex1_32f_C1__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C1__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C1__.normalized = false;
  cudaBindTexture2D(0, &tex1_32f_C1__, src->data(), &channel_desc, src->width(),
                    src->height(), src->pitch());

  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(roi.width, dimBlock.x), iu::divUp(roi.height, dimBlock.y));

  cuFilterEdgeKernel_32f_C1<<<dimGrid, dimBlock>>>(dst->data(roi.x, roi.y), dst->stride(),
                                                   roi.x, roi.y, roi.width, roi.height);

  // unbind textures
  cudaUnbindTexture(&tex1_32f_C1__);

  // error check
  return iu::checkCudaErrorState();
}


// -- C1 -> C4 - Eval --------------------------------------------------------
// kernel: edge filter + evaluation; 32-bit; 1-channel
__global__ void  cuFilterEdgeKernel_32f_C1(float4* dst, float alpha, float beta, float minval,
                                           const size_t stride,
                                           const int xoff, const int yoff,
                                           const int width, const int height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  x += xoff;
  y += yoff;

  float xx = x+0.5f;
  float yy = y+0.5f;

  if(x>=0 && y>= 0 && x<width && y<height)
  {
    dst[y*stride+x] = make_float4(max(minval, exp(-alpha*pow(abs(tex2D(tex1_32f_C1__, xx+1.0f, yy)      - tex2D(tex1_32f_C1__, xx, yy)), beta))),
                                  max(minval, exp(-alpha*pow(abs(tex2D(tex1_32f_C1__, xx, yy+1.0f)      - tex2D(tex1_32f_C1__, xx, yy)), beta))),
                                  max(minval, exp(-alpha*pow(abs(tex2D(tex1_32f_C1__, xx+1.0f, yy+1.0f) - tex2D(tex1_32f_C1__, xx, yy)), beta))),
                                  max(minval, exp(-alpha*pow(abs(tex2D(tex1_32f_C1__, xx+1.0f, yy-1.0f) - tex2D(tex1_32f_C1__, xx, yy)), beta))) );
  }
}

// ----------------------------------------------------------------------------
// wrapper: edge filter  + evaluation
IuStatus cuFilterEdge(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C4* dst,
                      const IuRect& roi, float alpha, float beta, float minval)
{
  // bind textures
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
  tex1_32f_C1__.filterMode = cudaFilterModeLinear;
  tex1_32f_C1__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C1__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C1__.normalized = false;
  cudaBindTexture2D(0, &tex1_32f_C1__, src->data(), &channel_desc, src->width(),
                    src->height(), src->pitch());

  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(roi.width, dimBlock.x), iu::divUp(roi.height, dimBlock.y));

  cuFilterEdgeKernel_32f_C1<<<dimGrid, dimBlock>>>(dst->data(roi.x, roi.y), alpha, beta,
                                                   minval, dst->stride(), roi.x, roi.y,
                                                   roi.width, roi.height);

  // unbind textures
  cudaUnbindTexture(&tex1_32f_C1__);

  // error check
  return iu::checkCudaErrorState();
}


// -- C1 -> C2 - Eval --------------------------------------------------------
// kernel: edge filter + evaluation; 32-bit; 1-channel
__global__ void  cuFilterEdgeKernel_32f_C1(float2* dst, float alpha, float beta, float minval,
                                           const size_t stride,
                                           const int xoff, const int yoff,
                                           const int width, const int height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  x += xoff;
  y += yoff;

  float xx = x+0.5f;
  float yy = y+0.5f;

  if(x>=0 && y>= 0 && x<width && y<height)
  {
    dst[y*stride+x] = make_float2(max(minval, exp(-alpha*pow(abs(tex2D(tex1_32f_C1__, xx+1.0f, yy) - tex2D(tex1_32f_C1__, xx, yy)), beta))),
                                  max(minval, exp(-alpha*pow(abs(tex2D(tex1_32f_C1__, xx, yy+1.0f) - tex2D(tex1_32f_C1__, xx, yy)), beta))) );
  }
}

// ----------------------------------------------------------------------------
// wrapper: edge filter  + evaluation
IuStatus cuFilterEdge(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C2* dst,
                      const IuRect& roi, float alpha, float beta, float minval)
{
  // bind textures
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
  tex1_32f_C1__.filterMode = cudaFilterModeLinear;
  tex1_32f_C1__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C1__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C1__.normalized = false;
  cudaBindTexture2D(0, &tex1_32f_C1__, src->data(), &channel_desc, src->width(),
                    src->height(), src->pitch());

  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(roi.width, dimBlock.x), iu::divUp(roi.height, dimBlock.y));

  cuFilterEdgeKernel_32f_C1<<<dimGrid, dimBlock>>>(dst->data(roi.x, roi.y), alpha, beta,
                                                   minval, dst->stride(), roi.x, roi.y,
                                                   roi.width, roi.height);

  // unbind textures
  cudaUnbindTexture(&tex1_32f_C1__);

  // error check
  return iu::checkCudaErrorState();
}


// -- C1 -> C1 - Eval --------------------------------------------------------
// kernel: edge filter + evaluation; 32-bit; 1-channel
__global__ void  cuFilterEdgeKernel_32f_C1(float* dst, float alpha, float beta, float minval,
                                           const size_t stride,
                                           const int xoff, const int yoff,
                                           const int width, const int height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  x += xoff;
  y += yoff;

  float xx = x+0.5f;
  float yy = y+0.5f;

  if(x>=0 && y>= 0 && x<width && y<height)
  {
    float2 grad = make_float2(tex2D(tex1_32f_C1__, xx+1.0f, yy) - tex2D(tex1_32f_C1__, xx, yy),
                              tex2D(tex1_32f_C1__, xx, yy+1.0f) - tex2D(tex1_32f_C1__, xx, yy) );
    dst[y*stride+x] = max(minval, exp(-alpha*pow(length(grad), beta)));
  }
}

// ----------------------------------------------------------------------------
// wrapper: edge filter  + evaluation
IuStatus cuFilterEdge(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst, const IuRect& roi,
                      float alpha, float beta, float minval)
{
  // bind textures
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
  tex1_32f_C1__.filterMode = cudaFilterModeLinear;
  tex1_32f_C1__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C1__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C1__.normalized = false;
  cudaBindTexture2D(0, &tex1_32f_C1__, src->data(), &channel_desc, src->width(),
                    src->height(), src->pitch());

  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(roi.width, dimBlock.x), iu::divUp(roi.height, dimBlock.y));

  cuFilterEdgeKernel_32f_C1 <<< dimGrid, dimBlock >>> (dst->data(roi.x, roi.y), alpha, beta,
                                                       minval, dst->stride(), roi.x, roi.y,
                                                       roi.width, roi.height);

  // unbind textures
  cudaUnbindTexture(&tex1_32f_C1__);

  // error check
  return iu::checkCudaErrorState();
}


// -- RGB -> C1 - Eval --------------------------------------------------------
// kernel: edge filter + evaluation; 32-bit; 4-channel
__global__ void  cuFilterEdgeKernel_32f_C4(float* dst, float alpha, float beta, float minval,
                                           const size_t stride,
                                           const int xoff, const int yoff,
                                           const int width, const int height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  x += xoff;
  y += yoff;

  float xx = x+0.5f;
  float yy = y+0.5f;

  if(x>=0 && y>= 0 && x<width && y<height)
  {
    float4 gradx = tex2D(tex1_32f_C4__, xx+1.0f, yy) - tex2D(tex1_32f_C4__, xx, yy);
    float4 grady = tex2D(tex1_32f_C4__, xx, yy+1.0f) - tex2D(tex1_32f_C4__, xx, yy);
    float3 grad;
    grad.x = sqrtf(gradx.x*gradx.x + grady.x*grady.x);
    grad.y = sqrtf(gradx.y*gradx.y + grady.y*grady.y);
    grad.z = sqrtf(gradx.z*gradx.z + grady.z*grady.z);
    dst[y*stride+x] = max(minval, exp(-alpha*pow((grad.x+grad.y+grad.z)/3.0f, beta)));
  }
}

// ----------------------------------------------------------------------------
// wrapper: edge filter  + evaluation
IuStatus cuFilterEdge(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C1* dst, const IuRect& roi,
                      float alpha, float beta, float minval)
{
  // bind textures
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
  tex1_32f_C4__.filterMode = cudaFilterModeLinear;
  tex1_32f_C4__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C4__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C4__.normalized = false;
  cudaBindTexture2D(0, &tex1_32f_C4__, src->data(), &channel_desc, src->width(),
                    src->height(), src->pitch());

  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(roi.width, dimBlock.x), iu::divUp(roi.height, dimBlock.y));

  cuFilterEdgeKernel_32f_C4 <<< dimGrid, dimBlock >>> (dst->data(roi.x, roi.y), alpha, beta,
                                                       minval, dst->stride(), roi.x, roi.y,
                                                       roi.width, roi.height);

  // unbind textures
  cudaUnbindTexture(&tex1_32f_C4__);

  // error check
  return iu::checkCudaErrorState();
}

// -- RGB -> C2 - Eval --------------------------------------------------------
// kernel: edge filter + evaluation; 32-bit; 4-channel
__global__ void  cuFilterEdgeKernel_32f_C4(float2* dst, float alpha, float beta, float minval,
                                           const size_t stride,
                                           const int xoff, const int yoff,
                                           const int width, const int height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  x += xoff;
  y += yoff;

  float xx = x+0.5f;
  float yy = y+0.5f;

  if(x>=0 && y>= 0 && x<width && y<height)
  {
    float4 gradx = tex2D(tex1_32f_C4__, xx+1.0f, yy) - tex2D(tex1_32f_C4__, xx, yy);
    float4 grady = tex2D(tex1_32f_C4__, xx, yy+1.0f) - tex2D(tex1_32f_C4__, xx, yy);
    float valx = (abs(gradx.x) + abs(gradx.y) + abs(gradx.z))/3.0f;
    float valy = (abs(grady.x) + abs(grady.y) + abs(grady.z))/3.0f;

    dst[y*stride+x] = make_float2(max(minval, exp(-alpha*pow(valx, beta))),
                                  max(minval, exp(-alpha*pow(valy, beta)))  );
  }
}

// ----------------------------------------------------------------------------
// wrapper: edge filter  + evaluation
IuStatus cuFilterEdge(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C2* dst, const IuRect& roi,
                      float alpha, float beta, float minval)
{
  // bind textures
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
  tex1_32f_C4__.filterMode = cudaFilterModeLinear;
  tex1_32f_C4__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C4__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C4__.normalized = false;
  cudaBindTexture2D(0, &tex1_32f_C4__, src->data(), &channel_desc, src->width(),
                    src->height(), src->pitch());

  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(roi.width, dimBlock.x), iu::divUp(roi.height, dimBlock.y));

  cuFilterEdgeKernel_32f_C4 <<< dimGrid, dimBlock >>> (dst->data(roi.x, roi.y), alpha, beta,
                                                       minval, dst->stride(), roi.x, roi.y,
                                                       roi.width, roi.height);

  // unbind textures
  cudaUnbindTexture(&tex1_32f_C4__);

  // error check
  return iu::checkCudaErrorState();
}


// -- RGB -> C4 - Eval --------------------------------------------------------
// kernel: edge filter + evaluation; 32-bit; 4-channel
__global__ void  cuFilterEdgeKernel_32f_C4(float4* dst, float alpha, float beta, float minval,
                                           const size_t stride,
                                           const int xoff, const int yoff,
                                           const int width, const int height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  x += xoff;
  y += yoff;

  float xx = x+0.5f;
  float yy = y+0.5f;

  if(x>=0 && y>= 0 && x<width && y<height)
  {
    float4 grad = tex2D(tex1_32f_C4__, xx+1.0f, yy) - tex2D(tex1_32f_C4__, xx, yy);
    float valx = (abs(grad.x) + abs(grad.y) + abs(grad.z))/3.0f;
    grad = tex2D(tex1_32f_C4__, xx, yy+1.0f) - tex2D(tex1_32f_C4__, xx, yy);
    float valy = (abs(grad.x) + abs(grad.y) + abs(grad.z))/3.0f;
    grad = tex2D(tex1_32f_C4__, xx+1.0f, yy+1.0f) - tex2D(tex1_32f_C4__, xx, yy);
    float valxy = (abs(grad.x) + abs(grad.y) + abs(grad.z))/3.0f;
    grad = tex2D(tex1_32f_C4__, xx+1.0f, yy-1.0f) - tex2D(tex1_32f_C4__, xx, yy);
    float valxy2 = (abs(grad.x) + abs(grad.y) + abs(grad.z))/3.0f;

    dst[y*stride+x] = make_float4(max(minval, exp(-alpha*pow(valx, beta))),
                                  max(minval, exp(-alpha*pow(valy, beta))),
                                  max(minval, exp(-alpha*pow(valxy, beta))),
                                  max(minval, exp(-alpha*pow(valxy2, beta))) );
  }
}

// ----------------------------------------------------------------------------
// wrapper: edge filter  + evaluation
IuStatus cuFilterEdge(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst, const IuRect& roi,
                      float alpha, float beta, float minval)
{
  // bind textures
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
  tex1_32f_C4__.filterMode = cudaFilterModeLinear;
  tex1_32f_C4__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C4__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C4__.normalized = false;
  cudaBindTexture2D(0, &tex1_32f_C4__, src->data(), &channel_desc, src->width(),
                    src->height(), src->pitch());

  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(roi.width, dimBlock.x), iu::divUp(roi.height, dimBlock.y));

  cuFilterEdgeKernel_32f_C4 <<< dimGrid, dimBlock >>> (dst->data(roi.x, roi.y), alpha, beta,
                                                       minval, dst->stride(), roi.x, roi.y,
                                                       roi.width, roi.height);

  // unbind textures
  cudaUnbindTexture(&tex1_32f_C4__);

  // error check
  return iu::checkCudaErrorState();
}


} // namespace iuprivate

#endif // IUPRIVATE_FILTER_CU

