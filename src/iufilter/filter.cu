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
#include <iucore/memory_modification.h>
#include "filter.cuh"

namespace iuprivate {

// textures only used within this file
texture<float1, 2, cudaReadModeElementType> tex_u_32f_C1__;
texture<float1, 2, cudaReadModeElementType> tex_v_32f_C1__;
texture<float2, 2, cudaReadModeElementType> tex_p_32f_C2__;


/******************************************************************************
    CUDA KERNELS
*******************************************************************************/

// kernel: median filter; 32-bit; 1-channel
__global__ void  cuFilterMedian3x3Kernel_32f_C1(Npp32f* dst, const size_t stride,
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
      sh_in[shc] = tex2D(tex1_32f_C1__, xx, yy).x;

      // Note: the FLT_MAX prevents us from overemphasizing the border pixels if they are outliers!

      /////////////////////////////////////////////////////////////////////////////
      // boundary conditions
      /////////////////////////////////////////////////////////////////////////////
      if (x == 0) // at left image border
      {
        if (y == 0)
          sh_in[shc-shp-1] = FLT_MAX; // left-upper corner (image)
        else if (ty == 1)
          sh_in[shc-shp-1] = tex2D(tex1_32f_C1__, xx, yy-1.0f).x; // left-upper corner (block)

        sh_in[shc-1] = sh_in[shc];     // left border (image)

        if (y == height-1)
          sh_in[shc+shp-1] = FLT_MAX; // left-lower corner (image)
        else if (ty == blockDim.y)
          sh_in[shc+shp-1] = tex2D(tex1_32f_C1__, xx, yy+1.0f).x; // left-lower corner (block)
      }
      else if (tx == 1) // at left block border (inside image w.r.t x)
      {
        if (y == 0)
          sh_in[shc-shp-1] = tex2D(tex1_32f_C1__, xx-1.0f, yy).x; // left-upper corner (block, outside)
        else if (ty == 1)
          sh_in[shc-shp-1] = tex2D(tex1_32f_C1__, xx-1.0f, yy-1.0f).x; // left-upper corner (block, inside)

        sh_in[shc-1] = tex2D(tex1_32f_C1__, xx-1.0f, yy).x; // left border (block)

        if (y == height-1)
          sh_in[shc+shp-1] = tex2D(tex1_32f_C1__, xx-1.0f, yy).x; // left-lower corner (block, outside)
        else if (ty == blockDim.y)
          sh_in[shc+shp-1] = tex2D(tex1_32f_C1__, xx-1.0f, yy+1.0f).x; // left-lower corner (block, inside)
      }


      if (x == width-1) // at right image border
      {
        if (y == 0)
          sh_in[shc-shp+1] = FLT_MAX; // right-upper corner (image)
        else if (ty == 1)
          sh_in[shc-shp+1] = tex2D(tex1_32f_C1__, xx, yy-1.0f).x; // right-upper corner (block)

        sh_in[shc+1] = sh_in[shc]; // right border (image)

        if (y == height-1)
          sh_in[shc+shp+1] = FLT_MAX; // right-lower corner (image)
        else if (ty == blockDim.y)
          sh_in[shc+shp+1] = tex2D(tex1_32f_C1__, xx, yy+1.0f).x; // right-lower corner (block)
      }
      else if (tx == blockDim.x) // at right block border (inside image w.r.t x)
      {
        if (y == 0)
          sh_in[shc-shp+1] = tex2D(tex1_32f_C1__, xx+1.0f, yy).x; // right-upper corner (block, outside)
        else if (ty == 1)
          sh_in[shc-shp+1] = tex2D(tex1_32f_C1__, xx+1.0f, yy-1.0f).x; // right-upper corner (block, inside)

        sh_in[shc+1] = tex2D(tex1_32f_C1__, xx+1.0f, yy).x; // right border (block)

        if (y == height-1)
          sh_in[shc+shp+1] = tex2D(tex1_32f_C1__, xx+1.0f, yy).x; // right-lower corner (block, outside)
        else if (ty == blockDim.y)
          sh_in[shc+shp+1] = tex2D(tex1_32f_C1__, xx+1.0f, yy+1.0f).x; // right-lower corner (block, inside)
      }

      if (y == 0)
        sh_in[shc-shp] = sh_in[shc]; // upper border (image)
      else if (ty == 1)
        sh_in[shc-shp] = tex2D(tex1_32f_C1__, xx, yy-1.0f).x; // upper border (block)

      if (y == height-1)
        sh_in[shc+shp] = sh_in[shc]; // lower border (image)
      else if (ty == blockDim.y)
        sh_in[shc+shp] = tex2D(tex1_32f_C1__, xx, yy+1.0f).x; // lower border (block)

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
__global__ void cuFilterGaussKernel_32f_C1(Npp32f* dst, const size_t stride,
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
      float g0 = 1.0f / (sqrt(2.0f * 3.141592653589793f) * sigma);
      float g1 = exp(-0.5f / (sigma * sigma));
      float g2 = g1 * g1;
      sum = g0 * tex2D(tex1_32f_C1__, xx, yy).x;
      float sum_coeff = g0;
      for (int i = 1; i <= half_kernel_elements; i++)
      {
        g0 *= g1;
        g1 *= g2;
        int cur_xx = max(0, min(width - 1, (int) x + (int) i));
        sum += g0 * tex2D(tex1_32f_C1__, cur_xx, yy).x;
        cur_xx = max(0, min(width - 1, (int) x - (int) i));
        sum += g0 * tex2D(tex1_32f_C1__, cur_xx, yy).x;
        sum_coeff += 2.0f*g0;
      }
      dst[oc] = sum/sum_coeff;
    }
    else
    {
      // convolve vertically
      float g0 = 1.0f / (sqrt(2.0f * 3.141592653589793f) * sigma);
      float g1 = exp(-0.5f / (sigma * sigma));
      float g2 = g1 * g1;
      sum = g0 * tex2D(tex1_32f_C1__, xx, yy).x;
      float sum_coeff = g0;
      for (int j = 1; j <= half_kernel_elements; j++)
      {
        g0 *= g1;
        g1 *= g2;
        int cur_yy = max(0, min(height - 1, (int) y + (int) j));
        sum += g0 * tex2D(tex1_32f_C1__, xx, cur_yy).x;
        cur_yy = max(0, min(height - 1, (int) y - (int) j));
        sum += g0 *  tex2D(tex1_32f_C1__, xx, cur_yy).x;
        sum_coeff += 2.0f*g0;
      }
      dst[oc] = sum/sum_coeff;
    }
  }
}

/** Primal kernel for ROF denoising
 * @param dst          pointer to output image (linear memory)
 * @param dst_v        splitting variable for optimization
 * @param stride       length of image row [pixels]
 * @param xoff         x-coordinate offset where to start the region [pixels]
 * @param yoff         y-coordinate offset where to start the region [pixels]
 * @param width        width of region [pixels]
 * @param height       height of region [pixels]
 * @param lambda       weighting of regularization and data term (amount of smoothing)
 * @param tau_p        stepwidth of primal update
 */
__global__ void cuFilterRofPrimalKernel_32f_C1(
    float* dst, float* dst_v, const size_t stride,
    const int xoff, const int yoff, const int width, const int height,
    float lambda, float tau_p)
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
    // texture fetches
    float u = tex2D(tex_u_32f_C1__, xx, yy).x;
    float v = u;
    float f = tex2D(tex1_32f_C1__, xx, yy).x;

    float2 p_c = tex2D(tex_p_32f_C2__, xx, yy);
    float2 p_w = tex2D(tex_p_32f_C2__, xx-1.0f, yy);
    float2 p_n = tex2D(tex_p_32f_C2__, xx, yy-1.0f);

    if (x == 0)
      p_w = make_float2(0.0f, 0.0f);

    if (y == 0)
      p_n = make_float2(0.0f, 0.0f);

    float divergence = p_c.x - p_w.x + p_c.y - p_n.y;

    u = u + tau_p*divergence;
    u = (u + tau_p*lambda*f)/(1.0f+tau_p*lambda);

    dst[oc] = u;
    dst_v[oc] = 2*u-v;
  }
}

/** Dual kernel for ROF denoising
 * @param dst_v        dual variable for optimization
 * @param stride       length of image row [pixels]
 * @param xoff         x-coordinate offset where to start the region [pixels]
 * @param yoff         y-coordinate offset where to start the region [pixels]
 * @param width        width of region [pixels]
 * @param height       height of region [pixels]
 * @param tau_d        stepwidth of dual update
 */
__global__ void cuFilterRofDualKernel_32f_C1(
    float2* dst_p, const size_t stride,
    const int xoff, const int yoff,
    const int width, const int height, float tau_d)
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
    // texture fetches
    float2 p = tex2D(tex_p_32f_C2__, xx, yy);
    float u = tex2D(tex_v_32f_C1__, xx, yy).x;

    float u_x = tex2D(tex_v_32f_C1__, xx+1.0f, yy).x - u;
    float u_y = tex2D(tex_v_32f_C1__, xx, yy+1.0f).x - u;

    // update dual variable
    float p_new_1 = p.x + tau_d*u_x;
    float p_new_2 = p.y + tau_d*u_y;

    float denom = 1.0f/max(1.0f, sqrt(p_new_1*p_new_1 + p_new_2*p_new_2));

    p.x = p_new_1 * denom;
    p.y = p_new_2 * denom;

    // do not update border pixels
    if (x == width-1)
      p.x = 0.0f;
    if (y == height-1)
      p.y = 0.0f;

    dst_p[oc] = p;
  }
}


/******************************************************************************
  CUDA INTERFACES
*******************************************************************************/

// wrapper: median filter; 32-bit; 1-channel
NppStatus cuFilterMedian3x3(const iu::ImageNpp_32f_C1* src, iu::ImageNpp_32f_C1* dst, const IuRect& roi)
{
  // bind textures
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float1>();
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
  IU_CHECK_CUDA_ERRORS();
  return NPP_SUCCESS;
}

// wrapper: Gaussian filter; 32-bit; 1-channel
NppStatus cuFilterGauss(const iu::ImageNpp_32f_C1* src, iu::ImageNpp_32f_C1* dst, const IuRect& roi, float sigma, int kernel_size)
{
  if (kernel_size == 0)
    kernel_size = max(5, (unsigned int)ceil(sigma*  3)*  2 + 1);
  if (kernel_size%2 == 0)
    ++kernel_size;

  // temporary variable for filtering (separabed kernel!)
  iu::ImageNpp_32f_C1 tmp(src->size());

  // bind textures
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float1>();

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
  IU_CHECK_CUDA_ERRORS();
  return NPP_SUCCESS;
}

// wrapper: Rof filter; 32-bit; 1-channel
NppStatus cuFilterRof(const iu::ImageNpp_32f_C1* src, iu::ImageNpp_32f_C1* dst,
                      const IuRect& roi, float lambda, int iterations)
{
  // helper variables (v=splitting var; p=dual var)
  iu::ImageNpp_32f_C1 v(src->size());
  iu::ImageNpp_32f_C2 p(src->size());
  Npp32f p_init = 0.0f;
  iuprivate::setValue(p_init, &p, p.roi());

  // init output und splitting var with input image.
  iuprivate::copy(src, dst);
  iuprivate::copy(src, &v);

  // bind textures
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float1>();
  cudaBindTexture2D(0, &tex1_32f_C1__, src->data(), &channel_desc, src->width(), src->height(), src->pitch());
  cudaBindTexture2D(0, &tex_u_32f_C1__, dst->data(), &channel_desc, dst->width(), dst->height(), dst->pitch());
  cudaBindTexture2D(0, &tex_v_32f_C1__, v.data(), &channel_desc, v.width(), v.height(), v.pitch());
  cudaChannelFormatDesc channel_desc_C2 = cudaCreateChannelDesc<float2>();
  cudaBindTexture2D(0, &tex_p_32f_C2__, p.data(), &channel_desc_C2, p.width(), p.height(), p.pitch());

  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(roi.width, dimBlock.x), iu::divUp(roi.height, dimBlock.y));

  float tau_p = 0.01f;
  float tau_d = 1.0f/tau_p/8.0f;

  for(int i = 0; i < iterations; i++)
  {
    cuFilterRofPrimalKernel_32f_C1 <<< dimGrid, dimBlock >>>
        (dst->data(roi.x, roi.y), v.data(roi.x, roi.y), dst->stride(),
         roi.x, roi.y, dst->width(), dst->height(), lambda, tau_p);
    IU_CHECK_CUDA_ERRORS();

    cuFilterRofDualKernel_32f_C1 <<< dimGrid, dimBlock >>>
        ((float2*)(p.data()), p.stride()/2,
         roi.x, roi.y, p.width(), p.height(), tau_d);
    IU_CHECK_CUDA_ERRORS();
  }

  // unbind textures
  cudaUnbindTexture(&tex1_32f_C1__);
  cudaUnbindTexture(&tex_u_32f_C1__);
  cudaUnbindTexture(&tex_v_32f_C1__);
  cudaUnbindTexture(&tex_p_32f_C2__);

  // error check
  IU_CHECK_CUDA_ERRORS();
  return NPP_SUCCESS;
}

} // namespace iuprivate

#endif // IUPRIVATE_FILTER_CU

