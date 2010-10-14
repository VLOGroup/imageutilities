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
 * Module      : Geometric Transformation
 * Class       : none
 * Language    : CUDA
 * Description : Implementation of CUDA wrappers for reduce operations
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#include <iudefs.h>
#include <iucutil.h>
#include <iucore/iutextures.cuh>

namespace iuprivate {

/* ***************************************************************************
 *  CUDA KERNELS
 * ***************************************************************************/

//-----------------------------------------------------------------------------
/** Reduces src image (bound to texture) by the scale_factor rate using bicubic interpolation.
 * @param dst Reduced (output) image.
 * @param dst_stride Pith for output image in bytes.
 * @param dst_width Width of output image.
 * @param dst_height Height of output image.
 * @param rate Scale factor for x AND y direction. (val>1 for multiplication in kernel)
 */
__global__ void cuReduceCubicKernel(float* dst,
                                    size_t dst_stride, int dst_width, int dst_height,
                                    float x_factor, float y_factor)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;

  if (x<dst_width && y<dst_height)
  {
    // texture coordinates
    const float xx = (x + 0.5f) * x_factor;
    const float yy = (y + 0.5f) * y_factor;

    // bilinear reduction
    dst[y*dst_stride + x] = iu::cubicTex2D(tex1_32f_C1__, xx, yy);
  }
}

//-----------------------------------------------------------------------------
/** Reduces src image (bound to texture) by the scale_factor rate using linear or nearest neighbour interpolation.
 * @param dst Reduced (output) image.
 * @param dst_stride Pith for output image in bytes.
 * @param dst_width Width of output image.
 * @param dst_height Height of output image.
 * @param rate Scale factor for x AND y direction. (val>1 for multiplication in kernel)
 */
__global__ void cuReduceKernel(float* dst,
                               size_t dst_stride, int dst_width, int dst_height,
                               float x_factor, float y_factor)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;

  if (x<dst_width && y<dst_height)
  {
    // texture coordinates
    const float xx = (x + 0.5f) * x_factor;
    const float yy = (y + 0.5f) * y_factor;

    // bilinear reduction
    dst[y*dst_stride + x] = tex2D(tex1_32f_C1__, xx, yy);
  }
}



/* ***************************************************************************
 *  CUDA WRAPPERS
 * ***************************************************************************/

//-----------------------------------------------------------------------------
IuStatus cuReduce(iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst,
                  IuInterpolationType interpolation)
{
  // x_/y_factor > 0 (for multiplication with dst coords in the kernel!)
  float x_factor = (float)src->width() / (float)dst->width();
  float y_factor = (float)src->height() / (float)dst->height();

  tex1_32f_C1__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C1__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C1__.normalized = false;

  // bind src image to texture and use as input for reduction
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
  cudaBindTexture2D(0, &tex1_32f_C1__, src->data(), &channel_desc,
                    src->width(), src->height(), src->pitch());

  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGridOut(iu::divUp(dst->width(), dimBlock.x),
                  iu::divUp(dst->height(), dimBlock.y));

  switch(interpolation)
  {
  case IU_INTERPOLATE_NEAREST:
  case IU_INTERPOLATE_CUBIC:
    tex1_32f_C1__.filterMode = cudaFilterModePoint;
    break;
  case IU_INTERPOLATE_LINEAR:
    tex1_32f_C1__.filterMode = cudaFilterModeLinear;
    break;
  }

  switch(interpolation)
  {
  case IU_INTERPOLATE_NEAREST:
  case IU_INTERPOLATE_LINEAR: // fallthrough intended
    cuReduceKernel <<< dimGridOut, dimBlock >>> (
        dst->data(), dst->stride(), dst->width(), dst->height(), x_factor, y_factor);
    break;
  case IU_INTERPOLATE_CUBIC:
    cuReduceCubicKernel <<< dimGridOut, dimBlock >>> (
        dst->data(), dst->stride(), dst->width(), dst->height(), x_factor, y_factor);
    break;
  }

  cudaUnbindTexture(&tex1_32f_C1__);

  return iu::checkCudaErrorState();
}

} // namespace iuprivate
