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
 * Description : Implementation of CUDA kernels for reduce operations
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#include <iucore/iutextures.cuh>

#ifndef IUPRIVATE_REDUCE_KERNELS_CU
#define IUPRIVATE_REDUCE_KERNELS_CU


namespace iuprivate {

/* ***************************************************************************
 *  Kernels for image reduction
 * ***************************************************************************/
//-----------------------------------------------------------------------------
/** Reduces src image (bound to texture) by the scale_factor rate using bicubic interpolation.
 * @param dst Reduced (output) image.
 * @param dst_step_bytes Pith for output image in bytes.
 * @param dst_width Width of output image.
 * @param dst_height Height of output image.
 * @param rate Scale factor for x AND y direction. (val>1 for multiplication in kernel)
 */
__global__ void cuReduceCubicKernel(float* dst,
                                    size_t dst_step_bytes, int dst_width, int dst_height,
                                    float x_factor, float y_factor)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;
  const size_t step_pixels = dst_step_bytes / sizeof(float);

  if (x<dst_width && y<dst_height)
  {
    // texture coordinates
    const float xx = (x + 0.5f) * x_factor;
    const float yy = (y + 0.5f) * y_factor;

    // bilinear reduction
    dst[y*step_pixels + x] = iu::cubicTex2D(tex1_32f_C1__, xx, yy);
  }
}

//-----------------------------------------------------------------------------
/** Reduces src image (bound to texture) by the scale_factor rate using linear or nearest neighbour interpolation.
 * @param dst Reduced (output) image.
 * @param dst_step_bytes Pith for output image in bytes.
 * @param dst_width Width of output image.
 * @param dst_height Height of output image.
 * @param rate Scale factor for x AND y direction. (val>1 for multiplication in kernel)
 */
__global__ void cuReduceKernel(float* dst,
                               size_t dst_step_bytes, int dst_width, int dst_height,
                               float x_factor, float y_factor)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;
  const size_t step_pixels = dst_step_bytes / sizeof(float);

  if (x<dst_width && y<dst_height)
  {
    // texture coordinates
    const float xx = (x + 0.5f) * x_factor;
    const float yy = (y + 0.5f) * y_factor;

    // bilinear reduction
    dst[y*step_pixels + x] = tex2D(tex1_32f_C1__, xx, yy);
  }
}

} // namespace iuprivate

#endif // IUPRIVATE_REDUCE_KERNELS_CU
