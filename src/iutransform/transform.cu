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
 * Description : Implementation of CUDA kernels for  transform operations
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#include <iudefs.h>
#include <iucutil.h>
#include <iucore/iutextures.cuh>

#ifndef IUTRANSFORM_TRANSFORM_CU
#define IUTRANSFORM_TRANSFORM_CU

namespace iuprivate {

/* ***************************************************************************
 *  CUDA KERNELS
 * ***************************************************************************/

//-----------------------------------------------------------------------------
/** Reduces src image (bound to texture) by the scale_factor rate
 * @param dst Reduced (output) image.
 * @param dst_stride Pith for output image in bytes.
 * @param dst_width Width of output image.
 * @param dst_height Height of output image.
 * @param rate Scale factor for x AND y direction. (val>1 for multiplication in kernel)
 */

/*
  1-channel; 32-bit
*/

//-----------------------------------------------------------------------------
static __global__ void cuTransformCubicSplineKernel_32f_C1(float* dst,
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
static __global__ void cuTransformCubicKernel_32f_C1(float* dst,
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
    dst[y*dst_stride + x] = iu::cubicTex2DSimple(tex1_32f_C1__, xx, yy);
  }
}

//-----------------------------------------------------------------------------
static __global__ void cuTransformKernel_32f_C1(float* dst,
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

/*
  2-channel; 32-bit
*/

////-----------------------------------------------------------------------------
//static __global__ void cuTransformCubicSplineKernel_32f_C2(float2* dst,
//                                                    size_t dst_stride, int dst_width, int dst_height,
//                                                    float x_factor, float y_factor)
//{
//  const int x = blockIdx.x*blockDim.x + threadIdx.x;
//  const int y = blockIdx.y*blockDim.y + threadIdx.y;

//  if (x<dst_width && y<dst_height)
//  {
//    // texture coordinates
//    const float xx = (x + 0.5f) * x_factor;
//    const float yy = (y + 0.5f) * y_factor;

//    // bilinear reduction
//    dst[y*dst_stride + x] = iu::cubicTex2D(tex1_32f_C2__, xx, yy);
//  }
//}

////-----------------------------------------------------------------------------
//static __global__ void cuTransformCubicKernel_32f_C2(float2* dst,
//                                              size_t dst_stride, int dst_width, int dst_height,
//                                              float x_factor, float y_factor)
//{
//  const int x = blockIdx.x*blockDim.x + threadIdx.x;
//  const int y = blockIdx.y*blockDim.y + threadIdx.y;

//  if (x<dst_width && y<dst_height)
//  {
//    // texture coordinates
//    const float xx = (x + 0.5f) * x_factor;
//    const float yy = (y + 0.5f) * y_factor;

//    // bilinear reduction
//    dst[y*dst_stride + x] = iu::cubicTex2DSimple(tex1_32f_C2__, xx, yy);
//  }
//}

//-----------------------------------------------------------------------------
static __global__ void cuTransformKernel_32f_C2(float2* dst,
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
    dst[y*dst_stride + x] = tex2D(tex1_32f_C2__, xx, yy);
  }
}


/*
  4-channel; 32-bit
*/

////-----------------------------------------------------------------------------
//static __global__ void cuTransformCubicSplineKernel_32f_C4(float4* dst,
//                                                    size_t dst_stride, int dst_width, int dst_height,
//                                                    float x_factor, float y_factor)
//{
//  const int x = blockIdx.x*blockDim.x + threadIdx.x;
//  const int y = blockIdx.y*blockDim.y + threadIdx.y;

//  if (x<dst_width && y<dst_height)
//  {
//    // texture coordinates
//    const float xx = (x + 0.5f) * x_factor;
//    const float yy = (y + 0.5f) * y_factor;

//    // bilinear reduction
//    dst[y*dst_stride + x] = iu::cubicTex2D(tex1_32f_C4__, xx, yy);
//  }
//}

////-----------------------------------------------------------------------------
//static __global__ void cuTransformCubicKernel_32f_C4(float4* dst,
//                                              size_t dst_stride, int dst_width, int dst_height,
//                                              float x_factor, float y_factor)
//{
//  const int x = blockIdx.x*blockDim.x + threadIdx.x;
//  const int y = blockIdx.y*blockDim.y + threadIdx.y;

//  if (x<dst_width && y<dst_height)
//  {
//    // texture coordinates
//    const float xx = (x + 0.5f) * x_factor;
//    const float yy = (y + 0.5f) * y_factor;

//    // bilinear reduction
//    dst[y*dst_stride + x] = iu::cubicTex2DSimple(tex1_32f_C4__, xx, yy);
//  }
//}

//-----------------------------------------------------------------------------
static __global__ void cuTransformKernel_32f_C4(float4* dst,
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
    dst[y*dst_stride + x] = tex2D(tex1_32f_C4__, xx, yy);
  }
}


} // namespace iuprivate

#endif // IUTRANSFORM_TRANSFORM_CU
