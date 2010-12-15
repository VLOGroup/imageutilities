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
 * Description : Implementation of CUDA wrappers for prolongate operations
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#include <iostream>
#include <iudefs.h>
#include <iucutil.h>
#include <iucore/iutextures.cuh>
#include "transform.cu"

#ifndef IUTRANSFORM_PROLONGATE_CU
#define IUTRANSFORM_PROLONGATE_CU

namespace iuprivate {

/* ***************************************************************************
 *  CUDA WRAPPERS
 * ***************************************************************************/

//-----------------------------------------------------------------------------
IuStatus cuProlongate(iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst,
                      IuInterpolationType interpolation)
{
  IuSize src_roi = src->size();
  IuSize dst_roi = dst->size();

  // x_/y_factor < 0 (for multiplication with dst coords in the kernel!)
  float x_factor = static_cast<float>(src_roi.width) /
      static_cast<float>(dst_roi.width);
  float y_factor = static_cast<float>(src_roi.height) /
      static_cast<float>(dst_roi.height);

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
    cuTransformKernel_32f_C1 <<< dimGridOut, dimBlock >>> (
        dst->data(), dst->stride(), dst->width(), dst->height(), x_factor, y_factor);
    break;
  case IU_INTERPOLATE_CUBIC:
    cuTransformCubicKernel_32f_C1 <<< dimGridOut, dimBlock >>> (
        dst->data(), dst->stride(), dst->width(), dst->height(), x_factor, y_factor);
    break;
  case IU_INTERPOLATE_CUBIC_SPLINE:
    cuTransformCubicSplineKernel_32f_C1 <<< dimGridOut, dimBlock >>> (
        dst->data(), dst->stride(), dst->width(), dst->height(), x_factor, y_factor);
    break;
  }

  cudaUnbindTexture(&tex1_32f_C1__);

  return iu::checkCudaErrorState();
}

//-----------------------------------------------------------------------------
IuStatus cuProlongate(iu::ImageGpu_32f_C2* src, iu::ImageGpu_32f_C2* dst,
                      IuInterpolationType interpolation)
{
  IuSize src_roi = src->size();
  IuSize dst_roi = dst->size();

  // x_/y_factor < 0 (for multiplication with dst coords in the kernel!)
  float x_factor = static_cast<float>(src_roi.width) /
      static_cast<float>(dst_roi.width);
  float y_factor = static_cast<float>(src_roi.height) /
      static_cast<float>(dst_roi.height);

  tex1_32f_C2__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C2__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C2__.normalized = false;

  // bind src image to texture and use as input for reduction
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float2>();
  cudaBindTexture2D(0, &tex1_32f_C2__, src->data(), &channel_desc,
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
    tex1_32f_C2__.filterMode = cudaFilterModePoint;
    break;
  case IU_INTERPOLATE_LINEAR:
    tex1_32f_C2__.filterMode = cudaFilterModeLinear;
    break;
  }

  switch(interpolation)
  {
  case IU_INTERPOLATE_NEAREST:
  case IU_INTERPOLATE_LINEAR: // fallthrough intended
    cuTransformKernel_32f_C2 <<< dimGridOut, dimBlock >>> (
        dst->data(), dst->stride(), dst->width(), dst->height(), x_factor, y_factor);
    break;
//  case IU_INTERPOLATE_CUBIC:
//    cuTransformCubicKernel_32f_C2 <<< dimGridOut, dimBlock >>> (
//        dst->data(), dst->stride(), dst->width(), dst->height(), x_factor, y_factor);
//    break;
//  case IU_INTERPOLATE_CUBIC_SPLINE:
//    cuTransformCubicSplineKernel_32f_C2 <<< dimGridOut, dimBlock >>> (
//        dst->data(), dst->stride(), dst->width(), dst->height(), x_factor, y_factor);
//    break;
  default:
    std::cerr << "Interpolation type not supported for this element type" << std::endl;
  }

  cudaUnbindTexture(&tex1_32f_C2__);

  return iu::checkCudaErrorState();
}

//-----------------------------------------------------------------------------
IuStatus cuProlongate(iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst,
                      IuInterpolationType interpolation)
{
  IuSize src_roi = src->size();
  IuSize dst_roi = dst->size();

  // x_/y_factor < 0 (for multiplication with dst coords in the kernel!)
  float x_factor = static_cast<float>(src_roi.width) /
      static_cast<float>(dst_roi.width);
  float y_factor = static_cast<float>(src_roi.height) /
      static_cast<float>(dst_roi.height);

  tex1_32f_C4__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C4__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C4__.normalized = false;

  // bind src image to texture and use as input for reduction
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
  cudaBindTexture2D(0, &tex1_32f_C4__, src->data(), &channel_desc,
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
    tex1_32f_C4__.filterMode = cudaFilterModePoint;
    break;
  case IU_INTERPOLATE_LINEAR:
    tex1_32f_C4__.filterMode = cudaFilterModeLinear;
    break;
  }

  switch(interpolation)
  {
  case IU_INTERPOLATE_NEAREST:
  case IU_INTERPOLATE_LINEAR: // fallthrough intended
    cuTransformKernel_32f_C4 <<< dimGridOut, dimBlock >>> (
        dst->data(), dst->stride(), dst->width(), dst->height(), x_factor, y_factor);
    break;
//  case IU_INTERPOLATE_CUBIC:
//    cuTransformCubicKernel_32f_C4 <<< dimGridOut, dimBlock >>> (
//        dst->data(), dst->stride(), dst->width(), dst->height(), x_factor, y_factor);
//    break;
//  case IU_INTERPOLATE_CUBIC_SPLINE:
//    cuTransformCubicSplineKernel_32f_C4 <<< dimGridOut, dimBlock >>> (
//        dst->data(), dst->stride(), dst->width(), dst->height(), x_factor, y_factor);
//    break;
  default:
    std::cerr << "Interpolation type not supported for this element type" << std::endl;
  }

  cudaUnbindTexture(&tex1_32f_C4__);

  return iu::checkCudaErrorState();
}

} // namespace iuprivate

#endif // IUTRANSFORM_PROLONGATE_CU
