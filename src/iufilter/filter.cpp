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
 * Module      : Filter;
 * Class       : none
 * Language    : C++
 * Description : Implementation of filter routines
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#include <core/copy.h>
#include <math/arithmetic.h>
#include "filter.h"
#include "filter.cuh"

namespace iuprivate {

// device; 32-bit; 1-channel
void filterGauss(const iu::ImageNpp_32f_C1* src, iu::ImageNpp_32f_C1* dst, const IuRect& roi, float sigma, int kernel_size)
{
  NppStatus status = cuFilterGauss(src, dst, roi, sigma, kernel_size);
  IU_ASSERT(status == NPP_NO_ERROR);
}

// device; 32-bit; 1-channel
void filterMedian3x3(const iu::ImageNpp_32f_C1* src, iu::ImageNpp_32f_C1* dst, const IuRect& roi)
{
  NppStatus status = cuFilterMedian3x3(src, dst, roi);
  IU_ASSERT(status == NPP_NO_ERROR);
}

// device; 32-bit; 1-channel
void filterRof(const iu::ImageNpp_32f_C1* src, iu::ImageNpp_32f_C1* dst,
               const IuRect& roi, float lambda, int iterations)
{
  NppStatus status = cuFilterRof(src, dst, roi, lambda, iterations);
  IU_ASSERT(status == NPP_NO_ERROR);
}

// device; 32-bit; 1-channel
void decomposeStructureTextureGauss(const iu::ImageNpp_32f_C1* src, iu::ImageNpp_32f_C1* dst,
                                    const IuRect& roi, float weight, float sigma, int kernel_size)
{
  iu::ImageNpp_32f_C1 tmp(dst->size());
  tmp.setRoi(roi);
  iuprivate::filterGauss(src, &tmp, roi, sigma, kernel_size);
  iuprivate::addWeighted(src, 1.0f, &tmp, -weight, dst, roi);
}

// device; device; 32-bit; 1-channel
void decomposeStructureTextureRof(const iu::ImageNpp_32f_C1* src, iu::ImageNpp_32f_C1* dst,
                                  const IuRect& roi, float weight, float lambda, int iterations)
{
  iu::ImageNpp_32f_C1 tmp(dst->size());
  tmp.setRoi(dst->roi());
  iuprivate::filterRof(src, &tmp, roi, lambda, iterations);
  iuprivate::addWeighted(src, 1.0f, &tmp, -weight, dst, roi);
}

} // namespace iuprivate
