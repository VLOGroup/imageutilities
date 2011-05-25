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

#include <iucore/copy.h>
#include <iumath/arithmetic.h>
#include "filter.h"

namespace iuprivate {

// device; 32-bit; 1-channel
void filterGauss(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst, const IuRect& roi, float sigma, int kernel_size)
{
  IuStatus status = cuFilterGauss(src, dst, roi, sigma, kernel_size);
  IU_ASSERT(status == IU_SUCCESS);
}

// device; 32-bit; 4-channel
void filterGauss(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst, const IuRect& roi, float sigma, int kernel_size)
{
  IuStatus status = cuFilterGauss(src, dst, roi, sigma, kernel_size);
  IU_ASSERT(status == IU_SUCCESS);
}

// device; 32-bit; 1-channel
void filterMedian3x3(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst, const IuRect& roi)
{
  IuStatus status = cuFilterMedian3x3(src, dst, roi);
  IU_ASSERT(status == IU_SUCCESS);
}

// device; 32-bit; 1-channel
void cubicBSplinePrefilter(iu::ImageGpu_32f_C1* srcdst)
{
  IuStatus status = cuCubicBSplinePrefilter_32f_C1I(srcdst);
  IU_ASSERT(status == IU_SUCCESS);
}


// edge filter; device; 32-bit; 1-channel
void filterEdge(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C2* dst, const IuRect& roi)
{
  IuStatus status = cuFilterEdge(src, dst, roi);
  IU_ASSERT(status == IU_SUCCESS);
}

// edge filter + evaluation; device; 32-bit; 1-channel
void filterEdge(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst, const IuRect& roi,
                    float alpha, float beta, float minval)
{
  IuStatus status = cuFilterEdge(src, dst, roi, alpha, beta, minval);
  IU_ASSERT(status == IU_SUCCESS);
}

// edge filter + evaluation; device; 32-bit; 4-channel (RGB)
void filterEdge(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C1* dst, const IuRect& roi,
                    float alpha, float beta, float minval)
{
  IuStatus status = cuFilterEdge(src, dst, roi, alpha, beta, minval);
  IU_ASSERT(status == IU_SUCCESS);
}


} // namespace iuprivate
