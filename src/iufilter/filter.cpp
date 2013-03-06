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
#include "filter.cuh"

namespace iuprivate {

/* ***************************************************************************/

// device; 32-bit; 1-channel
void filterMedian3x3(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst, const IuRect& roi)
{
  cuFilterMedian3x3(src, dst, roi);
}


/* ***************************************************************************/

// device; 32-bit; 1-channel
void filterGauss(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst, const IuRect& roi, float sigma, int kernel_size)
{
  cuFilterGauss(src, dst, roi, sigma, kernel_size);
}

// device; Volume; 32-bit; 1-channel
void filterGauss(const iu::VolumeGpu_32f_C1* src, iu::VolumeGpu_32f_C1* dst, float sigma, int kernel_size)
{
  cuFilterGauss(src, dst, sigma, kernel_size);
}

// device; 32-bit; 4-channel
void filterGauss(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst, const IuRect& roi, float sigma, int kernel_size)
{
  cuFilterGauss(src, dst, roi, sigma, kernel_size);
}


/* ***************************************************************************/
// Bilateral filter

// C1
void filterBilateral(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst, const IuRect& roi,
                     const iu::ImageGpu_32f_C1* prior, const int iters,
                     const float sigma_spatial, const float sigma_range,
                     const int radius)
{
  if (prior == NULL)
    prior = src;
  cuFilterBilateral(src, dst, roi, prior, iters, sigma_spatial, sigma_range, radius);
}

// C1 but C4 prior
void filterBilateral(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst, const IuRect& roi,
                     const iu::ImageGpu_32f_C4* prior, const int iters,
                     const float sigma_spatial, const float sigma_range,
                     const int radius)
{
  cuFilterBilateral(src, dst, roi, prior, iters, sigma_spatial, sigma_range, radius);
}

// C4
void filterBilateral(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst, const IuRect& roi,
                     const iu::ImageGpu_32f_C4* prior, const int iters,
                     const float sigma_spatial, const float sigma_range,
                     const int radius)
{
  if (prior == NULL)
    prior = src;
  cuFilterBilateral(src, dst, roi, prior, iters, sigma_spatial, sigma_range, radius);
}

/* ***************************************************************************/

// edge filter; device; 32-bit; 1-channel
void filterEdge(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C2* dst, const IuRect& roi)
{
  cuFilterEdge(src, dst, roi);
}

// edge filter + evaluation; device; 32-bit; 1-channel
void filterEdge(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst, const IuRect& roi,
                    float alpha, float beta, float minval)
{
  cuFilterEdge(src, dst, roi, alpha, beta, minval);
}

// edge filter + evaluation; device; 32-bit; 1-channel
void filterEdge(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C2* dst, const IuRect& roi,
                    float alpha, float beta, float minval)
{
  cuFilterEdge(src, dst, roi, alpha, beta, minval);
}

// edge filter + evaluation; device; 32-bit; 1-channel
void filterEdge(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C4* dst, const IuRect& roi,
                    float alpha, float beta, float minval)
{
  cuFilterEdge(src, dst, roi, alpha, beta, minval);
}

// edge filter + evaluation; device; 32-bit; 4-channel (RGB)
void filterEdge(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C1* dst, const IuRect& roi,
                    float alpha, float beta, float minval)
{
  cuFilterEdge(src, dst, roi, alpha, beta, minval);
}

// edge filter + evaluation; device; 32-bit; 4-channel (RGB)
void filterEdge(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C2* dst, const IuRect& roi,
                    float alpha, float beta, float minval)
{
  cuFilterEdge(src, dst, roi, alpha, beta, minval);
}

// edge filter + evaluation; device; 32-bit; 4-channel (RGB)
void filterEdge(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst, const IuRect& roi,
                    float alpha, float beta, float minval)
{
  cuFilterEdge(src, dst, roi, alpha, beta, minval);
}


/* ***************************************************************************/

// device; 32-bit; 1-channel
void cubicBSplinePrefilter(iu::ImageGpu_32f_C1* srcdst)
{
  cuCubicBSplinePrefilter_32f_C1I(srcdst);
}


} // namespace iuprivate
