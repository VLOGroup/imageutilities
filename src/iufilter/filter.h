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
 * Project     : Utilities for IPP and NPP images
 * Module      : Filter;
 * Class       : none
 * Language    : C++
 * Description : Definition of filter routines
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUPRIVATE_FILTER_H
#define IUPRIVATE_FILTER_H

#include <iucore/coredefs.h>
#include <iucore/memorydefs.h>

namespace iuprivate {

/* ***************************************************************************
 *  Declaration of CUDA WRAPPERS
 * ***************************************************************************/

// Median filter; device; 32-bit; 1-channel;
void filterMedian3x3(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst,
                     const IuRect& roi);

/* ***************************************************************************/
// Gaussian convolution

// 32-bit; 1-channel
void filterGauss(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst,
                 const IuRect& roi, float sigma, int kernel_size,
                 iu::ImageGpu_32f_C1* temp=NULL, cudaStream_t stream=0);

// Volume; 32-bit; 1-channel
void filterGauss(const iu::VolumeGpu_32f_C1* src, iu::VolumeGpu_32f_C1* dst,
                 float sigma, int kernel_size);

// 32-bit; 4-channel
void filterGauss(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst,
                 const IuRect& roi, float sigma, int kernel_size);


/* ***************************************************************************/
// Bilateral filter
void filterBilateral(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst, const IuRect& roi,
                     const iu::ImageGpu_32f_C1* prior, const int iters,
                     const float sigma_spatial, const float sigma_range,
                     const int radius);
void filterBilateral(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst, const IuRect& roi,
                     const iu::ImageGpu_32f_C4* prior, const int iters,
                     const float sigma_spatial, const float sigma_range,
                     const int radius);
void filterBilateral(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst, const IuRect& roi,
                     const iu::ImageGpu_32f_C4* prior, const int iters,
                     const float sigma_spatial, const float sigma_range,
                     const int radius);

/* ***************************************************************************/
// Edge Filters
void filterEdge(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C2* dst, const IuRect& roi);
void filterEdge(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst, const IuRect& roi,
                float alpha, float beta, float minval);
void filterEdge(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C2* dst, const IuRect& roi,
                float alpha, float beta, float minval);
void filterEdge(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C4* dst, const IuRect& roi,
                float alpha, float beta, float minval);
void filterEdge(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C1* dst, const IuRect& roi,
                float alpha, float beta, float minval);
void filterEdge(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C2* dst, const IuRect& roi,
                float alpha, float beta, float minval);
void filterEdge(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst, const IuRect& roi,
                float alpha, float beta, float minval);

/* ***************************************************************************/
// Cubic B-Spline coefficients prefilter
void cubicBSplinePrefilter(iu::ImageGpu_32f_C1* srcdst);


} // namespace iuprivate

#endif // IUPRIVATE_FILTER_H
