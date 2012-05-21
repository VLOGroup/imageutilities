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

#ifndef IUPRIVATE_FILTER_CUH
#define IUPRIVATE_FILTER_CUH

#include <iucore/coredefs.h>
#include <iucore/memorydefs.h>

namespace iuprivate {

// median filter; 32-bit; 1-channel
IuStatus cuFilterMedian3x3(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst,
                            const IuRect& roi);

// Gaussian filter; 32-bit; 1-channel
IuStatus cuFilterGauss(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst,
                        const IuRect& roi, float sigma, int kernel_size);

// Gaussian filter; Volume; 32-bit; 1-channel
IuStatus cuFilterGauss(const iu::VolumeGpu_32f_C1* src, iu::VolumeGpu_32f_C1* dst,
                       float sigma, int kernel_size);


// Cubic bspline coefficients prefilter.
IuStatus cuCubicBSplinePrefilter_32f_C1I(iu::ImageGpu_32f_C1 *input);

// edge filter
IuStatus cuFilterEdge(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C2* dst, const IuRect& roi);

// edge filter  + evaluation
IuStatus cuFilterEdge(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst, const IuRect& roi,
                      float alpha, float beta, float minval);

// edge filter  + evaluation
IuStatus cuFilterEdge(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C2* dst, const IuRect& roi,
                      float alpha, float beta, float minval);

// edge filter  + evaluation
IuStatus cuFilterEdge(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C4* dst, const IuRect& roi,
                      float alpha, float beta, float minval);

// edge filter  + evaluation
IuStatus cuFilterEdge(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C1* dst, const IuRect& roi,
                      float alpha, float beta, float minval);

// edge filter  + evaluation
IuStatus cuFilterEdge(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C2* dst, const IuRect& roi,
                      float alpha, float beta, float minval);

// edge filter  + evaluation
IuStatus cuFilterEdge(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst, const IuRect& roi,
                      float alpha, float beta, float minval);

// bilateral filter
void cuFilterBilateral(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst, const iu::ImageGpu_32f_C1* prior,
                       const IuRect& roi, float sigma_spatial, float sigma_range);



} // namespace iuprivate

#endif // IUPRIVATE_FILTER_CUH
