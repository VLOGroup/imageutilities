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

#include "core/coredefs.h"
#include "core/memorydefs.h"

namespace iuprivate {

// Median filter; device; 32-bit; 1-channel;
void filterMedian3x3(const iu::ImageNpp_32f_C1* src, iu::ImageNpp_32f_C1* dst,
                     const IuRect& roi);

// Gaussian convolution; device; 32-bit; 1-channel
void filterGauss(const iu::ImageNpp_32f_C1* src, iu::ImageNpp_32f_C1* dst,
                 const IuRect& roi, float sigma, int kernel_size);

// ROF denoising; device; 32-bit; 1-channel
void filterRof(const iu::ImageNpp_32f_C1* src, iu::ImageNpp_32f_C1* dst,
               const IuRect& roi, float lambda, int interations);

// Gaussian structure-texture decomposition; 32-bit; 1-channel
void decomposeStructureTextureGauss(const iu::ImageNpp_32f_C1* src, iu::ImageNpp_32f_C1* dst,
                                    const IuRect& roi, float weight, float sigma, int kernel_size);

// ROF structure-texture decomposition; device; 32-bit; 1-channel
void decomposeStructureTextureRof(const iu::ImageNpp_32f_C1* src, iu::ImageNpp_32f_C1* dst,
                                  const IuRect& roi, float weight, float lambda, int iterations);

} // namespace iuprivate

#endif // IUPRIVATE_FILTER_H
