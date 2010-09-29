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
 * Module      : Filter Module
 * Class       : Wrapper
 * Language    : C
 * Description : Implementation of public interfaces to filter module
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */


#include "iufilter.h"
#include "iufilter/filter.h"

namespace iu {

/* ***************************************************************************
     Denoising Filter
 * ***************************************************************************/

// 2D device; 32-bit; 1-channel
void filterMedian3x3(const ImageGpu_32f_C1* src, ImageGpu_32f_C1* dst, const IuRect& roi)
{iuprivate::filterMedian3x3(src, dst, roi);}

// device; 32-bit; 1-channel
void filterGauss(const ImageGpu_32f_C1* src, ImageGpu_32f_C1* dst, const IuRect& roi,
                 float sigma, int kernel_size)
{iuprivate::filterGauss(src, dst, roi, sigma, kernel_size);}

// device; 32-bit; 1-channel
void filterRof(const ImageGpu_32f_C1* src, ImageGpu_32f_C1* dst, const IuRect& roi,
               float lambda, int iterations)
{iuprivate::filterRof(src, dst, roi, lambda, iterations);}

/* ***************************************************************************
     Structure-texture decomposition
 * ***************************************************************************/
// device; 32-bit; 1-channel
void decomposeStructureTextureGauss(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst,
                                    const IuRect& roi, float weight, float sigma, int kernel_size)
{iuprivate::decomposeStructureTextureGauss(src, dst, roi, weight, sigma, kernel_size);}

// device; 32-bit; 1-channel
void decomposeStructureTextureRof(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst,
                                  const IuRect& roi, float weight, float lambda, int iterations)
{iuprivate::decomposeStructureTextureRof(src, dst, roi, weight, lambda, iterations);}


} // namespace iu
