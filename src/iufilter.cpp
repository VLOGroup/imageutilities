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
// device; volume; 32-bit; 1-channel
void filterGauss(const VolumeGpu_32f_C1* src, VolumeGpu_32f_C1* dst,
                 float sigma, int kernel_size)
{iuprivate::filterGauss(src, dst, sigma, kernel_size);}
// device; 32-bit; 4-channel
void filterGauss(const ImageGpu_32f_C4* src, ImageGpu_32f_C4* dst, const IuRect& roi,
                 float sigma, int kernel_size)
{iuprivate::filterGauss(src, dst, roi, sigma, kernel_size);}


/* ***************************************************************************
     edge calculation
 * ***************************************************************************/

// edge filter; device; 32-bit; 1-channel
void filterEdge(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C2* dst, const IuRect& roi)
{ iuprivate::filterEdge(src, dst, roi); }

// edge filter + evaluation; device; 32-bit; 1-channel
void filterEdge(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst, const IuRect& roi,
                float alpha, float beta, float minval)
{ iuprivate::filterEdge(src, dst, roi, alpha, beta, minval); }

// edge filter + evaluation (4n); device; 32-bit; 1-channel
void filterEdge(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C2* dst, const IuRect& roi,
                float alpha, float beta, float minval)
{ iuprivate::filterEdge(src, dst, roi, alpha, beta, minval); }

// edge filter + evaluation (8n); device; 32-bit; 1-channel
void filterEdge(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C4* dst, const IuRect& roi,
                float alpha, float beta, float minval)
{ iuprivate::filterEdge(src, dst, roi, alpha, beta, minval); }

// edge filter + evaluation; device; 32-bit; 4-channel (RGB)
void filterEdge(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C1* dst, const IuRect& roi,
                float alpha, float beta, float minval)
{ iuprivate::filterEdge(src, dst, roi, alpha, beta, minval); }

// edge filter + evaluation; device; 32-bit; 4-channel (RGB)
void filterEdge(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C2* dst, const IuRect& roi,
                float alpha, float beta, float minval)
{ iuprivate::filterEdge(src, dst, roi, alpha, beta, minval); }

// edge filter + evaluation; device; 32-bit; 4-channel (RGB)
void filterEdge(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst, const IuRect& roi,
                float alpha, float beta, float minval)
{ iuprivate::filterEdge(src, dst, roi, alpha, beta, minval); }



/* ***************************************************************************
     other filters
 * ***************************************************************************/
void cubicBSplinePrefilter(iu::ImageGpu_32f_C1* srcdst)
{ iuprivate::cubicBSplinePrefilter(srcdst); }

} // namespace iu
