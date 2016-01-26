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
//#include <iumath/arithmetic.h>
#include "filter.h"
#include "filter.cuh"

namespace iuprivate {

/* ***************************************************************************/

// device; 32-bit; 1-channel
void filterMedian3x3(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst)
{
  cuFilterMedian3x3(src, dst);
}


/* ***************************************************************************/

// device; 32-bit; 1-channel
void filterGauss(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst,
                 float sigma, int kernel_size, iu::ImageGpu_32f_C1 *temp, cudaStream_t stream)
{
  cuFilterGauss(src, dst, sigma, kernel_size, temp, stream);
}

// device; Volume; 32-bit; 1-channel
void filterGauss(const iu::VolumeGpu_32f_C1* src, iu::VolumeGpu_32f_C1* dst, float sigma, int kernel_size)
{
  cuFilterGauss(src, dst, sigma, kernel_size);
}

// device; 32-bit; 4-channel
void filterGauss(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst, float sigma, int kernel_size)
{
  cuFilterGauss(src, dst, sigma, kernel_size);
}



/* ***************************************************************************/

// edge filter; device; 32-bit; 1-channel
void filterEdge(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C2* dst)
{
  cuFilterEdge(src, dst);
}

// edge filter + evaluation; device; 32-bit; 1-channel
void filterEdge(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst,
                    float alpha, float beta, float minval)
{
  cuFilterEdge(src, dst, alpha, beta, minval);
}

// edge filter + evaluation; device; 32-bit; 1-channel
void filterEdge(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C2* dst,
                    float alpha, float beta, float minval)
{
  cuFilterEdge(src, dst, alpha, beta, minval);
}

// edge filter + evaluation; device; 32-bit; 1-channel
void filterEdge(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C4* dst,
                    float alpha, float beta, float minval)
{
  cuFilterEdge(src, dst, alpha, beta, minval);
}

// edge filter + evaluation; device; 32-bit; 4-channel (RGB)
void filterEdge(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C1* dst,
                    float alpha, float beta, float minval)
{
  cuFilterEdge(src, dst, alpha, beta, minval);
}

// edge filter + evaluation; device; 32-bit; 4-channel (RGB)
void filterEdge(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C2* dst,
                    float alpha, float beta, float minval)
{
  cuFilterEdge(src, dst, alpha, beta, minval);
}

// edge filter + evaluation; device; 32-bit; 4-channel (RGB)
void filterEdge(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst,
                    float alpha, float beta, float minval)
{
  cuFilterEdge(src, dst, alpha, beta, minval);
}


/* ***************************************************************************/

// device; 32-bit; 1-channel
void cubicBSplinePrefilter(iu::ImageGpu_32f_C1* srcdst)
{
  cuCubicBSplinePrefilter_32f_C1I(srcdst);
}


} // namespace iuprivate
