
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
 * Description : Implementation of CUDA kernels for reduce operations
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#include <iucore/iutextures.cuh>

namespace iuprivate {

/* ***************************************************************************
 *  Kernels for creating cubic bspline coefficients
 * ***************************************************************************/
// The code below is based on the work of Philippe Thevenaz.
// See <http://bigwww.epfl.ch/thevenaz/interpolation/>

inline __device__ __host__ unsigned int UMIN(unsigned int a, unsigned int b)
{
  return a < b ? a : b;
}

#define IU_BSPLINE_POLE (sqrtf(3.0f)-2.0f)  // pole for cubic b-spline

template<class PixelType> __device__ PixelType InitialCausalCoefficient(
  PixelType* c,        // coefficients
  unsigned int DataLength,  // number of coefficients
  int step)
{
  const unsigned int Horizon = UMIN(28, DataLength);

  // this initialization corresponds to mirror boundaries
  // accelerated loop
  float zn = IU_BSPLINE_POLE;
  PixelType Sum = *c;
  for (unsigned int n = 1; n < Horizon; n++) {
    c += step;
    Sum += zn * *c;
    zn *= IU_BSPLINE_POLE;
  }
  return(Sum);
}

template<class PixelType> __device__ PixelType InitialAntiCausalCoefficient(
  PixelType* c,        // last coefficient
  unsigned int DataLength,  // number of samples or coefficients
  int step)
{
  // this initialization corresponds to mirror boundaries
  return((IU_BSPLINE_POLE / (IU_BSPLINE_POLE * IU_BSPLINE_POLE - 1.0f)) *
         (IU_BSPLINE_POLE * c[-step] + *c));
}

//-----------------------------------------------------------------------------
template<class PixelType> __device__ void ConvertToInterpolationCoefficients(
  PixelType* coeffs,   // input samples --> output coefficients
  unsigned int DataLength,  // number of samples or coefficients
  int step)
{
  // compute the overall gain
  const float lambda = (1.0f - IU_BSPLINE_POLE) *
      (1.0f - 1.0f / IU_BSPLINE_POLE);

  // causal initialization
  PixelType* c = coeffs;
  PixelType previous_c;  //cache the previously calculated c rather than look it up again (faster!)
  *c = previous_c = lambda * InitialCausalCoefficient(c, DataLength, step);
  // causal recursion
  for (unsigned int n = 1; n < DataLength; ++n) {
    c += step;
    *c = previous_c = lambda * (*c) + IU_BSPLINE_POLE * previous_c;
  }

  // anticausal initialization
  *c = previous_c = InitialAntiCausalCoefficient(c, DataLength, step);
  // anticausal recursion
  for (int n = DataLength - 2; 0 <= n; --n) {
    c -= step;
    *c = previous_c = IU_BSPLINE_POLE * (previous_c - *c);
  }
}

#undef IU_BSPLINE_POLE

//-----------------------------------------------------------------------------
template<class PixelType> __global__ void cuSamplesToCoefficients2DX(
  PixelType* image, unsigned int width, unsigned int height, size_t stride)
{
  const unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y < height)
  {
    // process lines in x-direction
    PixelType* line = image + y * stride;  //direct access

    ConvertToInterpolationCoefficients(line, width, 1);
  }
}

//-----------------------------------------------------------------------------
template<class PixelType> __global__ void cuSamplesToCoefficients2DY(
  PixelType* image, unsigned int width, unsigned int height, size_t stride)
{
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < width)
  {
    // process lines in y-direction
    PixelType* line = image + x;  //direct access

    ConvertToInterpolationCoefficients(line, height, stride);
  }
}


} // namespace iuprivate
