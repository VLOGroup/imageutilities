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
#include "cubictexture.cu"

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

#define BSpline_Pole (sqrt(3.0f)-2.0f)  //pole for cubic b-spline

template<class floatN>
__device__ floatN InitialCausalCoefficient(
   floatN* c,        // coefficients
   unsigned int DataLength,  // number of coefficients
   int step)
{
   const unsigned int Horizon = UMIN(28, DataLength);

   // this initialization corresponds to mirror boundaries
   // accelerated loop
   float zn = BSpline_Pole;
   floatN Sum = *c;
   for (unsigned int n = 1; n < Horizon; n++) {
      c += step;
      Sum += zn * *c;
      zn *= BSpline_Pole;
   }
   return(Sum);
}

template<class floatN>
__device__ floatN InitialAntiCausalCoefficient(
   floatN* c,        // last coefficient
   unsigned int DataLength,  // number of samples or coefficients
   int step)
{
   // this initialization corresponds to mirror boundaries
   return((BSpline_Pole / (BSpline_Pole * BSpline_Pole - 1.0f)) * (BSpline_Pole * c[-step] + *c));
}

template<class floatN>
__device__ void ConvertToInterpolationCoefficients(
   floatN* coeffs,   // input samples --> output coefficients
   unsigned int DataLength,  // number of samples or coefficients
   int step)
{
   // compute the overall gain
   const float Lambda = (1.0f - BSpline_Pole) * (1.0f - 1.0f / BSpline_Pole);

   // causal initialization
   floatN* c = coeffs;
   floatN previous_c;  //cache the previously calculated c rather than look it up again (faster!)
   *c = previous_c = Lambda * InitialCausalCoefficient(c, DataLength, step);
   // causal recursion
   for (unsigned int n = 1; n < DataLength; ++n) {
      c += step;
      *c = previous_c = Lambda * (*c) + BSpline_Pole * previous_c;
   }

   // anticausal initialization
   *c = previous_c = InitialAntiCausalCoefficient(c, DataLength, step);
   // anticausal recursion
   for (int n = DataLength - 2; 0 <= n; --n) {
      c -= step;
      *c = previous_c = BSpline_Pole * (previous_c - *c);
   }
}

#undef BSpline_Pole

//-----------------------------------------------------------------------------
template<class floatN> __global__ void cuSamplesToCoefficients2DX(
  floatN* image, unsigned int width, unsigned int height, size_t stride)
{
  const unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y < height)
  {
    // process lines in x-direction
    floatN* line = image + y * stride;  //direct access

    ConvertToInterpolationCoefficients(line, width, 1);
  }
}

//-----------------------------------------------------------------------------
template<class floatN> __global__ void cuSamplesToCoefficients2DY(
  floatN* image, unsigned int width, unsigned int height, size_t stride)
{
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < width)
  {
    // process lines in y-direction
    floatN* line = image + x;  //direct access

    ConvertToInterpolationCoefficients(line, height, stride);
  }
}


/* ***************************************************************************
 *  Kernels for image reduction
 * ***************************************************************************/
//-----------------------------------------------------------------------------
/** Reduces src image (bound to texture) by the scale_factor rate using bicubic interpolation.
 * @param dst Reduced (output) image.
 * @param dst_step_bytes Pith for output image in bytes.
 * @param dst_width Width of output image.
 * @param dst_height Height of output image.
 * @param rate Scale factor for x AND y direction. (val>1 for multiplication in kernel)
 */
__global__ void cuReduceCubicKernel(float* dst,
                                    size_t dst_step_bytes, int dst_width, int dst_height,
                                    float x_factor, float y_factor)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;
  const size_t step_pixels = dst_step_bytes / sizeof(float);

  if (x<dst_width && y<dst_height)
  {
    // texture coordinates
    const float xx = (x + 0.5f) * x_factor;
    const float yy = (y + 0.5f) * y_factor;

    // bilinear reduction
    dst[y*step_pixels + x] = cubicTex2D(tex1_32f_C1__, xx, yy);
  }
}

//-----------------------------------------------------------------------------
/** Reduces src image (bound to texture) by the scale_factor rate using linear or nearest neighbour interpolation.
 * @param dst Reduced (output) image.
 * @param dst_step_bytes Pith for output image in bytes.
 * @param dst_width Width of output image.
 * @param dst_height Height of output image.
 * @param rate Scale factor for x AND y direction. (val>1 for multiplication in kernel)
 */
__global__ void cuReduceKernel(float* dst,
                               size_t dst_step_bytes, int dst_width, int dst_height,
                               float x_factor, float y_factor)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;
  const size_t step_pixels = dst_step_bytes / sizeof(float);

  if (x<dst_width && y<dst_height)
  {
    // texture coordinates
    const float xx = (x + 0.5f) * x_factor;
    const float yy = (y + 0.5f) * y_factor;

    // bilinear reduction
    dst[y*step_pixels + x] = tex2D(tex1_32f_C1__, xx, yy);
  }
}

} // namespace iuprivate
