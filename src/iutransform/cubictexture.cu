/*
 * Class       : $RCSfile$
 * Language    : C++
 * Description : Definition of
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#include <cutil_math.h>

#ifndef IUPRIVATE_CUBICTEXTURE_CU
#define IUPRIVATE_CUBICTEXTURE_CU

namespace iuprivate {

  inline __host__ __device__ float2 operator-(float a, float2 b)
  {
    return make_float2(a - b.x, a - b.y);
  }

  inline __host__ __device__ float3 operator-(float a, float3 b)
  {
    return make_float3(a - b.x, a - b.y, a - b.z);
  }

// Cubic B-spline function
// The 3rd order Maximal Order and Minimum Support function, that it is maximally differentiable.
inline __host__ __device__ float bspline(float t)
{
  t = fabs(t);
  const float a = 2.0f - t;

  if (t < 1.0f) return 2.0f/3.0f - 0.5f*t*t*a;
  else if (t < 2.0f) return a*a*a / 6.0f;
  else return 0.0f;
}

// The first order derivative of the cubic B-spline
inline __host__ __device__ float bspline_1st_derivative(float t)
{
  if (-2.0f < t && t <= -1.0f) return 0.5f*t*t + 2.0f*t + 2.0f;
  else if (-1.0f < t && t <= 0.0f) return -1.5f*t*t - 2.0f*t;
  else if ( 0.0f < t && t <= 1.0f) return  1.5f*t*t - 2.0f*t;
  else if ( 1.0f < t && t <  2.0f) return -0.5f*t*t + 2.0f*t - 2.0f;
  else return 0.0f;
}

// The second order derivative of the cubic B-spline
inline __host__ __device__ float bspline_2nd_derivative(float t)
{
  t = fabs(t);

  if (t < 1.0f) return 3.0f*t - 2.0f;
  else if (t < 2.0f) return 2.0f - t;
  else return 0.0f;
}

// Inline calculation of the bspline convolution weights, without conditional statements
template<class T> inline __device__ void bspline_weights(T fraction, T& w0, T& w1, T& w2, T& w3)
{
  const T one_frac = 1.0f - fraction;
  const T squared = fraction * fraction;
  const T one_sqd = one_frac * one_frac;

  w0 = 1.0f/6.0f * one_sqd * one_frac;
  w1 = 2.0f/3.0f - 0.5f * squared * (2.0f-fraction);
  w2 = 2.0f/3.0f - 0.5f * one_sqd * (2.0f-one_frac);
  w3 = 1.0f/6.0f * squared * fraction;
}

// Inline calculation of the first order derivative bspline convolution weights, without conditional statements
template<class T> inline __device__ void bspline_weights_1st_derivative(T fraction, T& w0, T& w1, T& w2, T& w3)
{
  const T squared = fraction * fraction;

  w0 = -0.5f * squared + fraction - 0.5f;
  w1 =  1.5f * squared - 2.0f * fraction;
  w2 = -1.5f * squared + fraction + 0.5f;
  w3 =  0.5f * squared;
}

// Inline calculation of the second order derivative bspline convolution weights, without conditional statements
template<class T> inline __device__ void bspline_weights_2nd_derivative(T fraction, T& w0, T& w1, T& w2, T& w3)
{
  w0 =  1.0f - fraction;
  w1 =  3.0f * fraction - 2.0f;
  w2 = -3.0f * fraction + 1.0f;
  w3 =  fraction;
}

//! Bilinearly interpolated texture lookup, using unnormalized coordinates.
//! This function merely serves as a reference for the bicubic versions.
//! @param tex  2D texture
//! @param x  unnormalized x texture coordinate
//! @param y  unnormalized y texture coordinate
template<class T, enum cudaTextureReadMode mode>
__device__ float linearTex2D(texture<T, 2, mode> tex, float x, float y)
{
  return tex2D(tex, x, y);
}

//! Bicubic interpolated texture lookup, using unnormalized coordinates.
//! Straight forward implementation, using 16 nearest neighbour lookups.
//! @param tex  2D texture
//! @param x  unnormalized x texture coordinate
//! @param y  unnormalized y texture coordinate
template<class T, enum cudaTextureReadMode mode>
__device__ float cubicTex2DSimple(texture<T, 2, mode> tex, float x, float y)
{
  // transform the coordinate from [0,extent] to [-0.5, extent-0.5]
  const float2 coord_grid = make_float2(x - 0.5, y - 0.5);
  float2 index = floor(coord_grid);
  const float2 fraction = coord_grid - index;
  index.x += 0.5;  //move from [-0.5, extent-0.5] to [0, extent]
  index.y += 0.5;  //move from [-0.5, extent-0.5] to [0, extent]

  float result = 0.0;
  for (float y=-1; y < 2.5; y++)
  {
    float bsplineY = bspline(y-fraction.y);
    float v = index.y + y;
    for (float x=-1; x < 2.5; x++)
    {
      float bsplineXY = bspline(x-fraction.x) * bsplineY;
      float u = index.x + x;
      result += bsplineXY * tex2D(tex, u, v);
    }
  }
  return result;
}

//! Bicubic interpolated texture lookup, using unnormalized coordinates.
//! Fast implementation, using 4 trilinear lookups.
//! @param tex  2D texture
//! @param x  unnormalized x texture coordinate
//! @param y  unnormalized y texture coordinate
inline static __device__ float cubicTex2D(texture<float, 2> tex, float x, float y/*, const int width, const int height*/)
{
  // transform the coordinate from [0,extent] to [-0.5, extent-0.5]
  const float2 coord_grid = make_float2(x - 0.5f, y - 0.5f);
  const float2 index = floor(coord_grid);
  const float2 fraction = coord_grid - index;
  float2 w0, w1, w2, w3;
  bspline_weights(fraction, w0, w1, w2, w3);

  const float2 g0 = w0 + w1;
  const float2 g1 = w2 + w3;
  const float2 h0 = (w1 / g0) - make_float2(0.5f) + index;  //h0 = w1/g0 - 1, move from [-0.5, extent-0.5] to [0, extent]
  const float2 h1 = (w3 / g1) + make_float2(1.5f) + index;  //h1 = w3/g1 + 1, move from [-0.5, extent-0.5] to [0, extent]

  // fetch the four linear interpolations
  float tex00 = tex2D(tex, h0.x, h0.y);
  float tex10 = tex2D(tex, h1.x, h0.y);
  float tex01 = tex2D(tex, h0.x, h1.y);
  float tex11 = tex2D(tex, h1.x, h1.y);

  // weigh along the y-direction
  tex00 = g0.y * tex00 + g1.y * tex01;
  tex10 = g0.y * tex10 + g1.y * tex11;

  // weigh along the x-direction
  return (g0.x * tex00 + g1.x * tex10);
}

//#define WEIGHTS bspline_weights
//#define CUBICTEX2D cubicTex2D
//#include "cubictexture_kernels.cu"
//#undef CUBICTEX2D
//#undef WEIGHTS

// Fast bicubic interpolated 1st order derivative texture lookup in x- and
// y-direction, using unnormalized coordinates.
inline static __device__ void bspline_weights_1st_derivative_x(float2 fraction, float2& w0, float2& w1, float2& w2, float2& w3)
{
  float t0, t1, t2, t3;
  bspline_weights_1st_derivative(fraction.x, t0, t1, t2, t3);
  w0.x = t0; w1.x = t1; w2.x = t2; w3.x = t3;
  bspline_weights(fraction.y, t0, t1, t2, t3);
  w0.y = t0; w1.y = t1; w2.y = t2; w3.y = t3;
}

inline static  __device__ void bspline_weights_1st_derivative_y(float2 fraction, float2& w0, float2& w1, float2& w2, float2& w3)
{
  float t0, t1, t2, t3;
  bspline_weights(fraction.x, t0, t1, t2, t3);
  w0.x = t0; w1.x = t1; w2.x = t2; w3.x = t3;
  bspline_weights_1st_derivative(fraction.y, t0, t1, t2, t3);
  w0.y = t0; w1.y = t1; w2.y = t2; w3.y = t3;
}

#define WEIGHTS bspline_weights_1st_derivative_x
#define CUBICTEX2D cubicTex2D_1st_derivative_x
#include "cubictexture_kernels.cu"
#undef CUBICTEX2D
#undef WEIGHTS

#define WEIGHTS bspline_weights_1st_derivative_y
#define CUBICTEX2D cubicTex2D_1st_derivative_y
#include "cubictexture_kernels.cu"
#undef CUBICTEX2D
#undef WEIGHTS

} // namespace iuprivate

#endif // IUPRIVATE_CUBICTEXTURE_CU
