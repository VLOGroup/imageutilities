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
