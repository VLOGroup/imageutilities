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
 * Project     : VMLibraries
 * Module      : common
 * Class       : none
 * Language    : CUDA
 * Description : Device functions for calculating derivatives and divergence.
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IU_DERIVATIVE_KERNELS_CUH
#define IU_DERIVATIVE_KERNELS_CUH

#include <cutil_math.h>


#ifdef __CUDACC__ // only include this in cuda files (seen by nvcc)

////////////////////////////////////////////////////////////////////////////////
/** Calculates x derivative with forward differences including zero border handling
 * @param tex  2D texture
 * @param x  x-coordinate
 * @param y  y-coordinate
 * @return x derivative calculated with forward differences.
 */
template<typename PixelType>
inline static __device__ PixelType dxp(const texture<PixelType, 2>& tex, const int x, const int y,
                                       const int width, const int height)
{
  PixelType val_x = tex2D(tex, x+1.5f, y+0.5f) - tex2D(tex, x+0.5f, y+0.5f);

  if (x >= width-1)
    val_x = 0.0f;

  return val_x;
}

/** Calculates x derivative with backward differences including zero border handling
 * @param tex  2D texture
 * @param x  x-coordinate
 * @param y  y-coordinate
 * @return x derivative calculated with backward differences.
 */
template<typename PixelType>
inline static __device__ PixelType dxm(const texture<PixelType, 2> tex, const int x, const int y,
                                       const int width, const int height)
{
  PixelType val = tex2D(tex, x+0.5f, y+0.5f);
  PixelType val_w = tex2D(tex, x-0.5f, y+0.5f);

  if (x >= width-1)
    val = 0.0f;
  if (x == 0)
    val_w = 0.0f;

  return (val-val_w);
}

////////////////////////////////////////////////////////////////////////////////
/** Calculates y derivative with forward differences including zero border handling
 * @param tex  2D texture
 * @param x  x-coordinate
 * @param y  y-coordinate
 * @return y derivative calculated with forward differences.
 */
template<typename PixelType>
inline static __device__ PixelType dyp(const texture<PixelType, 2>& tex, const int x, const int y,
                                       const int width, const int height)
{
  PixelType val_y = tex2D(tex, x+0.5f, y+1.5f) - tex2D(tex, x+0.5f, y+0.5f);

  if (y >= height-1)
    val_y = 0.0f;

  return val_y;
}

/** Calculates y derivative with backward differences including zero border handling
 * @param tex  2D texture
 * @param x  x-coordinate
 * @param y  y-coordinate
 * @return y derivative calculated with backward differences.
 */
template<typename PixelType>
inline static __device__ PixelType dym(const texture<PixelType, 2> tex, const int x, const int y,
                                       const int width, const int height)
{
  PixelType val = tex2D(tex, x+0.5f, y+0.5f);
  PixelType val_n = tex2D(tex, x+0.5f, y-0.5f);

  if (y >= height-1)
    val = 0.0f;
  if (y == 0)
    val_n = 0.0f;

  return (val-val_n);
}

////////////////////////////////////////////////////////////////////////////////

//-----------------------------------------------------------------------------
/* ok */
/** Calculates gradient with forward differences including zero border handling
 * @param tex  2D texture
 * @param x  x-coordinate
 * @param y  y-coordinate
 * @return y derivative calculated with forward differences.
 */
inline static __device__ float2 dp(const texture<float, 2> tex,
                                   const int x, const int y)
{
  float2 grad = make_float2(0.0f, 0.0f);
  float cval = tex2D(tex, x+0.5f, y+0.5f);
  grad.x = tex2D(tex, x+1.5f, y+0.5f) - cval;
  grad.y = tex2D(tex, x+0.5f, y+1.5f) - cval;
  return grad;
}

//-----------------------------------------------------------------------------
/* ok */
/** Calculates the divergence with backward differences including zero border handling
 * @param p  2D texture of p
 * @param x  x-coordinate
 * @param y  y-coordinate
 * @return divergence calculated with backward differences.
 */
inline static __device__ float dp_ad(const texture<float2, 2> tex,
                                     const int x, const int y, const int width, const int height)
{
  float2 cval = tex2D(tex, x+0.5f, y+0.5f);
  float2 wval = tex2D(tex, x-0.5f, y+0.5f);
  float2 nval = tex2D(tex, x+0.5f, y-0.5f);

  if (x == 0)
    wval.x = 0.0f;
  else if (x >= width-1)
    cval.x = 0.0f;


  if (y == 0)
    nval.y = 0.0f;
  else if (y >= height-1)
    cval.y = 0.0f;

  return (cval.x - wval.x + cval.y - nval.y);
}

////////////////////////////////////////////////////////////////////////////////

//-----------------------------------------------------------------------------
/* testing */
/** Calculates gradient with forward differences including zero border handling
 * @param u input
 * @param x  x-coordinate
 * @param y  y-coordinate
 * @return y derivative calculated with forward differences.
 */
inline static __device__ float2 dp(float* u, const int x, const int y, const size_t stride,
                                   const int width, const int height)
{
  float2 grad = make_float2(0.0f, 0.0f);
  if (x+1 < width)
    grad.x = u[y*stride+x+1] - u[y*stride+x];
  if (y+1 < height)
    grad.y = u[(y+1)*stride+x] - u[y*stride+x];
  return grad;
}

//-----------------------------------------------------------------------------
/* testing */
/** Calculates the divergence with backward differences including zero border handling
 * @param p
 * @param x  x-coordinate
 * @param y  y-coordinate
 * @return divergence calculated with backward differences.
 */
inline static __device__ float dp_ad(const float2* p,
                                     const int x, const int y, const size_t stride,
                                     const int width, const int height)
{
  float2 cval = p[y*stride+x];
  float2 wval = p[y*stride+x-1];
  float2 nval = p[(y-1)*stride+x];

  if (x == 0)
    wval.x = 0.0f;
  else if (x >= width-1)
    cval.x = 0.0f;

  if (y == 0)
    nval.y = 0.0f;
  else if (y >= height-1)
    cval.y = 0.0f;

  return (cval.x - wval.x + cval.y - nval.y);
}

////////////////////////////////////////////////////////////////////////////////

//-----------------------------------------------------------------------------
/* ok */
/** Calculates gradient with forward differences including zero border handling and edge weighting
 * @param tex  2D texture
 * @param tex_g 2D edge weight texture
 * @param x  x-coordinate
 * @param y  y-coordinate
 * @return y derivative calculated with forward differences.
 */
inline static __device__ float2 dp_weighted(const texture<float, 2> tex,
                                            const texture<float, 2> tex_g,
                                            const int x, const int y)
{
  const float xx = x+0.5f;
  const float yy = y+0.5f;
  float2 grad = make_float2(0.0f, 0.0f);
  float cval = tex2D(tex, xx, yy);

  // border handling is done via texture
  float g = tex2D(tex_g, xx, yy);
  grad.x = g*(tex2D(tex, xx+1.f, yy) - cval);
  grad.y = g*(tex2D(tex, xx, yy+1.f) - cval);

  return grad;
}

//-----------------------------------------------------------------------------
/* ok */
/** Calculates the divergence with backward differences including zero border handling
 * @param p  2D texture of p
 * @param tex_g 2D edge weight texture
 * @param x  x-coordinate
 * @param y  y-coordinate
 * @return divergence calculated with backward differences.
 */
inline static __device__ float dp_ad_weighted(const texture<float2, 2> tex,
                                              const texture<float, 2> tex_g,
                                              const int x, const int y,
                                              const int width, const int height)
{
  const float xx = x+0.5f;
  const float yy = y+0.5f;
  float2 c = tex2D(tex, xx,     yy);
  float2 w = tex2D(tex, xx-1.f, yy);
  float2 n = tex2D(tex, xx,     yy-1.f);

  float g =   tex2D(tex_g, xx,     yy);
  float g_w = tex2D(tex_g, xx-1.f, yy);
  float g_n = tex2D(tex_g, xx,     yy-1.f);

  if (x == 0)
    w.x = 0.0f;
  else if (x >= width-1)
    c.x = 0.0f;
  if (y == 0)
    n.y = 0.0f;
  else if (y >= height-1)
    c.y = 0.0f;

  return (c.x*g - w.x*g_w + c.y*g - n.y*g_n);
}

////////////////////////////////////////////////////////////////////////////////

//-----------------------------------------------------------------------------
/* ok */
/** Calculates gradient with forward differences including zero border handling and edge weighting
 * @param tex  2D texture
 * @param tex_gx 2D x-gradient texture
 * @param tex_gy 2D y-gradient texture
 * @param x  x-coordinate
 * @param y  y-coordinate
 * @return y derivative calculated with forward differences.
 */
inline static __device__ float2 dp_edges(const texture<float, 2> tex,
                                         const texture<float, 2> tex_gx,
                                         const texture<float, 2> tex_gy,
                                         const int x, const int y)
{
  const float xx = x+0.5f;
  const float yy = y+0.5f;
  float2 grad = make_float2(0.0f, 0.0f);
  float cval = tex2D(tex, xx, yy);

  // border handling is done via texture

  grad.x = tex2D(tex_gx, xx, yy)*(tex2D(tex, xx+1.f, yy) - cval);
  grad.y = tex2D(tex_gy, xx, yy)*(tex2D(tex, xx, yy+1.f) - cval);

  return grad;
}

//-----------------------------------------------------------------------------
/* ok */
/** Calculates the divergence with backward differences including zero border handling
 * @param p  2D texture of p
 * @param x  x-coordinate
 * @param y  y-coordinate
 * @return divergence calculated with backward differences.
 */
inline static __device__ float dp_ad_edges(const texture<float2, 2> tex,
                                           const texture<float, 2> tex_gx,
                                           const texture<float, 2> tex_gy,
                                           const int x, const int y,
                                           const int width, const int height)
{
  const float xx = x+0.5f;
  const float yy = y+0.5f;
  float2 c = tex2D(tex, xx,     yy);
  float2 w = tex2D(tex, xx-1.f, yy);
  float2 n = tex2D(tex, xx,     yy-1.f);

  float gx =   tex2D(tex_gx, xx,     yy);
  float gx_w = tex2D(tex_gx, xx-1.f, yy);
  float gy =   tex2D(tex_gy, xx,     yy);
  float gy_n = tex2D(tex_gy, xx,     yy-1.f);

  if (x == 0)
    w.x = 0.0f;
  else if (x >= width-1)
    c.x = 0.0f;
  if (y == 0)
    n.y = 0.0f;
  else if (y >= height-1)
    c.y = 0.0f;

  return (c.x*gx - w.x*gx_w + c.y*gy - n.y*gy_n);
}

//-----------------------------------------------------------------------------
/** Calculates the divergence with backward differences including zero border handling
 * @param p  2D texture of p (special case with only one chanel)
 * @param x  x-coordinate
 * @param y  y-coordinate
 * @return divergence calculated with backward differences.
 */
inline static __device__ float dp_ad_edges(const texture<float, 2> tex,
                                           const texture<float, 2> tex_gx,
                                           const texture<float, 2> tex_gy,
                                           const int x, const int y,
                                           const int width, const int height)
{
  const float xx = x+0.5f;
  const float yy = y+0.5f;
  float c = tex2D(tex, xx,     yy);
  float w = tex2D(tex, xx-1.f, yy);
  float n = tex2D(tex, xx,     yy-1.f);

  float gx =   tex2D(tex_gx, xx,     yy);
  float gx_w = tex2D(tex_gx, xx-1.f, yy);
  float gy =   tex2D(tex_gy, xx,     yy);
  float gy_n = tex2D(tex_gy, xx,     yy-1.f);

  if (x == 0)
    w = 0.0f;
  else if (x >= width-1)
    c = 0.0f;
  if (y == 0)
    n = 0.0f;
  else if (y >= height-1)
    c = 0.0f;

  return (c*gx - w*gx_w + c*gy - n*gy_n);
}


//-----------------------------------------------------------------------------
/** Calculates gradient with forward differences including zero border handling and edge weighting
 * @param tex  2D texture
 * @param tex_gx 2D x-gradient texture
 * @param tex_gy 2D y-gradient texture
 * @param x  x-coordinate
 * @param y  y-coordinate
 * @return y derivative calculated with forward differences.
 */
inline static __device__ float2 dp_edges(const texture<float, 2> tex,
                                         const texture<float2, 2> tex_g,
                                         const int x, const int y)
{
  const float xx = x+0.5f;
  const float yy = y+0.5f;
  float2 grad = make_float2(0.0f, 0.0f);
  float cval = tex2D(tex, xx, yy);
  float2 g = tex2D(tex_g, xx, yy);

  // border handling is done via texture

  grad.x = g.x*(tex2D(tex, xx+1.f, yy) - cval);
  grad.y = g.y*(tex2D(tex, xx, yy+1.f) - cval);

  return grad;
}

//-----------------------------------------------------------------------------
/** Calculates the divergence with backward differences including zero border handling
 * @param p  2D texture of p
 * @param x  x-coordinate
 * @param y  y-coordinate
 * @return divergence calculated with backward differences.
 */
inline static __device__ float dp_ad_edges(const texture<float2, 2> tex,
                                           const texture<float2, 2> tex_g,
                                           const int x, const int y,
                                           const int width, const int height)
{
  const float xx = x+0.5f;
  const float yy = y+0.5f;
  float2 c = tex2D(tex, xx,     yy);
  float2 w = tex2D(tex, xx-1.f, yy);
  float2 n = tex2D(tex, xx,     yy-1.f);

  float2 g   = tex2D(tex_g, xx,     yy);
  float2 g_w = tex2D(tex_g, xx-1.f, yy);
  float2 g_n = tex2D(tex_g, xx,     yy-1.f);

  if (x == 0)
    w.x = 0.0f;
  else if (x >= width-1)
    c.x = 0.0f;
  if (y == 0)
    n.y = 0.0f;
  else if (y >= height-1)
    c.y = 0.0f;

  return (c.x*g.x - w.x*g_w.x + c.y*g.y - n.y*g_n.y);
}


////////////////////////////////////////////////////////////////////////////////

//-----------------------------------------------------------------------------
/* ok */
/** Calculates gradient with forward differences including zero border handling and edge weighting
 * @param tex  2D texture
 * @param tex_gx 2D x-gradient texture
 * @param tex_gy 2D y-gradient texture
 * @param x  x-coordinate
 * @param y  y-coordinate
 * @return y derivative calculated with forward differences.
 */
inline static __device__ float2 dp_tensor(const texture<float, 2> tex,
                                          const texture<float, 2> tex_rwa,
                                          const texture<float, 2> tex_rwb,
                                          const texture<float, 2> tex_rwc,
                                          const int x, const int y)
{
  const float xx = x+0.5f;
  const float yy = y+0.5f;
  float2 grad = make_float2(0.0f, 0.0f);
  float cval = tex2D(tex, xx, yy);

  // border handling is done via texture
  grad.x =
      tex2D(tex_rwa, xx, yy)*(tex2D(tex, xx+1.f, yy     ) - cval) +
      tex2D(tex_rwc, xx, yy)*(tex2D(tex, xx    , yy+1.0f) - cval);
  grad.y =
      tex2D(tex_rwc, xx, yy)*(tex2D(tex, xx+1.0f, yy    ) - cval) +
      tex2D(tex_rwb, xx, yy)*(tex2D(tex, xx     , yy+1.f) - cval);

  return grad;
}

//-----------------------------------------------------------------------------
/* ok */
/** Calculates the divergence with backward differences including zero border handling
 * @param p  2D texture of p
 * @param x  x-coordinate
 * @param y  y-coordinate
 * @return divergence calculated with backward differences.
 */
inline static __device__ float dp_ad_tensor(const texture<float2, 2> tex,
                                            const texture<float, 2> tex_rwa,
                                            const texture<float, 2> tex_rwb,
                                            const texture<float, 2> tex_rwc,
                                            const int x, const int y,
                                            const int width, const int height)
{
  const float xx = x+0.5f;
  const float yy = y+0.5f;
  float2 c = tex2D(tex, xx,     yy);
  float2 w = tex2D(tex, xx-1.f, yy);
  float2 n = tex2D(tex, xx,     yy-1.f);

  float rwa =   tex2D(tex_rwa, xx,     yy);
  float rwa_w = tex2D(tex_rwa, xx-1.f, yy);
  float rwb =   tex2D(tex_rwb, xx,     yy);
  float rwb_n = tex2D(tex_rwb, xx,     yy-1.f);
  float rwc =   tex2D(tex_rwc, xx,     yy);
  float rwc_w = tex2D(tex_rwc, xx-1.f, yy);
  float rwc_n = tex2D(tex_rwc, xx,     yy-1.f);

  if (x == 0)
    w = make_float2(0.0f,0.0f);
  else if (x >= width-1)
    c.x = 0.0f;
  if (y == 0)
    n = make_float2(0.0f,0.0f);
  else if (y >= height-1)
    c.y = 0.0f;

  return ((c.x*rwa + c.y*rwc) - (w.x*rwa_w + w.y*rwc_w) +
          (c.x*rwc + c.y*rwb) - (n.x*rwc_n + n.y*rwb_n));
}

//-----------------------------------------------------------------------------
/* ok */
/** Calculates gradient with forward differences including zero border handling for tgv2 model
 * @param tex  2D texture
 * @param x  x-coordinate
 * @param y  y-coordinate
 * @return y derivative calculated with forward differences.
 */
inline static __device__ float4 dp_tgv2(const texture<float2, 2> tex, const int x, const int y)
{
  float4 grad = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float2 cval = tex2D(tex, x+0.5f, y+0.5f);
  grad.x = tex2D(tex, x+1.5f, y+0.5f).x - cval.x;
  grad.y = tex2D(tex, x+0.5f, y+1.5f).y - cval.y;
  grad.z = tex2D(tex, x+0.5f, y+1.5f).x - cval.x;
  grad.w = tex2D(tex, x+1.5f, y+0.5f).y - cval.y;

  return grad;
}


//-----------------------------------------------------------------------------
/* ok */
/** Calculates the divergence with backward differences including zero border handling for tgv2 model
 * @param p  2D texture of p
 * @param x  x-coordinate
 * @param y  y-coordinate
 * @return divergence calculated with backward differences.
 */
inline static __device__ float2 dp_ad_tgv2(const texture<float4, 2> tex,
                                           const int x, const int y,
                                           const int width, const int height)
{
  float4 cval = tex2D(tex, x+0.5f, y+0.5f);
  float4 wval = tex2D(tex, x-0.5f, y+0.5f);
  float4 nval = tex2D(tex, x+0.5f, y-0.5f);

  if (x == 0)
  {
    wval.x = 0.0f;
    wval.w = 0.0f;
  }
  else if (x >= width-1)
  {
    cval.x = 0.0f;
    cval.w = 0.0f;
  }

  if (y == 0)
  {
    nval.y = 0.0f;
    nval.z = 0.0f;
  }
  else if (y >= height-1)
  {
    cval.z = 0.0f;
    cval.y = 0.0f;
  }

  return make_float2(cval.x-wval.x + cval.z-nval.z, cval.w-wval.w + cval.y-nval.y);
}


////////////////////////////////////////////////////////////////////////////////


//-----------------------------------------------------------------------------
/** Calculates the divergence with backward differences including zero border handling
 * @param px 2D texture for x component of p
 * @param py 2D texture for y component of p
 * @param x  x-coordinate
 * @param y  y-coordinate
 * @return divergence calculated with backward differences.
 */
template<typename PixelType>
inline static __device__ PixelType dp_ad(const texture<PixelType, 2> tex_px, const texture<PixelType, 2> tex_py,
                                         const int x, const int y, const int width, const int height)
{
  PixelType px = tex2D(tex_px, x+0.5f, y+0.5f);
  PixelType px_w = tex2D(tex_px, x-0.5f, y+0.5f);
  PixelType py = tex2D(tex_py, x+0.5f, y+0.5f);
  PixelType py_n = tex2D(tex_py, x+0.5f, y-0.5f);

  if (x >= width-1)
    px = 0.0f;
  if (x == 0)
    px_w = 0.0f;

  if (y >= height-1)
    py = 0.0f;
  if (y == 0)
    py_n = 0.0f;

  return ((px-px_w) + (py-py_n));
}

////////////////////////////////////////////////////////////////////////////////
/** Calculates x derivative with central differences including zero border handling
 * @param tex  2D texture
 * @param x  x-coordinate
 * @param y  y-coordinate
 * @return x derivative calculated with central differences.
 */
template<typename PixelType>
inline static __device__ PixelType dxc(const texture<PixelType, 2> tex, const int x, const int y,
                                       const int width, const int height)
{
  if ((x == 0) || (x >= width-1))
    return 0.0f;
  else
    return 0.5f*(tex2D(tex, x+1.5f, y+0.5f) - tex2D(tex, x-0.5f, y+0.5f));
}

/** Calculates adjungated x derivative with central differences including zero border handling
 * @param tex  2D texture
 * @param x  x-coordinate
 * @param y  y-coordinate
 * @return y derivative calculated with central differences.
 */
template<typename PixelType>
inline static __device__ PixelType dxc_ad(const texture<PixelType, 2> tex, const int x, const int y,
                                          const int width, const int height)
{
  if (x <= 1)
    return 0.5f*tex2D(tex, x+1.5f, y+0.5f);
  else if (x >= width-2)
    return -0.5f*tex2D(tex, x-0.5f, y+0.5f);
  else
    return 0.5f*(tex2D(tex, x+1.5f, y+0.5f) - tex2D(tex, x-0.5f, y+0.5f));
}

////////////////////////////////////////////////////////////////////////////////
/** Calculates y derivative with central differences including zero border handling
 * @param tex  2D texture
 * @param x  x-coordinate
 * @param y  y-coordinate
 * @return y derivative calculated with central differences.
 */
template<typename PixelType>
inline static __device__ PixelType dyc(const texture<PixelType, 2> tex, const int x, const int y,
                                       const int width, const int height)
{
  if ((y == 0) || (y >= height-1))
    return 0.0f;
  else
    return 0.5f*(tex2D(tex, x+0.5f, y+1.5f) - tex2D(tex, x+0.5f, y-0.5f));
}

/** Calculates adjungated y derivative with central differences including zero border handling
 * @param tex  2D texture
 * @param x  x-coordinate
 * @param y  y-coordinate
 * @return y derivative calculated with central differences.
 */
template<typename PixelType>
inline static __device__ PixelType dyc_ad(const texture<PixelType, 2> tex, const int x, const int y,
                                          const int width, const int height)
{
  if (y <= 1)
    return 0.5f*tex2D(tex, x+0.5f, y+1.5f);
  else if (y >= height-2)
    return -0.5f*tex2D(tex, x+0.5f, y-0.5f);
  else
    return 0.5f*(tex2D(tex, x+0.5f, y+1.5f) - tex2D(tex, x+0.5f, y-0.5f));
}

////////////////////////////////////////////////////////////////////////////////
/** Calculates gradient with central differences including zero border handling
 * @param tex  2D texture
 * @param x  x-coordinate
 * @param y  y-coordinate
 * @return y derivative calculated with forward differences.
 */
inline static __device__ float2 dc(const texture<float, 2> tex, const int x, const int y,
                                   const int width, const int height)
{
  float2 grad = make_float2(0.0f, 0.0f);
  if ((x > 0) && (x < width-1))
    grad.x = 0.5f*(tex2D(tex, x+1.5f, y+0.5f) - tex2D(tex, x-0.5f, y+0.5f));
  if ((y > 0) && (y < height-1))
    grad.y = 0.5f*(tex2D(tex, x+0.5f, y+1.5f) - tex2D(tex, x+0.5f, y-0.5f));
  return grad;
}

/** Calculates the divergence with central differences including zero border handling
 * @param p  2D texture of p
 * @param x  x-coordinate
 * @param y  y-coordinate
 * @return divergence calculated with backward differences.
 */
inline static __device__ float dc_ad(const texture<float2, 2> tex,
                                     const int x, const int y, const int width, const int height)
{
  // load data
  float2 wval = tex2D(tex, x-0.5f, y+0.5f);
  float2 nval = tex2D(tex, x+0.5f, y-0.5f);
  float2 eval = tex2D(tex, x+1.5f, y+0.5f);
  float2 sval = tex2D(tex, x+0.5f, y+1.5f);

  float div = 0.0f;

  // x component
  if (x <= 1)
    div += 0.5f*eval.x;
  else if (x >= width-2)
    div += -0.5f*wval.x;
  else
    div += 0.5f*(eval.x - wval.x);

  // y component
  if (y <= 1)
    div += 0.5f*sval.y;
  else if (y >= height-2)
    div += -0.5f*nval.y;
  else
    div += 0.5f*(sval.y - nval.y);

  return div;
}


////////////////////////////////////////////////////////////////////////////////
/** Calculates gradient with weighted forward differences including zero border handling
 * @param tex  2D texture of primal variable
 * @param x    x-coordinate
 * @param y    y-coordinate
 * @return y derivative calculated with forward differences.
 */
inline static __device__ float2 wdp(const texture<float, 2> tex,  const texture<float, 2> weight,
                                    const int x, const int y)
{
  float cval = tex2D(tex, x+0.5f, y+0.5f);
  float2 grad = make_float2(tex2D(tex, x+1.5f, y+0.5f) - cval,
                            tex2D(tex, x+0.5f, y+1.5f) - cval);
  return tex2D(weight, x+0.5f, y+0.5f)*grad;
}

/** Calculates the divergence with weighted backward differences including zero border handling
 * @param tex    2D texture of dual variable
 * @param weight 2D texture of weight variable (scalar)
 * @param x      x-coordinate
 * @param y      y-coordinate
 * @return divergence calculated with backward differences.
 */
inline static __device__ float wdp_ad(const texture<float2, 2> tex, const texture<float, 2> weight,
                                      const int x, const int y, const int width, const int height)
{
  float2 cval = tex2D(weight, x+0.5f, y+0.5f)*tex2D(tex, x+0.5f, y+0.5f);
  float2 wval = tex2D(weight, x-0.5f, y+0.5f)*tex2D(tex, x-0.5f, y+0.5f);
  float2 nval = tex2D(weight, x+0.5f, y-0.5f)*tex2D(tex, x+0.5f, y-0.5f);

  if (x == 0)
    wval.x = 0.0f;
  else if (x >= width-1)
    cval.x = 0.0f;
  if (y == 0)
    nval.y = 0.0f;
  else if (y >= height-1)
    cval.y = 0.0f;

  return (cval.x - wval.x + cval.y - nval.y);
}

////////////////////////////////////////////////////////////////////////////////
/** Calculates the divergence with central differences including zero border handling
 * @param px 2D texture for x component of p
 * @param py 2D texture for y component of p
 * @param x  x-coordinate
 * @param y  y-coordinate
 * @return divergence calculated with central differences.
 */
template<typename PixelType>
inline static __device__ PixelType dc_ad(const texture<PixelType, 2> tex_px, const texture<PixelType, 2> tex_py,
                                         const int x, const int y, const int width, const int height)
{
  PixelType px_e = tex2D(tex_px, x+1.5f, y+0.5f);
  PixelType px_w = tex2D(tex_px, x-0.5f, y+0.5f);
  PixelType py_s = tex2D(tex_py, x+0.5f, y+1.5f);
  PixelType py_n = tex2D(tex_py, x+0.5f, y-0.5f);

  if (x >= width-2)
    px_e = 0.0f;
  if (x <= 1)
    px_w = 0.0f;

  if (y >= height-2)
    py_s = 0.0f;
  if (y <= 1)
    py_n = 0.0f;

  return (0.5f*(px_e-px_w) + 0.5f*(py_s-py_n));
}

////////////////////////////////////////////////////////////////////////////////
/** Calculates x derivative with forward differences including zero border handling and edge weighting
 * @param tex  2D texture
 * @param x  x-coordinate
 * @param y  y-coordinate
 * @return x derivative calculated with forward differences.
 */
template<typename PixelType>
inline static __device__ PixelType dxp_edges(const texture<PixelType, 2>& tex, const texture<PixelType, 2> tex_gx,
                                             const int x, const int y,
                                             const int width, const int height)
{
  PixelType val_x = tex2D(tex, x+1.5f, y+0.5f) - tex2D(tex, x+0.5f, y+0.5f);

  if (x >= width-1)
    val_x = 0.0f;

  return val_x*tex2D(tex_gx, x+0.5f, y+0.5f);
}

////////////////////////////////////////////////////////////////////////////////
/** Calculates y derivative with forward differences including zero border handlingand edge weighting
 * @param tex  2D texture
 * @param x  x-coordinate
 * @param y  y-coordinate
 * @return y derivative calculated with forward differences.
 */
template<typename PixelType>
inline static __device__ PixelType dyp_edges(const texture<PixelType, 2>& tex, const texture<PixelType, 2> tex_gy,
                                             const int x, const int y,
                                             const int width, const int height)
{
  PixelType val_y = tex2D(tex, x+0.5f, y+1.5f) - tex2D(tex, x+0.5f, y+0.5f);

  if (y >= height-1)
    val_y = 0.0f;

  return val_y*tex2D(tex_gy, x+0.5f, y+0.5f);
}

/** Calculates the divergence with backward differences including zero border handling and edge weighting
 * @param px 2D texture for x component of p
 * @param py 2D texture for y component of p
 * @param gx 2D texture for edge weights in x-direction
 * @param gy 2D texture for edge weights in y-direction
 * @param x  x-coordinate
 * @param y  y-coordinate
 * @return y derivative calculated with backward differences.
 */
template<typename PixelType>
inline static __device__ PixelType dp_ad_edges(const texture<PixelType, 2> tex_px, const texture<PixelType, 2> tex_py,
                                               const texture<PixelType, 2> tex_gx, const texture<PixelType, 2> tex_gy,
                                               const int x, const int y, const int width, const int height)
{
  PixelType px = tex2D(tex_px, x+0.5f, y+0.5f);
  PixelType px_w = tex2D(tex_px, x-0.5f, y+0.5f);
  PixelType py = tex2D(tex_py, x+0.5f, y+0.5f);
  PixelType py_n = tex2D(tex_py, x+0.5f, y-0.5f);

  PixelType gx = tex2D(tex_gx, x+0.5f, y+0.5f);
  PixelType gx_w = tex2D(tex_gx, x-0.5f, y+0.5f);
  PixelType gy = tex2D(tex_gx, x+0.5f, y+0.5f);
  PixelType gy_n = tex2D(tex_gx, x+0.5f, y-0.5f);

  if (x >= width-1)
    px = 0.0f;
  if (x == 0)
    px_w = 0.0f;

  if (y >= height-1)
    py = 0.0f;
  if (y == 0)
    py_n = 0.0f;

  return ((px*gx-px_w*gx_w) + (py*gy-py_n*gy_n));
}

//! TODO divc_edges
/* */
//


/** Calculates the divergence with backward differences including zero border handling and edge weighting
 * @param px 2D texture for x component of p
 * @param py 2D texture for y component of p
 * @param rwa matrix entry a for regularization weight (anistropic regularity)
 * @param rwb matrix entry b for regularization weight (anistropic regularity)
 * @param rwc matrix entry c for regularization weight (anistropic regularity)
 * @param x  x-coordinate
 * @param y  y-coordinate
 * @return y derivative calculated with backward differences.
 */
template<typename PixelType>
inline static __device__ PixelType dp_ad_anisotropic(const texture<PixelType, 2> tex_px, const texture<PixelType, 2> tex_py,
                                                     const texture<PixelType, 2> tex_rwa, const texture<PixelType, 2> tex_rwb,
                                                     const texture<PixelType, 2> tex_rwc,
                                                     const int x, const int y, const int width, const int height)
{
  PixelType px = tex2D(tex_px, x+0.5f, y+0.5f);
  PixelType px_w = tex2D(tex_px, x-0.5f, y+0.5f);
  PixelType px_n = tex2D(tex_px, x+0.5f, y-0.5f);
  PixelType py = tex2D(tex_py, x+0.5f, y+0.5f);
  PixelType py_w = tex2D(tex_py, x-0.5f, y+0.5f);
  PixelType py_n = tex2D(tex_py, x+0.5f, y-0.5f);

  if (x >= width-1)
    px = 0.0f;
  if (x == 0)
  {
    px_w = 0.0f;
    py_w = 0.0f;
  }

  if (y >= height-1)
    py = 0.0f;
  if (y == 0)
  {
    px_n = 0.0f;
    py_n = 0.0f;
  }

  PixelType rwa = tex2D(tex_rwa, x+0.5f, y+0.5f);
  PixelType rwa_w = tex2D(tex_rwa, x-0.5f, y+0.5f);
  PixelType rwb = tex2D(tex_rwb, x+0.5f, y+0.5f);
  PixelType rwb_n = tex2D(tex_rwb, x+0.5f, y-0.5f);
  PixelType rwc = tex2D(tex_rwc, x+0.5f, y+0.5f);
  PixelType rwc_w = tex2D(tex_rwc, x-0.5f, y+0.5f);
  PixelType rwc_n = tex2D(tex_rwc, x+0.5f, y-0.5f);

  return(
      (px*rwa + py*rwc) - (px_w*rwa_w + py_w*rwc_w) +
      (px*rwc + py*rwb) - (px_n*rwc_n + py_n*rwb_n) );
}

//! TODO divc_anisotropic
/* */
//

//-----------------------------------------------------------------------------
/* TODO: check me */
/** Calculates the divergence with backward differences including zero border handling
 * @param p  2D texture of p
 * @param x  x-coordinate
 * @param y  y-coordinate
 * @return divergence calculated with backward differences and linear subpixel interpolation.
 */
inline static __device__ float dp_ad_ip(const texture<float4, 2> tex,
                                        const int x, const int y, const int width, const int height)
{
  float4 cval = tex2D(tex, x+0.5f, y+0.5f);
  float4 cval_ip = tex2D(tex, x, y);

  float4 wval = tex2D(tex, x-0.5f, y+0.5f);
  float4 wval_ip = tex2D(tex, x+1.0f, y);

  float4 nval = tex2D(tex, x+0.5f, y-0.5f);
  float4 nval_ip = tex2D(tex, x, y+1.0f);

  float div = 0.0f;

  if (x == 0)
  {
    wval.x = 0.0f;
  }
  else if (x >= width-1)
  {
    cval.x = 0.0f;
  }


  if (y == 0)
  {
    nval.y = 0.0f;
  }
  else if (y >= height-1)
  {
    cval.y = 0.0f;
  }

  div -=  cval.x - wval.x;    //dxm(p11)
  div -=  cval.w - nval.w;    //dym(p22)

  // TODO: Check borderhandling
  div += cval_ip.z - wval_ip.z;  //dxm_ip(p12)
  div += cval_ip.y - nval_ip.y; //dxm_ip(p21)

  return div;
}


//-----------------------------------------------------------------------------
/* TODO: check me */
/** Calculates gradient with forward differences including zero border handling
 * @param tex  2D texture
 * @param x  x-coordinate
 * @param y  y-coordinate
 * @return y derivative calculated with forward differences and linear subpixel interpolation.
 */
inline static __device__ float4 dp_ip(const texture<float, 2> tex, const int x, const int y)
{
  float4 grad = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float cval = tex2D(tex, x+0.5f, y+0.5f);
  float cval_sw = tex2D(tex, x+1.0f, y+1.0f);

  grad.x = tex2D(tex, x+1.5f, y+0.5f) - cval;
  grad.y = cval_sw - tex2D(tex, x+1.0f, y);
  grad.z = cval_sw - tex2D(tex, x, y+1.0f);
  grad.w = tex2D(tex, x+0.5f, y+1.5f) - cval;

  return grad;
}

#endif // __CUDACC__



#endif // IU_DERIVATIVE_KERNELS_CUH
