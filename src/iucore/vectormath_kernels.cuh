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
 * Module      : global
 * Class       : none
 * Language    : C/CUDA
 * Description : Common cuda functionality that might also be interesting for other applications.
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IU_VECTORMATH_KERNELS_CUH
#define IU_VECTORMATH_KERNELS_CUH

namespace iu {
//
/** Round a / b to nearest higher integer value.
 * @param[in] a Numerator
 * @param[in] b Denominator
 * @return a / b rounded up
 */
static inline __host__ __device__ unsigned int divUp(unsigned int a, unsigned int b)
{
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

template<typename PixelType>
static inline __host__ __device__ PixelType sqr(PixelType a)
{
  return a*a;
}

} // namespace iu


///////////////////////////////////////////////////////////////////////////////
// the operator stuff is not part of the iu namespace
// so it can be used without #using namespace iu stuff!
//
// NOTE: it might occur that some of these are defined in future cuda versions
//       which causes a namespace clash. As a workaround simply delete the affected
//       operator after checking that it really does the same thing
//       or check the cuda version (which will be included in future releases here).
///////////////////////////////////////////////////////////////////////////////


/* ****************************************************************************
 *  uchar2 functions
 * ****************************************************************************/

// create uchar2 from a single uchar
static inline __host__ __device__ uchar2 make_uchar2(unsigned char x)
{
  uchar2 t; t.x = x; t.y = x; return t;
}

// !=
static inline __host__ __device__ bool operator!=(uchar2& a, uchar2& b)
{
  return (a.x != b.x) || (a.y != b.y);
}

// ==
static inline __host__ __device__ bool operator==(uchar2& a, uchar2& b)
{
  return (a.x == b.x) && (a.y == b.y);
}

/* ****************************************************************************
 *  uchar3 functions
 * ****************************************************************************/

// create uchar3 from a single uchar
static inline __host__ __device__ uchar3 make_uchar3(unsigned char x)
{
  uchar3 t; t.x = x; t.y = x; t.z = x; return t;
}

// !=
static inline __host__ __device__ bool operator!=(uchar3& a, uchar3& b)
{
  return (a.x != b.x) || (a.y != b.y) || (a.z != b.z);
}

// ==
static inline __host__ __device__ bool operator==(uchar3& a, uchar3& b)
{
  return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);
}

/* ****************************************************************************
 *  uchar4 functions
 * ****************************************************************************/

// create uchar4 from a single uchar
static inline __host__ __device__ uchar4 make_uchar4(unsigned char x)
{
  uchar4 t; t.x = x; t.y = x; t.z = x; t.w = x; return t;
}

// !=
static inline __host__ __device__ bool operator!=(uchar4& a, uchar4& b)
{
  return (a.x != b.x) || (a.y != b.y) || (a.z != b.z) || (a.w != b.w);
}

// ==
static inline __host__ __device__ bool operator==(uchar4& a, uchar4& b)
{
  return (a.x == b.x) && (a.y == b.y) && (a.z == b.z) && (a.w == b.w);
}

// multiply with constant
static inline __host__ __device__ uchar4 operator*(uchar4 a, unsigned char s)
{
    return make_uchar4(a.x * s, a.y * s, a.z * s,  a.w * s);
}
static inline __host__ __device__ uchar4 operator*(unsigned char s, uchar4 a)
{
    return make_uchar4(a.x * s, a.y * s, a.z * s,  a.w * s);
}
// elementwise multiply
static inline __host__ __device__ uchar4 operator*(uchar4 a, uchar4 b)
{
    return make_uchar4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}

/* ****************************************************************************
 *  float2 functions
 * ****************************************************************************/


// !=
static inline __host__ __device__ bool operator!=(float2& a, float2& b)
{
  return (a.x != b.x) || (a.y != b.y);
}

// ==
static inline __host__ __device__ bool operator==(float2& a, float2& b)
{
  return (a.x == b.x) && (a.y == b.y);
}

// float-float2
#ifndef CUDA_VERSION_32
static inline __host__ __device__ float2 operator-(float a, float2 b)
{
  return make_float2(a - b.x, a - b.y);
}
#endif

/* ****************************************************************************
 *  float3 functions
 * ****************************************************************************/

// !=
static inline __host__ __device__ bool operator!=(float3& a, float3& b)
{
  return (a.x != b.x) || (a.y != b.y) || (a.z != b.z);
}

// ==
static inline __host__ __device__ bool operator==(float3& a, float3& b)
{
  return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);
}

// float-float3
#ifndef CUDA_VERSION_32
static inline __host__ __device__ float3 operator-(float a, float3 b)
{
  return make_float3(a - b.x, a - b.y, a - b.z);
}
#endif

/* ****************************************************************************
 *  float4 functions
 * ****************************************************************************/

// !=
static inline __host__ __device__ bool operator!=(float4& a, float4& b)
{
  return (a.x != b.x) || (a.y != b.y) || (a.z != b.z) || (a.w != b.w);
}

// ==
static inline __host__ __device__ bool operator==(float4& a, float4& b)
{
  return (a.x == b.x) && (a.y == b.y) && (a.z == b.z) && (a.w == b.w);
}

// elementwise multiply
#ifndef CUDA_VERSION_32
static inline __host__ __device__ float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
#endif

#endif // IU_VECTORMATH_KERNELS_CUH
