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
 * Module      : Global
 * Class       : none
 * Language    : C++
 * Description : Global typedefinitions and macros for ImageUtilities. (e.g. dll export stuff, ...)
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IU_CUTIL_H
#define IU_CUTIL_H

#include <cutil_math.h>

//// ERROR CHECKS ////////////////////////////////////////////
#ifdef __CUDACC__ // only include this error check in cuda files (seen by nvcc)

// MACROS
//

//-----------------------------------------------------------------------------
#define __IU_CHECK_FOR_CUDA_ERRORS_ENABLED__ // enables checking for cuda errors
/** CUDA ERROR HANDLING (CHECK FOR CUDA ERRORS)
 */
#ifdef __IU_CHECK_FOR_CUDA_ERRORS_ENABLED__
#include <stdio.h>
#define IU_CHECK_CUDA_ERRORS() \
do { \
  cudaThreadSynchronize(); \
  if (cudaError_t err = cudaGetLastError()) \
  { \
    fprintf(stderr,"\n\n ImageUtilities: CUDA Error: %s\n",cudaGetErrorString(err)); \
    fprintf(stderr,"  file:       %s\n",__FILE__); \
    fprintf(stderr,"  function:   %s\n",__FUNCTION__); \
    fprintf(stderr,"  line:       %d\n\n",__LINE__); \
    return NPP_ERROR; \
  } \
} while(false)
#else // __IU_CHECK_FOR_CUDA_ERRORS_ENABLED__
#define IU_CHECK_CUDA_ERRORS() {}
#endif // __IU_CHECK_FOR_CUDA_ERRORS_ENABLED__


#endif // __CUDACC__



//// SMALL CUDA HELPERS (DEFINED INLINE) ////////////////////////////////////////////
namespace iu {
//
/** Round a / b to nearest higher integer value.
 * @param[in] a Numerator
 * @param[in] b Denominator
 * @return a / b rounded up
 */
__host__ __device__ inline unsigned int divUp(unsigned int a, unsigned int b)
{
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

__host__ __device__ inline float sqr(float a)
{
  return a*a;
}


} // namespace iu


///////////////////////////////////////////////////////////////////////////////
// the operator stuff is not part of the iu namespace
// so it can be used wituout this using iu stuff!
///////////////////////////////////////////////////////////////////////////////


/* ****************************************************************************
 *  uchar2 functions
 * ****************************************************************************/

// create uchar2 from a single uchar
static __inline__ __host__ __device__ uchar2 make_uchar2(unsigned char x)
{
  uchar2 t; t.x = x; t.y = x; return t;
}

// !=
inline __host__ __device__ bool operator!=(uchar2& a, uchar2& b)
{
  return (a.x != b.x) || (a.y != b.y);
}

// ==
inline __host__ __device__ bool operator==(uchar2& a, uchar2& b)
{
  return (a.x == b.x) && (a.y == b.y);
}

/* ****************************************************************************
 *  uchar3 functions
 * ****************************************************************************/

// create uchar3 from a single uchar
static __inline__ __host__ __device__ uchar3 make_uchar3(unsigned char x)
{
  uchar3 t; t.x = x; t.y = x; t.z = x; return t;
}

// !=
inline __host__ __device__ bool operator!=(uchar3& a, uchar3& b)
{
  return (a.x != b.x) || (a.y != b.y) || (a.z != b.z);
}

// ==
inline __host__ __device__ bool operator==(uchar3& a, uchar3& b)
{
  return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);
}

/* ****************************************************************************
 *  uchar4 functions
 * ****************************************************************************/

// create uchar4 from a single uchar
static __inline__ __host__ __device__ uchar4 make_uchar4(unsigned char x)
{
  uchar4 t; t.x = x; t.y = x; t.z = x; t.w = x; return t;
}

// !=
inline __host__ __device__ bool operator!=(uchar4& a, uchar4& b)
{
  return (a.x != b.x) || (a.y != b.y) || (a.z != b.z) || (a.w != b.w);
}

// ==
inline __host__ __device__ bool operator==(uchar4& a, uchar4& b)
{
  return (a.x == b.x) && (a.y == b.y) && (a.z == b.z) && (a.w == b.w);
}

// multiply with constant
inline __host__ __device__ uchar4 operator*(uchar4 a, unsigned char s)
{
    return make_uchar4(a.x * s, a.y * s, a.z * s,  a.w * s);
}
inline __host__ __device__ uchar4 operator*(uchar s, uchar4 a)
{
    return make_uchar4(a.x * s, a.y * s, a.z * s,  a.w * s);
}
// elementwise multiply
inline __host__ __device__ uchar4 operator*(uchar4 a, uchar4 b)
{
    return make_uchar4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}

/* ****************************************************************************
 *  float2 functions
 * ****************************************************************************/


// !=
inline __host__ __device__ bool operator!=(float2& a, float2& b)
{
  return (a.x != b.x) || (a.y != b.y);
}

// ==
inline __host__ __device__ bool operator==(float2& a, float2& b)
{
  return (a.x == b.x) && (a.y == b.y);
}

/* ****************************************************************************
 *  float3 functions
 * ****************************************************************************/

// !=
inline __host__ __device__ bool operator!=(float3& a, float3& b)
{
  return (a.x != b.x) || (a.y != b.y) || (a.z != b.z);
}

// ==
inline __host__ __device__ bool operator==(float3& a, float3& b)
{
  return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);
}

/* ****************************************************************************
 *  float4 functions
 * ****************************************************************************/

// !=
inline __host__ __device__ bool operator!=(float4& a, float4& b)
{
  return (a.x != b.x) || (a.y != b.y) || (a.z != b.z) || (a.w != b.w);
}

// ==
inline __host__ __device__ bool operator==(float4& a, float4& b)
{
  return (a.x == b.x) && (a.y == b.y) && (a.z == b.z) && (a.w == b.w);
}

// elementwise multiply
inline __host__ __device__ float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}

#endif // IUCUTIL_H
