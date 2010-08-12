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

#ifndef IU_CUDADEFS_H
#define IU_CUDADEFS_H

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

#endif // __CUDACC__

#endif // IU_CUDADEFS_H
