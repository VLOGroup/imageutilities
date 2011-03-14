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

#ifndef IU_CUTIL_H
#define IU_CUTIL_H

#ifdef WIN32
  #undef max
  #undef min
  #define NOMINMAX
#endif

//#ifdef __CUDACC__ // only include this include in cuda files (seen by nvcc)
#include <cutil_math.h>
//#endif

#include "iucore/coredefs.h"

// including some common device functions
#include "common/vectormath_kernels.cuh"
#include "common/derivative_kernels.cuh"
#include "common/bsplinetexture_kernels.cuh"
//#include "common/bind_textures.cuh"

///// SIMPLE MIN MAX HELPERS
template<typename Type1, typename Type2>
inline __host__ __device__ Type2 IUMIN(Type1 a, Type2 b) {return (a<b)?a:b;}

template<typename Type1, typename Type2>
inline __host__ __device__ Type2 IUMAX(Type1 a, Type2 b) {return (a>b)?a:b;}



// includes for time measurements
#ifdef WIN32
  #include <time.h>
  #include <windows.h>
#else
  #include <sys/time.h>
#endif

//// ERROR CHECKS ////////////////////////////////////////////


#include <stdio.h>

/** Print CUDA error. */
namespace iu {
static inline IuStatus checkCudaErrorState(bool print_error = true)
{
  IuStatus status;
  cudaThreadSynchronize();
  if (cudaError_t err = cudaGetLastError())
  {
    fprintf(stderr,"\n\n CUDA Error: %s\n",cudaGetErrorString(err));
    fprintf(stderr,"  file:       %s\n",__FILE__);
    fprintf(stderr,"  function:   %s\n",__FUNCTION__);
    fprintf(stderr,"  line:       %d\n\n",__LINE__);
    status = IU_CUDA_ERROR;
  }
  else
  {
    status = IU_NO_ERROR;
  }
  return status;
}
}

#ifdef __CUDACC__ // only include this error check in cuda files (seen by nvcc)

// MACROS
//

//-----------------------------------------------------------------------------
#define __IU_CHECK_FOR_CUDA_ERRORS_ENABLED__ // enables checking for cuda errors

#ifdef __IU_CHECK_FOR_CUDA_ERRORS_ENABLED__

/** CUDA ERROR HANDLING (CHECK FOR CUDA ERRORS)
 */
#define IU_CHECK_AND_RETURN_CUDA_ERRORS() \
  cudaThreadSynchronize(); \
  if (cudaError_t err = cudaGetLastError()) \
  { \
    fprintf(stderr,"\n\n CUDA Error: %s\n",cudaGetErrorString(err)); \
    fprintf(stderr,"  file:       %s\n",__FILE__); \
    fprintf(stderr,"  function:   %s\n",__FUNCTION__); \
    fprintf(stderr,"  line:       %d\n\n",__LINE__); \
    return IU_CUDA_ERROR; \
  } \
  else \
  { \
    return IU_NO_ERROR; \
  }

/** CUDA ERROR HANDLING (CHECK FOR CUDA ERRORS)
 */
#define IU_CHECK_CUDA_ERRORS() \
  cudaThreadSynchronize(); \
  if (cudaError_t err = cudaGetLastError()) \
  { \
    fprintf(stderr,"\n\n CUDA Error: %s\n",cudaGetErrorString(err)); \
    fprintf(stderr,"  file:       %s\n",__FILE__); \
    fprintf(stderr,"  function:   %s\n",__FUNCTION__); \
    fprintf(stderr,"  line:       %d\n\n",__LINE__); \
  }

//#else // __IU_CHECK_FOR_CUDA_ERRORS_ENABLED__
//#define IU_CHECK_AND_RETURN_CUDA_ERRORS() {}
//#define IU_CHECK_CUDA_ERRORS() {}
//#define IU_PRINT_CUDA_ERRORS() {}
#endif // __IU_CHECK_FOR_CUDA_ERRORS_ENABLED__

#endif // __CUDACC__



//// SMALL CUDA HELPERS (DEFINED static inline) ////////////////////////////////////////////
namespace iu {

//// VARIOUS OTHER HELPER FUNCTIONS ////////////////////////////////////////////
// getTime
static inline double getTime()
{
  cudaThreadSynchronize();
#ifdef WIN32
  LARGE_INTEGER current_time,frequency;
  QueryPerformanceCounter (&current_time);
  QueryPerformanceFrequency(&frequency);
  return current_time.QuadPart*1000.0/frequency.QuadPart;
#else
  timeval time;
  gettimeofday(&time, NULL);
  return time.tv_sec * 1000.0 + time.tv_usec / 1000.0;
#endif
}

} // namespace iu

#endif // IUCUTIL_H
