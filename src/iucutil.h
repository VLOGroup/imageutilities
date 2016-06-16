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

//#ifdef WIN32
//  #undef max
//  #undef min
//  #define NOMINMAX
//#endif

////#ifdef __CUDACC__ // only include this include in cuda files (seen by nvcc)
//#include <helper_math.h>
////#endif

#include <driver_types.h>
#include "iucore/coredefs.h"

// including some common device functions
//#include "common/vectormath_kernels.cuh"
//#include "common/derivative_kernels.cuh"
//#include "common/bsplinetexture_kernels.cuh"
//#include "common/bind_textures.cuh"

/////// SIMPLE MIN MAX HELPERS
////#ifndef __CUDACC__
//template<typename Type>
//inline __host__ __device__ Type IUMIN(Type a, Type b, bool check_inf_or_nan=true)
//{
////  if (check_inf_or_nan)
////  {
////    if (isnan(a) || isinf(a))
////      return b;
////    if (isnan(b) || isinf(b))
////      return a;
////  }
//  return a<b ? a : b;
//}

//template<typename Type>
//inline __host__ __device__ Type IUMAX(Type a, Type b, bool check_inf_or_nan=true)
//{
////  if (check_inf_or_nan)
////  {
////    if (isnan(a) || isinf(a))
////      return b;
////    if (isnan(b) || isinf(b))
////      return a;
////  }
//  return a>b ? a : b;
//}

//#else
//template<typename Type>
//inline __host__ __device__ Type IUMIN(Type a, Type b) {return min(a,b);}

//template<typename Type>
//inline __host__ __device__ Type IUMAX(Type a, Type b) {return max(a,b);}
//#endif

namespace iu {
template<typename Type>
inline __host__ __device__ Type sqr(Type a) {return a*a;}
}

// includes for time measurements
#ifdef WIN32
  #include <time.h>
  #include <windows.h>
#else
  #include <sys/time.h>
#endif

//// ERROR CHECKS ////////////////////////////////////////////

#include <stdio.h>

//#define IU_CUDACALL(err) iu::cudaCall(err, __FILE__, __LINE__ );

///** Compile time conditional check for CUDA error (throws IuCudaException).
//    Define IU_CUDA_CHECK_ENABLED to enable it.
// */
//#ifdef IU_CUDA_CHECK_ENABLED
//  #define IU_CUDA_CHECK() checkCudaErrorState(__FILE__, __FUNCTION__, __LINE__)
//#else
//  #define IU_CUDA_CHECK() do{}while(0)
//#endif

class IuCudaException : public IuException
{
public:
  IuCudaException(const cudaError_t cudaErr,
                  const char* file=NULL, const char* function=NULL, int line=0) throw() :
    IuException(std::string("CUDA Error: ") + cudaGetErrorString(cudaErr), file, function, line),
    cudaErr_( cudaErr )
  {
  }

protected:
  cudaError_t cudaErr_;
};

//class IuCudaBadAllocException : public IuCudaException
//{
//public:
//  IuCudaBadAllocException(const cudaError_t cudaErr,
//                  const char* file=NULL, const char* function=NULL, int line=0) throw() :
//    IuCudaException(cudaErr, file, function, line) {}
//};

namespace iu {


/** Check for CUDA error (throws IuCudaException) */
static inline void checkCudaErrorState( const char* file, const char* function, const int line )
{
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if( err != cudaSuccess )
    throw IuCudaException( err, file, function, line );
}

/** Check for CUDA error (throws IuCudaException) */
static inline void checkCudaErrorState(cudaError_t err, const char *file, const char* function,
                         const int line)
{
  if (cudaSuccess != err)
  {
    throw IuCudaException(err, file, function, line);
  }
}


static inline float getTotalGPUMemory()
{
  size_t total = 0;
  size_t free = 0;
  cudaMemGetInfo(&free, &total);
  return total/(1024.0f*1024.0f);   // return value in Megabytes
}

static inline float getFreeGPUMemory()
{
  size_t total = 0;
  size_t free = 0;
  cudaMemGetInfo(&free, &total);
  return free/(1024.0f*1024.0f);   // return value in Megabytes
}

static inline void printGPUMemoryUsage()
{
  float total = iu::getTotalGPUMemory();
  float free = iu::getFreeGPUMemory();

  printf("GPU memory usage\n");
  printf("----------------\n");
  printf("   Total memory: %.2f MiB\n", total);
  printf("   Used memory:  %.2f MiB\n", total-free);
  printf("   Free memory:  %.2f MiB\n", free);
}




} //namespace iu

//#ifdef __CUDACC__ // only include this error check in cuda files (seen by nvcc)

// MACROS

#define IU_CUDA_CHECK         iu::checkCudaErrorState(__FILE__, __FUNCTION__, __LINE__)
#define IU_CUDA_SAFE_CALL(fun)       iu::checkCudaErrorState(fun, __FILE__, __FUNCTION__, __LINE__)


////-----------------------------------------------------------------------------
//#define __IU_CHECK_FOR_CUDA_ERRORS_ENABLED__ // enables checking for cuda errors


/** CUDA ERROR HANDLING (CHECK FOR CUDA ERRORS)
 */
/*
#ifdef __IU_CHECK_FOR_CUDA_ERRORS_ENABLED__
#define IU_CHECK_AND_RETURN_CUDA_ERRORS() \
{ \
  cudaDeviceSynchronize(); \
  if (cudaError_t err = cudaGetLastError()) \
  { \
    fprintf(stderr,"\n\nCUDA Error: %s\n",cudaGetErrorString(err)); \
    fprintf(stderr,"  file:       %s\n",__FILE__); \
    fprintf(stderr,"  function:   %s\n",__FUNCTION__); \
    fprintf(stderr,"  line:       %d\n\n",__LINE__); \
    return IU_ERROR; \
  } \
  else \
    return IU_NO_ERROR; \
}
*/

/*
#define IU_CHECK_CUDA_ERRORS() \
{ \
  cudaDeviceSynchronize(); \
  if (cudaError_t err = cudaGetLastError()) \
  { \
    fprintf(stderr,"\n\nCUDA Error: %s\n",cudaGetErrorString(err)); \
    fprintf(stderr,"  file:       %s\n",__FILE__); \
    fprintf(stderr,"  function:   %s\n",__FUNCTION__); \
    fprintf(stderr,"  line:       %d\n\n",__LINE__); \
  } \
}
*/

//#else // __IU_CHECK_FOR_CUDA_ERRORS_ENABLED__
//#define IU_CHECK_AND_RETURN_CUDA_ERRORS() {}
//#define IU_CHECK_CUDA_ERRORS() {}
//#endif // __IU_CHECK_FOR_CUDA_ERRORS_ENABLED__


//#endif // __CUDACC__



//// SMALL CUDA HELPERS (DEFINED static inline) ////////////////////////////////////////////
namespace iu {

//// VARIOUS OTHER HELPER FUNCTIONS ////////////////////////////////////////////
// getTime
static inline double getTime()
{
  cudaDeviceSynchronize();
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
