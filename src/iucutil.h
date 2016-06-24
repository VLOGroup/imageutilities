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


/**
 * @brief Exceptions related to cuda issues.
 *
 * Can be used to raise IuException s related to CUDA.
 */
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

// MACROS

#define IU_CUDA_CHECK         iu::checkCudaErrorState(__FILE__, __FUNCTION__, __LINE__)
#define IU_CUDA_SAFE_CALL(fun)       iu::checkCudaErrorState(fun, __FILE__, __FUNCTION__, __LINE__)



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
