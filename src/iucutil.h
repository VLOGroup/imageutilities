#pragma once

#include <driver_types.h>
#include "iucore/coredefs.h"
#include <cufft.h>

// includes for time measurements
#ifdef _WIN32
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
 * @ingroup UTILS
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
/** \defgroup UTILS Utilities
 * \brief various bits and pieces, error handling, conveniencs functions
 * \{
 */

/** Check for CUDA error (throws IuCudaException)
 * @ingroup UTILS
 */
static inline void checkCudaErrorState( const char* file, const char* function, const int line )
{
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if( err != cudaSuccess )
    throw IuCudaException( err, file, function, line );
}

/** Check for CUDA error (throws IuCudaException)
* @ingroup UTILS
*/
static inline void checkCudaErrorState(cudaError_t err, const char *file, const char* function,
                         const int line)
{
  if (cudaSuccess != err)
  {
    throw IuCudaException(err, file, function, line);
  }
}

/**
 * @brief Exceptions related to cufft issues.
 *
 * @ingroup UTILS
 * Can be used to raise IuException s related to CUFFT.
 */
class IuCufftException: public IuException
{
public:
  IuCufftException(const cufftResult cudaErr, const char* file = NULL,
                   const char* function = NULL, int line = 0) throw () :
      IuException(std::string("CUFFT Error: ") + cufftGetErrorString(cudaErr),
                  file, function, line),
      cufftResult_(cudaErr)
  {
  }

protected:
  cufftResult cufftResult_;

private:
  static const char *cufftGetErrorString(cufftResult err)
  {
    switch (err)
    {
      case CUFFT_SUCCESS:
        return "The cuFFT operation was successful.";
      case CUFFT_INVALID_PLAN:
        return "cuFFT was passed an invalid plan handle.";
      case CUFFT_ALLOC_FAILED:
        return "cuFFT failed to allocate GPU or CPU memory.";
      case CUFFT_INVALID_VALUE:
        return "User specified an invalid pointer or parameter.";
      case CUFFT_INTERNAL_ERROR:
        return "Driver or internal cuFFT library error";
      case CUFFT_EXEC_FAILED:
        return "Failed to execute an FFT on the GPU";
      case CUFFT_SETUP_FAILED:
        return "The cuFFT library failed to initialize";
      case CUFFT_INVALID_SIZE:
        return "User specified an invalid transform size";
      case CUFFT_INCOMPLETE_PARAMETER_LIST:
        return "Missing parameters in call";
      case CUFFT_INVALID_DEVICE:
        return "Execution of a plan was on different GPU than plan creation";
      case CUFFT_PARSE_ERROR:
        return "Internal plan database error";
      case CUFFT_NO_WORKSPACE:
        return "No workspace has been provided prior to plan execution";
      default:
        return "Unknown CUFFT error.";
    }
  }
};

/** Check for CUFFT error (throws IuCudaException)
* @ingroup UTILS
*/
static inline void checkCufftErrorState(const cufftResult status,
                                        const char* file, const char* function,
                                        const int line)
{
  if (status != CUFFT_SUCCESS)
    throw IuCufftException(status, file, function, line);
}

/** Get total GPU memory.
* @ingroup UTILS
*/
static inline float getTotalGPUMemory()
{
  size_t total = 0;
  size_t free = 0;
  cudaMemGetInfo(&free, &total);
  return total/(1024.0f*1024.0f);   // return value in Megabytes
}

/** Get free GPU memory.
* @ingroup UTILS
*/
static inline float getFreeGPUMemory()
{
  size_t total = 0;
  size_t free = 0;
  cudaMemGetInfo(&free, &total);
  return free/(1024.0f*1024.0f);   // return value in Megabytes
}

/** Print GPU memory usage information.
* @ingroup UTILS
*/
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



/** Get current system time with high resolution.
* @ingroup UTILS
*/
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
/** \} */
} //namespace iu

// MACROS

#define IU_CUDA_CHECK         iu::checkCudaErrorState(__FILE__, __FUNCTION__, __LINE__)
#define IU_CUDA_SAFE_CALL(fun)       iu::checkCudaErrorState(fun, __FILE__, __FUNCTION__, __LINE__)
#define IU_CUFFT_SAFE_CALL(state) iu::checkCufftErrorState(state, __FILE__, __FUNCTION__, __LINE__ )
