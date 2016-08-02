#pragma once

#include "error.h"
#include <cuda_runtime.h>

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

//__________________________cuda_check_error() throws an exception ____________________________

void cuda_check_error_function(const char * __file, int __line);

inline void cuda_check_error_function(const char * __file, int __line){
	cudaError_t r = cudaGetLastError();
	if (r != cudaSuccess){
		//const char * msg = cudaGetErrorString(r);
		throw error_stream().set_file_line(__file, __line) << "cuda ERROR: " << cudaGetErrorString(r) << " " << r << "\n";
	};
};

#define cuda_check_error() cuda_check_error_function(__FILE__,__LINE__);


