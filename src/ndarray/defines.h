#pragma once

#ifndef __CUDACC__
	#ifndef __host__
		#define __host__
	#endif
	#ifndef __device__
		#define __device__
	#endif
//	#ifndef __restrict__
//		#define __restrict__ restrict
//	#endif

#endif

#ifndef __forceinline__
#ifndef __CUDACC__
	#if _MSC_VER > 1000
		#define __forceinline__ __forceinline
		#define __restrict__ __restrict
	#else// not msvc
		#define __forceinline__ inline
	#endif
#endif
#endif

#if defined _WIN32 || defined __CYGWIN__
  #ifdef BUILDING_DLL
    #ifdef __GNUC__
      #define DLL_PUBLIC __attribute__ ((dllexport))
    #else
      #define DLL_PUBLIC __declspec(dllexport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #else
    #ifdef __GNUC__
      #define DLL_PUBLIC __attribute__ ((dllimport))
    #else
      #define DLL_PUBLIC __declspec(dllimport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #endif
  #define DLL_LOCAL
#else
  #if __GNUC__ >= 4
    #define DLL_PUBLIC __attribute__ ((visibility ("default")))
    #define DLL_LOCAL  __attribute__ ((visibility ("hidden")))
  #else
    #define DLL_PUBLIC
    #define DLL_LOCAL
  #endif
#endif

struct host_stream{};
struct device_stream{};
