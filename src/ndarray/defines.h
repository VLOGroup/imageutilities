#pragma once

//#ifndef __CUDACC__
//	#ifndef __host__
//		#define __host__
//	#endif
//	#ifndef __device__
//		#define __device__
//	#endif
////	#ifndef __restrict__
////		#define __restrict__ restrict
////	#endif
//
//#endif
//
//#ifndef __forceinline__
//#ifndef __CUDACC__
//	#if _MSC_VER > 1000
//		#define __forceinline__ __forceinline
//		#define __restrict__ __restrict
//	#else// not msvc
//		#define __forceinline__ inline
//	#endif
//#endif
//#endif

//#if defined _WIN32 || defined __CYGWIN__
//  #ifdef BUILDING_DLL
//    #ifdef __GNUC__
//      #define DLL_PUBLIC __attribute__ ((dllexport))
//    #else
//      #define DLL_PUBLIC __declspec(dllexport) // Note: actually gcc seems to also supports this syntax.
//    #endif
//  #else
//    #ifdef __GNUC__
//      #define DLL_PUBLIC __attribute__ ((dllimport))
//    #else
//      #define DLL_PUBLIC __declspec(dllimport) // Note: actually gcc seems to also supports this syntax.
//    #endif
//  #endif
//  #define DLL_LOCAL
//#else
//  #if __GNUC__ >= 4
//    #define DLL_PUBLIC __attribute__ ((visibility ("default")))
//    #define DLL_LOCAL  __attribute__ ((visibility ("hidden")))
//  #else
//    #define DLL_PUBLIC
//    #define DLL_LOCAL
//  #endif
//#endif

#ifdef __CUDA_ARCH__
	#undef __HOSTDEVICE__
	#undef __HOST__
	#define __HOSTDEVICE__ __host__ __device__ __forceinline__
	#define __HOST__ __host__ __forceinline__
#else
	#undef __HOSTDEVICE__
	#undef __HOST__
	#define __HOSTDEVICE__ inline
	#define __HOST__ inline
#endif

#if (_MSC_VER > 1000) && (_MSC_VER <= 1800) // c++11 support (lack of)
	#pragma push_macro("constexpr")
	#ifndef __cpp_constexpr
	#define constexpr const
	#endif
	#define __attribute__(A) /* do nothing */
	#define inherit_constructors(derived, base) template <typename... Args> derived(Args&&... args) : base(std::forward<Args>(args)...){}
#else
	#define inherit_constructors(derived, base) using base::base;
#endif

struct host_stream{};
struct device_stream{};
