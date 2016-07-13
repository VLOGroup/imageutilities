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

