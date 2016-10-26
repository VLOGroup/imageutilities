#pragma once

#include "ndarray_ref.kernel.h"

//#include <cuda.h>
//#include <device_functions.h>
//#include <cuda_runtime.h>

#ifndef  __CUDA_ARCH__
#include <string>
#include <sstream>
#endif

// ____________________prints____________________________________________
template<class tstream, typename type>
__host__ __device__ void print_array(tstream & ss, const char * s, type * A, int N, int debug_lvl = 1){
	if (debug_lvl <= DEBUG_LVL){
		ss << s << "(";
		for (int i = 0; i < N; ++i){
			ss << A[i];
			if (i == N - 1)break;
			ss << ", ";
		};
		ss << ")\n";
	};
};

template<class tstream, typename type>
__host__  __device__ void print_array(tstream & ss, const char * s, const dslice<type, 1> & A, int N, int debug_lvl = 1){
	if (debug_lvl <= DEBUG_LVL){
		ss << "depricated";
	};
};

template<class tstream, typename type>
__host__  __device__ void print_array(tstream & ss, const char * s, const kernel::ndarray_ref<type, 1> & A, int debug_lvl = 1){
	if (debug_lvl <= DEBUG_LVL){
		ss << s << "\n(";
		for (int i = 0; i < A.size(0); ++i){
			ss << A(i);
			ss << ", ";
		};
		ss << ")\n";
	};
};

template<class tstream, typename type>
__HOSTDEVICE__ void print_array(tstream & ss, const char * s, const kernel::ndarray_ref<type, 2> & A, int debug_lvl = 1){
	if (debug_lvl <= DEBUG_LVL){
		ss << s << "\n";
		for (int i = 0; i < A.size(0); ++i){
			ss << "(";
			for (int j = 0; j < A.size(1); ++j){
				ss << A(i,j);
				ss << ", ";
			};
			ss << ")\n";
		};
	};
};

template<class tstream, typename type>
__HOSTDEVICE__ void print_array(tstream & ss, const char * s, const kernel::ndarray_ref<type, 3> & A, int debug_lvl = 1){
	if (debug_lvl <= DEBUG_LVL){
		ss << s << "\n";
		for (int i = 0; i < A.size(2); ++i){
			print_array(ss, "", A.template subdim<2>(i), debug_lvl);
			//ss << "\n";
		};
	};
}

template<class tstream, typename type>
__HOSTDEVICE__ void print_array(tstream & ss, const char * s, const kernel::ndarray_ref<type, 4> & A, int debug_lvl = 1){
	if (debug_lvl <= DEBUG_LVL){
		ss << s << "\n";
		for (int i = 0; i < A.size(3); ++i){
			ss << "A(:,:,:,"<< i <<")=";
			print_array(ss, "", A.template subdim<3>(i), debug_lvl);
			//ss << "\n";
		};
	};
}


//____________________________________________________________

//_________________________default pf_stream printing_______________________

template<typename type>
__HOSTDEVICE__ void print_array(const char * s, type * A, int N, int debug_lvl = 1){
	pf_stream S1(debug_lvl);
	print_array(S1, s, A, N, debug_lvl);
}

template<typename type, int dims>
__HOSTDEVICE__ void print_array(const char * s, const kernel::ndarray_ref<type, dims> & A, int debug_lvl = 1){
	pf_stream S1(debug_lvl);
	print_array(S1, s, A, debug_lvl);
}

//_________________________________any ostream______________________________

/*
template <int n, class tstream>
__host__ tstream & operator << (tstream & ss, const intn<n> & x){
	ss << "(";
	for(int i=0;i<n;++i){
		ss << x[i] <<",";
	};
	ss << ")";
	return ss;
}
 */

