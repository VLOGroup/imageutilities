#ifndef ndarray_print_h
#define ndarray_print_h

#include "ndarray_ref.h"

#ifndef DEBUG_LVL
#define DEBUG_LVL 1
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
__host__  __device__ void print_array(tstream & ss, const char * s, const ndarray_ref<type, 1> & A, int debug_lvl = 1){
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
__host__  __device__ void print_array(tstream & ss, const char * s, const ndarray_ref<type, 2> & A, int debug_lvl = 1){
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
__host__  __device__ void print_array(tstream & ss, const char * s, const ndarray_ref<type, 3> & A, int debug_lvl = 1){
	if (debug_lvl <= DEBUG_LVL){
		ss << s << "\n";
		for (int i = 0; i < A.size(2); ++i){
			print_array(ss, "", A.template subdim<2>(i), debug_lvl);
			//ss << "\n";
		};
	};
};

#endif
