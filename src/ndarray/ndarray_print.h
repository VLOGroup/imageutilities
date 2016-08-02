#pragma once

#include "ndarray_ref.kernel.h"

//#include <cuda.h>
//#include <device_functions.h>
//#include <cuda_runtime.h>

#include <string>
#include <sstream>


#ifndef DEBUG_LVL
#define DEBUG_LVL 0
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
}

template<class tstream, typename type>
__host__  __device__ void print_array(tstream & ss, const char * s, const ndarray_ref<type, 4> & A, int debug_lvl = 1){
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

struct pf_stream{
	int lvl;
	__host__ __device__ pf_stream(int _lvl = 0) :lvl(_lvl){
		//printf("lvl= %i: ", lvl);
	}
	__host__ __device__ pf_stream & operator << (const char * s) {
		if (DEBUG_LVL >= lvl){
            printf("%s", s);
		};
		return *this;
	}

	__host__ __device__ pf_stream & operator << (void * p) {
		if (DEBUG_LVL >= lvl){
            printf("%p", p);
		};
		return *this;
	}

	__host__ pf_stream & operator << (const std::string & s) {
		if (DEBUG_LVL >= lvl){
			printf("%s", s.c_str());
		};
		return *this;
	}

	__host__ __device__ pf_stream & operator << (const char & s) {
		if (DEBUG_LVL >= lvl){
			printf("%c", s);
		};
		return *this;
	}

	__host__ __device__ pf_stream & operator << (int x) {
		if (DEBUG_LVL >= lvl){
			printf("%i", x);
		};
		return *this;
	}

	template <int n, typename type> __host__ __device__ pf_stream & operator << (const type (&x)[n]) {
		if (DEBUG_LVL >= lvl){
			(*this) << "[";
			for(int i=0; i < n; ++i){
			  (*this) << x[i] << " ";
			};
			(*this) << "]";
		};
		return *this;
	}

	__host__ __device__  pf_stream & operator << (float x)  {
		if (DEBUG_LVL >= lvl){
			printf("%5.2f", x);
		};
		return *this;
	}

	__host__ __device__  pf_stream & operator << (double x)  {
		if (DEBUG_LVL >= lvl){
			printf("%5.2f", x);
		};
		return *this;
	}

	template <int n>
	__host__ __device__ pf_stream & operator << (const intn<n> & x) {
		if (DEBUG_LVL >= lvl){
			printf("(");
			for(int i=0;i<n;++i){
				printf("%i,", x[i]);
			};
			printf(")");
		};
		return *this;
	}
};

//_________________________default pf_stream printing_______________________

template<typename type>
__host__ __device__ void print_array(const char * s, type * A, int N, int debug_lvl = 1){
	pf_stream S1(debug_lvl);
	print_array(S1, s, A, N, debug_lvl);
};

template<typename type, int dims>
__host__  __device__ void print_array(const char * s, const ndarray_ref<type, dims> & A, int debug_lvl = 1){
	pf_stream S1(debug_lvl);
	print_array(S1, s, A, debug_lvl);
};

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

