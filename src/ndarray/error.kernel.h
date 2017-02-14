#pragma once

#include "defines.h"
#include <assert.h>

#ifndef DEBUG_LVL
#define DEBUG_LVL 0
#endif


//____________________stream_eater__________________________________

struct stream_eater{
	__host__ __device__ __forceinline__ stream_eater & operator << (int a){return *this;}
	__host__ __device__ __forceinline__ stream_eater & operator << (long long a){return *this;}
	__host__ __device__ __forceinline__ stream_eater & operator << (size_t a){return *this;}
	__host__ __device__ __forceinline__ stream_eater & operator << (float a){return *this;}
	__host__ __device__ __forceinline__ stream_eater & operator << (double a){return *this;}
	__host__ __device__ __forceinline__ stream_eater & operator << (void * a){return *this;}
	__host__ __device__ __forceinline__ stream_eater & operator << (const char * a){return *this;}
	__host__ __device__ __forceinline__ stream_eater & operator << (const std::string & a){return *this;}
};


//____________________stream to printf______________________________
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

	__host__ __device__ pf_stream & operator << (const std::string & s) {
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

	__host__ __device__ pf_stream & operator << (unsigned int x) {
		if (DEBUG_LVL >= lvl){
			printf("%i", x);
		};
		return *this;
	}

	__host__ __device__ pf_stream & operator << (short int x) {
		if (DEBUG_LVL >= lvl){
			printf("%i", int(x));
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
			printf("%5.3f", x);
		};
		return *this;
	}

	__host__ __device__  pf_stream & operator << (double x)  {
		if (DEBUG_LVL >= lvl){
			printf("%5.3f", x);
		};
		return *this;
	}

	/*
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
	*/
};

//_________________in-kernel runtime check stream_______________________
struct pf_error_stream : public pf_stream{
public:
	__host__ __device__ pf_error_stream(int _lvl):pf_stream(_lvl){};
	__host__ __device__ ~pf_error_stream(){
#ifdef  __CUDA_ARCH__
		//(*this) << "Thread=(" << threadIdx.x << "," << threadIdx.y << "," << threadIdx.z <<") ";
		//(*this) << "Block=(" << blockIdx.x << "," << blockIdx.y << "," << blockIdx.z <<")";
		//asm("trap;");
#else
		//__asm__ volatile("int3");
#endif
	}
};

//#ifdef  __CUDA_ARCH__
//	#define runtime_check_k(expression) if(!(expression)) pf_error_stream(0) << __FILE__ << __LINE__ << "Runtime check failed " << #expression << "\n"
//#endif
