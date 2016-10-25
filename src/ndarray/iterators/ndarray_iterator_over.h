#pragma once

#include "ndarray_ref.host.h"
#include <thrust/iterator/iterator_facade.h>
//#include "bit_index.h"


#ifdef _MSC_VER
	#include <intrin.h>
	#define clz(x) __lzcnt(x)
#elif __GNUC__
	#define clz(x) __builtin_clz(x)
#endif

inline int fls(int x){
	return x ? sizeof(x) * 8 - clz(x) : 0;
}

#undef DEBUG_THIS
#define DEBUG_THIS false
//#define DEBUG_THIS true

template<typename type, int dims, typename System>
struct ndarray_iterator_over;

namespace detail{
	template<typename type, int dims, typename System>
	struct ndarray_iterator_over_base{
		typedef type & reference;
		typedef type value_type;
		typedef int difference_type;
		typedef typename thrust::random_access_traversal_tag traversal_category;
	public:
		// The iterator facade type from which iterator will be derived.
		typedef thrust::iterator_facade<
				ndarray_iterator_over<type, dims, System>,
				value_type,
				System,
				traversal_category,
				reference,
				difference_type
				> parent;
	}; // end ndarray_iterator_base
};

template<typename type, int dims, typename System>
struct ndarray_iterator_over : public detail::ndarray_iterator_over_base<type, dims, System>::parent{
private:
	typedef typename detail::ndarray_iterator_over_base<type, dims, System>::parent parent;
public:
	using typename parent::reference;
	using typename parent::difference_type;
	using typename parent::value_type;
	const kernel::ndarray_ref<type, dims> src;
public:
	int dims1; // compressed size, dims1 <= dims
	tstride<char> offset;
	int ii;
	intn<dims> mask;
	intn<dims> asize;
	intn<dims> bb;
	bool _in_range; // temporary
	const float mask_val;
public:
	__forceinline__ __host__ __device__
	ndarray_iterator_over() = default;
	__host__
	ndarray_iterator_over(const ndarray_ref<type, dims> & x, float _mask_val):src(x.kernel()), dims1(src.nz_dims()),mask_val(_mask_val){
		//p = x.ptr();
		ii = 0;
		offset = 0;
		asize = 0;
		mask = 0;
		_in_range = true;
		int bits = 0;
		intn<dims> size = x.size();
		//std::cout << "dims1=" << dims1 << "\n";
		for(int d=0; d<dims; ++d){
			bb[d] = bits;
			if(d == dims1){ //end of useful sizes
				for(int d1 = d; d1< dims;++d1){
					//std::cout << "d1" << d1<<"\n";
					asize[d1] = asize[d-1];
					mask[d1] = (1 << (31-bits) -1) << bits;
					bb[d1] = bb[d-1];
				};
				break;
				//size[d] = 1;
			};
			int b = fls(size[d] - 1);
			mask[d] = ((1 << b) -1) << bits;
			asize[d] = size[d] << bits;
			bits += b;
		};
		//std::cout << "asize=" << asize << " bits:"<<bits << "\n";
		//std::cout << "mask=" << mask << " bb:"<<bb << "\n";
		//std::cout << "_____\n";
	}

	__forceinline__ __host__ __device__
		ndarray_iterator_over(const ndarray_iterator_over<type, dims, System> & x) = default;

	private:
	friend class thrust::iterator_core_access;
	__host__ __device__
	//! dereference
	typename parent::reference dereference() const{
//#ifdef  __CUDA_ARCH__
//		if(threadIdx.x<4 && blockIdx.x == 0)printf("(%i, %i) : %i\n", (int)(threadIdx.x),(int)(blockIdx.x),ii);
//#endif
		if(DEBUG_THIS)runtime_check(ii>=0 && ii < asize[dims-1]) << "ii=" << ii;
		if(!in_range()){
			return const_cast<float&>(mask_val);
			//return * src.ptr();
		};
		type * __restrict__ p = src.ptr() + offset;
		return *p;
	}
	__forceinline__ __host__ __device__ bool in_range()const{
		return _in_range;
	}

	//! equal
	__forceinline__ __host__ __device__
	bool equal(const ndarray_iterator_over<type, dims, System> &other) const{
		//if(!in_range()){
		//return !other.in_range();
		//};
		return ii == other.ii;
	};

	// Advancing
	__forceinline__ __host__ __device__
	void advance(typename parent::difference_type n){
//#ifdef  __CUDA_ARCH__
//		if(threadIdx.x==1 && blockIdx.x == 1)
//				printf("%i : adv %i + %i\n",int(threadIdx.x), ii, n);
//#endif

#ifdef  __CUDA_ARCH__
		if(threadIdx.x==1)if(DEBUG_THIS)
				printf("%i : adv %i + %i\n",int(threadIdx.x), ii, n);
#else
		//if(ii%1000000==0) std::cout<< "advance "<<  ii << "+" << n;
#endif
		ii += n;
		_in_range = true;
		offset = 0;
		for(int d = 0; d < dims; ++d){
			if(d < dims1){
				int i = ii & mask[d];
				if(i < asize[d]){
					i = i >> bb[d];
					offset += i*src.stride(d);
				}else{// out of bounds
					_in_range = false;
					break;
				};
			};
//			};
		};
#ifdef  __CUDA_ARCH__
		if(threadIdx.x==1)if(DEBUG_THIS)
				printf("%i :   -> %i\n", int(threadIdx.x), ii);
#else
		//if(ii%1000000==1) std::cout<< "->" << ii <<"\n";
#endif
	}

	// Incrementing
	__forceinline__ __host__ __device__ void increment(){
		advance(1);
	}

	// Decrementing
	__forceinline__ __host__ __device__
	void decrement(){
		advance(-1);
	}

	// Distance
	__forceinline__ __host__ __device__
	typename parent::difference_type
	distance_to(const ndarray_iterator_over<type, dims, System> &other) const{
		int dist = (other.ii - ii);
#ifndef  __CUDA_ARCH__
		//std::cout << "distance_to("<< ii << "," << other.ii << ")=" << dist <<"\n";
#else
		if(threadIdx.x==1)if(DEBUG_THIS)
			printf("%i :   distance(%i,%i) = %i\n", int(threadIdx.x), ii, other.ii, ii);
#endif

		return dist;
	}
	/*
	public:
	__host__ __device__ __forceinline__ reference operator[](difference_type n) const{
		return *((*this) + n);
	}
	*/
};

template<typename type, int dims>
template<typename System>
ndarray_iterator_over<type, dims, System> ndarray_ref<type,dims>::begin_it1() const{
	return ndarray_iterator_over<type, dims, System>(compress_dims(),0);
}

template<typename type, int dims>
template<typename System>
ndarray_iterator_over<type, dims, System> ndarray_ref<type,dims>::end_it1() const{
	ndarray_iterator_over<type, dims, System> a(compress_dims(),0);
	a += a.asize[dims-1];
	return a;
}
