#pragma once

#include "ndarray_ref.host.h"
#include <thrust/iterator/iterator_facade.h>

#define DEBUG_THIS false
//#define DEBUG_THIS true

template<typename type, int dims, typename System>
struct ndarray_iterator;

namespace detail{
	template<typename type, int dims, typename System>
	struct ndarray_iterator_base{
		typedef type & reference;
		typedef type value_type;
		typedef int difference_type;
		typedef typename thrust::random_access_traversal_tag traversal_category;
	public:
		// The iterator facade type from which iterator will be derived.
		typedef thrust::iterator_facade<
				ndarray_iterator<type, dims, System>,
				value_type,
				System,
				traversal_category,
				reference,
				difference_type
				> parent;
	}; // end ndarray_iterator_base
};

template<typename type, int dims, typename System>
struct ndarray_iterator : public detail::ndarray_iterator_base<type, dims, System>::parent{
private:
	typedef typename detail::ndarray_iterator_base<type, dims, System>::parent parent;
public:
	// this is not changed by the iterator and hopefully can get optimized across copies in threads
	//const kernel::ndarray_ref<type, dims> src;
	const ndarray_ref<type, dims> src;
public:
	intn<dims> ii;
	int dims1; // compressed size, dims1 <= dims
	//type __restrict__ * p;
	tstride<char> offset;
	struct compressed{
		intn<dims> mask;
		intn<dims> asize;
		int ii;
		int offset;
		compressed(intn<dims> size){
			asize = 0;
			mask = 0;
			ii = 0;
			offset = 0;
			int bits = 0;
			for(int d=0; d<dims; ++d){
				if(size[d]>0){
					int b = fls(size[d] - 1);
					mask[d] = ((1 << b) -1) << bits;
					asize[d] = size[d] << bits;
					bits += b;
				};
			};
		};
	};
public:
	__forceinline__ __host__ __device__
	ndarray_iterator() = default;
	__host__
	ndarray_iterator(const ndarray_ref<type, dims> & x):src(x), dims1(src.nz_dims()){
		//p = x.ptr();
		ii = 0;
		offset = 0;
#ifdef  __CUDA_ARCH__
		if(threadIdx.x<2)
		if(DEBUG_THIS)printf("%i: %i (%i,%i,%i) dims1=%i\n",int(threadIdx.x),offset.value(),ii[0],ii[1],ii[2],dims1);
#else
//		std::cout << "compressed" << src << "\n";
//		if(DEBUG_THIS)printf("%i (%i,%i,%i) dims1=%i \n",offset.value(),ii[0],ii[1],ii[2],dims1);
#endif
	}

	__forceinline__ __host__ __device__
		ndarray_iterator(const ndarray_iterator<type, dims, System> & x) = default;

	/*
	__host__
	ndarray_iterator(const ndarray_ref<type, dims> & x){
		ndarray_ref<type, dims> src = x;
		src.size() = 0;
		src.stride_bytes() = 0;
		// compress source
		dims1 = 0;
		int s = x.size(0);
		int st = x.stride_bytes(0);
		for(int d = 1; d< dims; ++d){
			if(x.size(d)==1 || x.stride_bytes(d) == st*s){// no padding from d-1 to d -- can compress
				s *= x.size(d); // multiply out and treat linearly
				continue;
			}else{// index is discontinuous - cannot compress
				src.size()[dims1] = s;
				src.stride_bytes()[dims1] = st;
				st = x.stride_bytes(d);
				s = x.size(d);
				++dims1;
			};
		};
		src.size()[dims1] = s;
		src.stride_bytes()[dims1] = st;
		++dims1; // compressed dimensions
		p = x.ptr();
		ii = 0;
		std::cout << "compressed to dims1=" <<dims1 <<": " << src << "\n";
		const_cast<kernel::ndarray_ref<type, dims>&>(this->src) = src;
	}
	*/

	/*
	__forceinline__ __host__ __device__
	ndarray_iterator(const ndarray_iterator<type, dims, System> & x):parent(x){
		ii = x.ii;
		dims1 = x.dims1;
		p = x.p;
		src = x.src;
#ifdef  __CUDA_ARCH__
		if(threadIdx.x==0){
			__shared__ int dev_cpy_count_xx;
			__shared__ int dev_cpy_count;
			if(dev_cpy_count_xx != dev_cpy_count){
				dev_cpy_count = 0;
				dev_cpy_count_xx = 0;
			};
			++dev_cpy_count;
			++dev_cpy_count_xx;
			if(DEBUG_THIS)printf("device copy: BlockSize x %i \n", dev_cpy_count);
			if(DEBUG_THIS)printf("BlockSize: (%i,%i,%i)\n", blockDim.x, blockDim.y, blockDim.z);
			//if(DEBUG_THIS)printf(" GridSize: (%i,%i,%i)\n", gridDim.x, gridDim.y, gridDim.z);
		};
#else
		if(DEBUG_THIS)printf("====host copy ==========\n");
#endif
	};
	*/
private:
	friend class thrust::iterator_core_access;
	__host__ __device__
	//! dereference
	typename parent::reference dereference() const{
#ifdef  __CUDA_ARCH__
		if(threadIdx.x<5)
		if(DEBUG_THIS)printf("%i: dereferencing %i (%i,%i,%i)\n",int(threadIdx.x), offset.value(), ii[0],ii[1],ii[2]);
#endif
		type *p = src.ptr() + offset;
#ifdef  __CUDA_ARCH__
		if(DEBUG_THIS){
			int numel = 1;
			for(int d=0; d<dims; ++d){
				if(d<dims1) numel *= src.size(d);
			};
			if(p-src.ptr()< 0 || p-src.ptr() >= numel){
				printf("%i: bad dereference %i (%i,%i,%i)\n",int(threadIdx.x), offset.value(), ii[0],ii[1],ii[2]);
				return *src.ptr();
			};
		}
#endif
		return *p;
	}
	//! equal
	__forceinline__ __host__ __device__
	bool equal(const ndarray_iterator<type, dims, System> &other) const{
		//return p == other.p;
		return offset == other.offset;
	};

	// Incrementing
	__forceinline__ __host__ __device__
	void increment(){
		advance(1);
		/*
#ifdef  __CUDA_ARCH__
		if(threadIdx.x<5)
		if(DEBUG_THIS)printf("%i: inc %i (%i,%i,%i)\n",int(threadIdx.x), offset.value(), ii[0],ii[1],ii[2]);
#endif
		for (int d = 0; d < dims; ++d){
			if(d < dims1){
				ii[d]++; // add to current index
				if (ii[d] < src.size(d)){ //no overflow
					offset += src.stride(d); // advance pointer
					break;
				}else if (d == dims1-1){ //nowhere to carry over
					offset = 0x7FFFFFFF; // advance pointer to max integer -- end marker
					break;
				}else{ //carry
					ii[d] = 0;
					offset -= src.stride(d)*(src.size(d)-1); // adjust pointer
				};
			};
		};
		*/
#ifdef  __CUDA_ARCH__
		if(threadIdx.x<5)
		if(DEBUG_THIS)printf("%i: ->ind %i (%i,%i,%i)\n",int(threadIdx.x), offset.value(), ii[0],ii[1],ii[2]);
#endif
	}

	// Advancing
	__forceinline__ __host__ __device__
	void advance(typename parent::difference_type n){
#ifdef  __CUDA_ARCH__
		if(threadIdx.x<5)
		if(DEBUG_THIS)printf("%i: adv %i (%i,%i,%i) + %i\n",int(threadIdx.x), offset.value(), ii[0],ii[1],ii[2], n);
#endif
		if(n<0){
			if(offset == 0x7FFFFFFF){ // revert to the _last_ element
				offset = 0;
				for (int d = 0; d < dims; ++d){
					if (d < dims1){
						ii[d] = src.size(d)-1;
						offset += src.stride(d) * ii[d];
					};
				};
			}else{ // in range
				for (int d = 0; d < dims; ++d){
					if(d < dims1 && n != 0){
						int i_old = ii[d]; // save
						ii[d] += n; // add to current index
						if (ii[d] >= 0){ //no underflow
							offset += (n * src.stride(d)); // advance pointer
							n = 0;
						}else if (d == dims1-1){ //underflow but nowhere to carry over
							offset = 0x7FFFFFFF; // end
							n = 0;
						}else{ //underflow carry
							// try a simple step -- suffices for BlokcSize < size(d)
							ii[d] += src.size(d);
							if(ii[d] >= 0){//simple step suffices
								n = -1; // carry
							}else{ // simple step was not sufficient
								int a = ii[d] / src.size(d) - 1; // negative quotient
								int b = ii[d] % src.size(d) + src.size(d); // positive reminder
								ii[d] = b; // reminder
								n = a - 1; // quotient plus simple step
							};
							offset += (ii[d] - i_old) * src.stride(d); // adjust pointer to new index in d
						};
					};
				};
			};
		}else{// n>=0
			for (int d = 0; d < dims; ++d){
				if(d < dims1 && n != 0){
					int i_new = ii[d] + n; // add to current index
					if (i_new < src.size(d)){ //no overflow
						ii[d] = i_new;
						offset += (n * src.stride(d)); // advance pointer
						n = 0;
					}else if (d == dims1-1){ //overflow but nowhere to carry over
						offset = 0x7FFFFFFF; // advance pointer to max integer -- end marker
						ii[d] = i_new;
						n = 0;
					}else{ //overflow with carry
						// try a simple step -- suffices for BlokcSize < size(d)
						i_new -= src.size(d);
						if(i_new < src.size(d)){//simple step suffices
							n = 1; // carry 1
						}else{ // simple step was not sufficient
							int a = i_new / src.size(d); // quotient
							i_new = i_new % src.size(d); // reminder
							n = a + 1; // quotient plus simple step
						};
						offset += (i_new - ii[d]) * src.stride(d); // adjust pointer to new index in d
						ii[d] = i_new;
					};
				};
			};
			/*
			for (int d = 0; d < dims; ++d){
				int i = compressed.ii & compressed.mask[d];
				int i_new = i + n; // add to current index
				if (i_new < compressed.asize[d] || d == dims1-1){ //no overflow or nowhere to carry over
					compressed.ii = compressed.ii - i + i_new;
					compressed.offset += (n * src.stride(d)); // advance pointer
				}else{//overflow with carry
					int a = i_new / compressed.asize[d]; // quotient
					i_new = i_new % compressed.asize[d]; // reminder
					n = a + 1; // quotient plus simple step
					offset += (i_new - ii[d]) * src.stride(d); // adjust pointer to new index in d
					ii[d] = i_new;
				};
			};
			*/
		};
		//p = src.ptr(ii);
#ifdef  __CUDA_ARCH__
		if(threadIdx.x<5)
		if(DEBUG_THIS)printf("%i: ->adv %i (%i,%i,%i)\n",int(threadIdx.x), offset.value(), ii[0],ii[1],ii[2]);
#endif
	}


	// Decrementing
	__forceinline__ __host__ __device__
	void decrement(){
		advance(-1);
/*
#ifdef  __CUDA_ARCH__
		if(DEBUG_THIS)printf("====device  decrement ==========\n");
#else
		if(DEBUG_THIS)printf("====host  decrement ==========\n");
#endif
 */
		/*
#ifdef  __CUDA_ARCH__
		if(threadIdx.x<5)
		if(DEBUG_THIS)printf("%i: dec %i (%i,%i,%i)\n",int(threadIdx.x), offset.value(), ii[0],ii[1],ii[2]);
#endif

		if(offset == 0x7FFFFFFF){ // revert to the _last_ element
			offset = 0;
			for (int d = 0; d < dims; ++d){
				ii[d] = src.size(d)-1;
				offset += src.stride(d) * ii[d];
				if (d == dims1-1)break;
			};
		}else{
			for (int d = 0; d < dims; ++d){
				ii[d]--; // add to current index
				if (ii[d] >= 0){ //no underflow
					offset -= src.stride(d); // advance pointer
					break;
				};
				if (d == dims1-1){ //nowhere to carry over
					offset = 0x7FFFFFFF; // advance pointer to max integer -- end marker
					break;
				};
				//carry
				ii[d] = src.size(d)-1;
				offset += src.stride(d)*(src.size(d)-1); // adjust pointer
			};
		};
*/
		/*
		// decrement iterator
		for (int d = 0; d < dims; ++d){
			if(d >= dims1) break; //
			if (ii[d] == 0 && d < dims1-1 ){
				ii[d] = src.size(d)-1;
				//p+= src.stride(d)*src.size(d);
				offset += src.stride(d)*src.size(d);
				continue;
			} else{
				--ii[d];
				//p-= src.stride(d);
				offset -= src.stride(d);
				break;
			};
		};
		*/
		/*
#ifdef  __CUDA_ARCH__
		if(threadIdx.x<5)
			if(DEBUG_THIS)printf("%i: ->dec %i (%i,%i,%i)\n",int(threadIdx.x), offset.value(), ii[0],ii[1],ii[2]);
#endif
*/
	}

	// Distance
	__forceinline__ __host__ __device__
	typename parent::difference_type
	distance_to(const ndarray_iterator<type, dims, System> &other) const{
		int i1 = other.src.size().index_to_integer(other.ii, other.dims1);
		int i2 = src.size().index_to_integer(ii, dims1);
#ifndef  __CUDA_ARCH__
		//std::cout << other.ii << "->" << i1 << "\n(" << other.src.size() << ":" << other.src.stride_bytes() << ")\n";
		//std::cout << ii << "->" << i2 <<"\n(" << src.size() << ":" << src.stride_bytes() << ")\n";
#endif
		int dist = (i1 - i2);
		return dist;
	}
};

template<typename type, int dims>
template<typename System>
ndarray_iterator<type, dims, System> ndarray_ref<type,dims>::begin_it() const{
	return ndarray_iterator<type, dims, System>(compress_dims());
	//return ndarray_iterator<type, dims, System>(*this);
}

template<typename type, int dims>
template<typename System>
ndarray_iterator<type, dims, System> ndarray_ref<type,dims>::end_it() const{
	return ndarray_iterator<type, dims, System>(compress_dims()) + this->numel();
	//return ndarray_iterator<type, dims, System>(*this) + this->numel();
}

struct A{
	int x;
	__host__ __device__ A(){};
	__host__ __device__ A(const A & a){
#ifdef  __CUDA_ARCH__
		if(threadIdx.x<2){
			if(DEBUG_THIS)printf("% i, ====device copy constructor ==========\n", threadIdx.x);
		};
#else
		if(DEBUG_THIS)printf("====host copy constructor ==========\n");
		if(DEBUG_THIS)printf("src:  %p\n",(void*)&a);
		if(DEBUG_THIS)printf("dest: %p\n",(void*)this);
#endif
	};
};

__global__ void kernel_00(A a){
	a.x = 0;
	if(threadIdx.x==0){
		if(DEBUG_THIS)printf("address in kernel:  %p\n",(void*)&a);
	};
	A b = a;
	if(threadIdx.x<2){
		if(DEBUG_THIS)printf("%i : address of b:  %p\n",int(threadIdx.x),(void*)&b);
	};
}



void test_cuda_constructors(){
	//int * ptr;
	//cudaMallocManaged(&ptr, 1000*sizeof(int), cudaMemAttachGlobal);
	A a;
	kernel_00 <<< 1, 1024 >>>(a);
	//cudaDeviceSynchronize();
}

/*
*/
