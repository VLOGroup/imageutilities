#include "ndarray.h"
#include "ndarray_iterator.h"
#include <thrust/iterator/iterator_facade.h>
#include "ndarray_iterator_over.h"

namespace test{
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
	struct ndarray_iterator { // : public detail::ndarray_iterator_base<type, dims, System>::parent{
	public:
		typedef typename detail::ndarray_iterator_base<type, dims, System>::parent parent;
	public:
		// this is not changed by the iterator and hopefully can get optimized across copies in threads
		const kernel::ndarray_ref<type, dims> src;
		const intn<dims> st;
		const intn<dims> sz;
	public:
		intn<dims> ii;
		int dims1; // compressed size, dims1 <= dims
		//type __restrict__ * p;
		tstride<char> offset;
	public:
		inline __host__ __device__
		ndarray_iterator() = default;
		__host__
		ndarray_iterator(const ndarray_ref<type, dims> & x):src(x), dims1(src.nz_dims()),sz(x.size()),st(x.stride_bytes()){
			//p = x.ptr();
			ii = 0;
			offset = 0;
		}

		inline __host__ __device__
		ndarray_iterator(const ndarray_iterator<type, dims, System> & x) = default;
	public:
		//friend class thrust::iterator_core_access;
		__host__ __device__
		//! dereference
		typename parent::reference dereference() const{
			type *p = src.ptr() + offset;
			return *p;
		}
		//! equal
		inline __host__ __device__
		bool equal(const ndarray_iterator<type, dims, System> &other) const{
			//return p == other.p;
			return offset == other.offset;
		};

		// Incrementing
		inline __host__ __device__
		void increment(){
			for (int d = 0; d < dims; ++d){
				ii[d]++; // add to current index
				if (ii[d] < sz[d]){ //no overflow
					offset += st[d]; // advance pointer
					break;
				};
				if (d == dims1-1){ //nowhere to carry over
					offset = 0x7FFFFFFF; // advance pointer to max integer -- end marker
					break;
				};
				//carry
				ii[d] = 0;
				offset -= st[d]*(sz[d]-1); // adjust pointer
			};
		}

		// Advancing
		inline __host__ __device__
		void advance(typename parent::difference_type n) const{
			for (int d = 0; d < dims; ++d){
				int i_old = ii[d]; // save
				const_cast<int &>(ii[d]) += n; // add to current index
				if (ii[d] < sz[d]){ //no overflow
					const_cast<tstride<char> &>(offset) += (n * st[d]); // advance pointer
					break;
				};
				if (d == dims1-1){ //nowhere to carry over
					const_cast<tstride<char> &>(offset) = 0x7FFFFFFF; // advance pointer to max integer -- end marker
					break;
				};
			};
		}


		// Decrementing
		inline __host__ __device__
		void decrement(){
			int d = 0;
			ii[d] = src.size(d)-1;
			offset += src.stride(d) * ii[d];
		}

		// Distance
		inline __host__ __device__
		typename parent::difference_type
		distance_to(const ndarray_iterator<type, dims, System> &other) const{
			int i1 = other.src.size().index_to_integer(other.ii, other.dims1);
			int i2 = src.size().index_to_integer(ii, dims1);
			int dist = (i1 - i2);
			return dist;
		}
	};
};

struct System{
};

__global__ void test_kernel__00__(intn<3> ii, float * p){
	int i = ii[0]*2 + ii[1]*3 + ii[2]*4 + threadIdx.x;
	p[i] += 3.0f;
}

__global__ void test_kernel__01__(intn<3> ii, kernel::ndarray_ref<float, 3> a){
	int i = threadIdx.x +blockIdx.x*blockDim.x;
	a.ptr()[i] += a(ii);
}

__global__ void test_kernel__02__(const ndarray_iterator<float, 3,System> beg, const ndarray_iterator<float,3,System> end){
	int i =  threadIdx.x +blockIdx.x*blockDim.x;
	ndarray_iterator<float, 3,System> it1 = beg;
	ndarray_iterator<float, 3,System> it2 = beg;
	it1 += 2*i;
	it2 += 2*i+1;
	if(it1!= end){
		(*it1) += 3.0f;
	};
	if(it2!= end){
		(*it2) += 3.0f;
	};
}

__global__ void test_kernel__03__(const test::ndarray_iterator<float, 3,System> it){
	const int n = threadIdx.x +blockIdx.x*blockDim.x;
	const int dims = 3;
	const intn<dims> sz = it.src.size();
	const intn<dims> st = it.src.stride_bytes();
	intn<dims> ii; ii = 0;
	int offset = 0;
	//it.advance(i);
#pragma unroll
	for (int d = 0; d < dims; ++d){
		if(d <= it.dims1){
			int i_old = ii[d]; // save
			ii[d] += n; // add to current index
			if (ii[d] < sz[d]){ //no overflow
				offset += (n * st[d]); // advance pointer
			};
			if (d == it.dims1-1){ //nowhere to carry over
				offset = 0x7FFFFFFF; // advance pointer to max integer -- end marker
			};
		};
	};

	//it.dereference() += 3.0f;
	*(float *)((char *)it.src.ptr() + offset) += 3.0f;
}

__global__ void test_kernel__04__(const ndarray_iterator_over<float, 3,System> beg, const ndarray_iterator_over<float,3,System> end){
	int i =  threadIdx.x +blockIdx.x*blockDim.x;
	auto it = beg;
	it += i;
	if(it!= end){
		(*it) += 3.0f;
	};
}

void test_iterator(){
	ndarray<float,3> a;
	a.create<memory::GPU>(20,30,256);
	int B = 256;
	int G = (a.numel()+B-1)/B;
	//test_kernel__02__<<<G,B>>>(a.begin_it<System>(),a.end_it<System>());
}

int main(){
	test_iterator();
	return 0;
}
