#pragma once

/*!
	This header defines ndarray_ref, a versatile placeholder to pass arrays around, with light dependencies
	- can be passed into cuda kernels
	- supports various padding (alignment)
	- operations are available through ndarray_op.h | ndarray_op.cu
	- memory allocation functionality is available through ndarray_mem.h | ndarray_mem.cpp
 */

#include <assert.h>
#include "intn.h"
#include "defines.h"


#define intsizeof(type) int(sizeof(type))


namespace hd{

	template<typename type>
	__host__ __device__ __forceinline__ void swap(type & a, type & b){
		const type c = a;
		a = b;
		b = c;
	}
}

//_______________________________tstride_____________________________________________
//! tstride: safe pointer arithmetics with strides measured diffrently, e.g. in types or in bytes

template <typename type> struct tstride{
protected:
	int type_stride;
public:
	__host__ __device__ __forceinline__ tstride(){
	}
	//! default copy ctr
	__host__ __device__ __forceinline__ tstride(const tstride<type> & b) = default;
	//! construct from int
	__host__ __device__ __forceinline__ tstride(int stride_in_types):type_stride(stride_in_types){
	}
	//! default operator =
	__host__ __device__ __forceinline__ tstride<type> & operator = (const tstride<type> & b) = default;
	//! assign from int
	__host__ __device__ __forceinline__ void operator = (int stride_in_types){
		type_stride = stride_in_types;
	}
	//! if you really need to know the stored value
	__host__ __device__ __forceinline__ int value()const{
		return type_stride;
	}
public: //arithmetics
	//! unary -
	__host__ __device__ __forceinline__ tstride<type> operator -() const {
		return tstride<type>(-type_stride);
	}
	//
	__host__ __device__ __forceinline__ tstride<type> & operator *=(int x){
		type_stride *= x;
		return *this;
	}

	__host__ __device__ __forceinline__ tstride<type> operator *(int x) const {
		tstride<type> r = *this;
		r *= x;
		return r;
	}

	//
	template <typename ptype>
	__host__ __device__ __forceinline__ ptype * __restrict__ & step_pointer(ptype *& p) const {
		return (ptype *&)((char*&)p += type_stride*intsizeof(type));
	}

#ifndef __restrict__
	template <typename ptype>
	__host__ __device__ __forceinline__ ptype * __restrict__ & step_pointer(ptype * __restrict__ & p) const {
		return (ptype *&)((char*&)p += type_stride*intsizeof(type));
	}
#endif

	//
	__host__ __device__ __forceinline__ tstride<type> & operator += (const tstride<type> & b){
		type_stride += b.value();
		return *this;
	}
	//
	__host__ __device__ __forceinline__ tstride<type> & operator -= (const tstride<type> & b){
		type_stride -= b.value();
		return *this;
	}
	__host__ __device__ __forceinline__ tstride<type> operator + (const tstride<type> & b) const {
		return tstride<type>(value()+b.value());
	}

	__host__ __device__ __forceinline__ bool operator == (const tstride<type> & b) const{
		return type_stride == b.type_stride;
	}

	__host__ __device__ __forceinline__ bool operator == (int v) const{
		return type_stride == v;
	}
public:// conversions
	template <typename type2>
	__host__ __device__ __forceinline__ void operator = (const tstride<type2> & b){
		type_stride = (b.type_stride * intsizeof(type2)) / intsizeof(type);
	}
	template <typename type2>
	explicit __host__ __device__ __forceinline__ tstride(const tstride<type2> & b){
		(*this) = b;
	}
};
//! tstride arithmetics
template <typename type>
__host__ __device__ __forceinline__ tstride<type> operator *(int x, tstride<type> s){
	return s *= x;
}
//! pointer arithmetics
template <typename ptype, typename stype>
__host__ __device__ __forceinline__ ptype * __restrict__ & operator += (ptype * __restrict__ & p, const tstride<stype> & s){
	return s.step_pointer(p);
}

template <typename ptype, typename stype>
__host__ __device__ __forceinline__ ptype * __restrict__ operator + (ptype * __restrict__ p, const tstride<stype> & s){
	ptype * P = p;
	return s.step_pointer(P);
}

template <typename ptype, typename stype>
__host__ __device__ __forceinline__ ptype * __restrict__ & operator -= (ptype * __restrict__ & p, const tstride<stype> & s){
	return (-s).step_pointer(p);
}

template <typename ptype, typename stype>
__host__ __device__ __forceinline__ ptype * __restrict__ operator - (ptype * __restrict__ p, const tstride<stype> & s){
	ptype * P = p;
	return (-s).step_pointer(P);
}

//_______________________________dslice_____________________________________________
//! compact reference class, useful inside kernels
/*
  Consist of a pointer and strides - offsets in bytes to step the pointer in each dimension
  strides are in bytes rather than in elements
	- padding in bytes is more general than in elements
	- saves a multiplication by sizeof(type) in address calculation (expensive 64 bit integr operation in GPU)
 */
template<typename type, int dims> struct dslice{
protected:
	type * _beg;  // first element
	//__host__ __device__ __forceinline__ char * beg_bytes() { return (char*)_beg; }
	//__host__ __device__ __forceinline__ char * beg_bytes() const { return (char*)_beg; }
	intn<dims> _stride_bytes;	//!< offsets to next element along each dimension, In Bytes, can be negative
public: // strides access
	//! stride along given dimension, as a safe proxy tstride<stype>
	template <typename stype = char>
	__host__ __device__ __forceinline__ tstride<stype> stride(int d) const {
		return tstride<stype> (_stride_bytes[d] / intsizeof(stype) );
	}
	//! whole vector of strides, in bytes
	__host__ __device__ __forceinline__ const intn<dims> & stride_bytes() const {
		return _stride_bytes;
	}
	//! whole vector of strides, in bytes
	__host__ __device__ __forceinline__ intn<dims> & stride_bytes(){
		return _stride_bytes;
	}
	//! single stride in bytes
	__host__ __device__ __forceinline__ const int & stride_bytes(int d) const {
		return _stride_bytes[d];
	}
	//! single stride in bytes
	__host__ __device__ __forceinline__ int & stride_bytes(int d){
		return _stride_bytes[d];
	}
public: // stride access
	//! get strides for linear memory - ascendin (first index fasterst) or descending order
	template<bool order_ascending = true>
	static __host__ __device__ __forceinline__ intn<dims> linear_stride_bytes(const intn<dims> & size){
		return size.template prefix_prod_ex<order_ascending>() *= intsizeof(type);
	}
public: // __________ constructors
	__host__ __device__ __forceinline__ dslice() = default;
	//! default copy ctr
	__host__ __device__ __forceinline__ dslice(const dslice & b) = default;
	//! default operator =
	__host__ __device__ __forceinline__ dslice& operator = (const dslice & b) = default;
public:
	//! construct from pointer and array of strides (count in bytes)
	__host__  __device__ __forceinline__ dslice(type * p, const intn<dims> & __stride_bytes){
		set_ref(p,__stride_bytes);
	}
	//! initialize from pointer and array of strides (count in bytes)
	__host__  __device__ __forceinline__ void set_ref(type * p, const intn<dims> & __stride_bytes){
		_beg = p;
		_stride_bytes = __stride_bytes;
	}
public:
	//! reinterpret as array of other base type (type2)
	template<typename type2>
	__host__  __device__ __forceinline__ dslice<type2, dims> recast()const{
		intn<dims> stb = _stride_bytes;
		stb[0] = _stride_bytes[0] * intsizeof(type2) / intsizeof(type);
		return dslice<type2, dims>((type2*)_beg, stb);
	}
public: //____________ element access
	__host__ __device__ __forceinline__ type * __restrict__ begin() const { return _beg; }
	__host__ __device__ __forceinline__ type & operator *() const { return *begin(); }
	__host__ __device__ __forceinline__ type * __restrict__ & ptr() { return _beg; }
	__host__ __device__ __forceinline__ type * __restrict__ const & ptr() const { return _beg; }
	__host__ __device__ __forceinline__ type * __restrict__ ptr(const intn<dims> & ii) const{
		type * p = begin();
		for (int d = 0; d < dims; ++d){
			p += ii[d] * stride<char>(d);
		}
		return p;
	}
	__host__ __device__ __forceinline__ type * __restrict__ ptr(int i0 = 0, int i1 = 0, int i2 = 0, int i3 = 0) const { // zero entries will be optimized out
		type * p = begin() + i0 * stride<char>(0);
		if (dims > 1){
			p += i1 * stride<char>(1);
		}
		if (dims > 2){
			p += i2 * stride<char>(2);
		}
		if (dims > 3){
			p += i3 * stride<char>(3);
		}
		return p;
	}
	__host__ __device__ __forceinline__ type & operator ()(int i0 = 0, int i1 = 0, int i2 = 0, int i3 = 0) const {
		return *ptr(i0, i1, i2, i3);
	}
	__host__ __device__ __forceinline__ type & operator ()(const intn<dims> & ii) const {
		return *ptr(ii);
	}

public: //___________offset and slice
	//! offset to a position
	__host__ __device__ __forceinline__ dslice<type, dims> offset(const intn<dims> & ii) const {
		dslice<type, dims> r;
		r._beg = ptr(ii);
		r._stride_bytes = _stride_bytes;
		return r;
	}
	//
	__host__ __device__ __forceinline__ dslice<type, dims> offset(int i0, int i1 = 0, int i2 = 0, int i3 = 0) const {
		return offset(intn<dims>(i0,i1,i2,i3));
	}
	//! slice by fixing a specific dimension
	template<int dim1 = 0>
	__host__ __device__ __forceinline__ dslice <type, dims - 1 > subdim(int i_dim1){
		static_assert(dims > 1, "subndarray_ref of 1D is just pointer");
		type * p = begin() + i_dim1 * stride(dim1);
		int stride_bytes[dims - 1];
		for (int d = 0; d < dims; ++d){
			if (d == dim1)continue;
			stride_bytes[d - (d>dim1)] = _stride_bytes[d];
		}
		return dslice <type, dims - 1 >(p, stride_bytes);
	}
public: //___________iterating over the array
	//! increment a pointer along a given dimension
	template<int dim>
	__host__ __device__ __forceinline__ void step_p_dim(type *& p, int step = 1) const{
		p += stride<char>(dim) * step;
	}
	//! increment self along a given dimension
	template<int dim>
	__host__ __device__ __forceinline__ void step_self_dim(int step = 1){
		_beg += stride<char>(dim) * step;
	}
	//! reverse the direction along a dimension
	template<int dim = 0>
	__host__ __device__ __forceinline__ dslice<type, dims> & reverse_dim(){
		_stride_bytes[dim] = -_stride_bytes[dim];
		return *this;
	}
	//! conditional reverse the direction along a dimension
	template<int dim = 0>
	__host__ __device__ __forceinline__ dslice<type, dims> & direction(int dir){
		if(dir< 0){
			return reverse_dim<dim>();
		}else{
			return *this;
		};
	}
	//! permutes dims 0 and 1
	__host__ __device__ __forceinline__ dslice<type, dims> transp() const{
		static_assert(dims == 2, "can only transpose 2D");
		return dslice(_beg, _stride_bytes[1], _stride_bytes[0]);
	}

	//! permute other two dimensions
	template<int dim1, int dim2>
	__host__ __device__ __forceinline__ dslice<type, dims> permute_dims() const{
		dslice<type, dims> r(*this);
		hd::swap(r._stride_bytes[dim1], r._stride_bytes[dim2]);
		return r;
	}
};
//____________________________shape______________________________________________________
//! this is just a shorthand proxy class to commonly handle (size, strides)
template<int n> class shapen{
public:
	intn<n> sz;
	intn<n> stride_bytes;
public:
	__host__ __device__ __forceinline__ shapen(const intn<n> & _size, const intn<n> & _stride_bytes):sz(_size),stride_bytes(_stride_bytes){}
	__host__ __device__ __forceinline__ int size(int i) const {return sz[i];}
	__host__ __device__ __forceinline__ const intn<n> & size() const {return sz;}
	//! default copy constructor
	//! default operator =
};

namespace kernel{
	template <typename type , int dims> class ndarray_ref;
}

template <typename type , int dims> class ndarray_ref;

//template <typename type, int dims, typename tstream> tstream & operator << (tstream & ss, const ndarray_ref<type,dims> & a);

namespace kernel{
	//_________________________kernel::ndarray_ref___________________________________________________
	//! kernel::ndarray_ref -- allowed to go to kernel
	template <typename type , int dims> class ndarray_ref : public dslice<type,dims>{
	private:
		typedef dslice<type, dims> parent;
	public:
		static constexpr int my_dims = dims;
		typedef type my_type;
		typedef intn<dims> my_index;
	public:
		intn<dims> sz; //!< size in elements
	public: // inherited stuff
		using parent::begin;
		using parent::ptr;
		using parent::stride;
		using parent::linear_stride_bytes;
	protected:// inherited stuff for protected use
		using parent::_beg;
		using parent::_stride_bytes;
	public: //constructors
		//! uninitialized
		__host__ __device__ __forceinline__ ndarray_ref() = default;
		//! default copy
		__host__ __device__ __forceinline__ ndarray_ref(const ndarray_ref<type,dims> & x) = default;
		//! copy from a host ndarray_ref array (it is inherited, but the copy is a checking barrier)
		__host__ ndarray_ref(const ::ndarray_ref<type,dims> & derived);
		//! from a pointer
		__host__ __device__ __forceinline__ ndarray_ref(type * const __beg, const intn<dims> & size, const intn<dims> & stride_bytes){
			set_ref(__beg,size,stride_bytes);
		}
		//! copy from dslice
		__host__ __device__ __forceinline__ ndarray_ref(const ::dslice<type,dims> & x, const intn<dims> & size):parent(x),sz(size){
		}
	public: //____________ initializers
		//! copy =
		__host__ __device__ __forceinline__ ndarray_ref<type, dims> & operator = (const ndarray_ref<type, dims> & x) = default;
		//! copy from derived (host-device barrier)
		__host__ ndarray_ref<type, dims> & operator = (const ::ndarray_ref<type, dims> & x);
		//! from a pointer and shape
		__host__ __device__ __forceinline__ ndarray_ref & set_ref(type * const __beg, const intn<dims> & size, const intn<dims> & stride_bytes){
			parent::set_ref(__beg, stride_bytes);
			sz = size;
			return *this;
		}
	public: //____________ size and shape
		//! full size
		__host__ __device__ __forceinline__ const intn<dims> & size()const{
			return sz;
		}
		//! full size
		__host__ __device__ __forceinline__ intn<dims> & size(){
			return sz;
		}
		//! size along a dimension
		__host__ __device__ __forceinline__ int size(int dim)const{
			return sz[dim];
		}
		//! size along a dimension
		__host__ __device__ __forceinline__ int & size(int dim){
			return sz[dim];
		}
		//! count the number of dimensions until the first zero size
		int nz_dims()const{
			for(int d = 0; d < dims; ++d){
				if(size(d)==0) return d; // hit first zero dimension
			};
			return dims;
		}
	public:
		//! element type conversion
		template<typename type2>
		__host__ __device__ __forceinline__ ndarray_ref<type2, dims> recast()const{
			intn<dims> sz2 = sz;
			sz2[0]  = sz[0] * intsizeof(type) / intsizeof(type2); // new size along dim 0
			return ndarray_ref<type2, dims>(parent::template recast<type2>(), sz2);
		}
	public: //____________ additional element access
		__host__ __device__ __forceinline__ type * __restrict__ ptr(const intn<dims> & ii) const{
			return parent::ptr(ii);
		}
		__host__ __device__ __forceinline__ type * __restrict__ ptr(int i0, int i1 = 0, int i2 = 0, int i3 = 0) const { // zero entries will be optimized out
			return parent::ptr(i0,i1,i2,i3);
		}
		__host__ __device__ __forceinline__ type & operator ()(int i0, int i1 = 0, int i2 = 0, int i3 = 0) const {
			return *ptr(i0, i1, i2, i3);
		}
		__host__ __device__ __forceinline__ type & operator ()(const intn<dims> & ii) const {
			return *ptr(ii);
		}
		__host__ __device__ __forceinline__ type & operator [](const intn<dims> & ii) const {
			return *ptr(ii);
		}
	public: //____________ slicing
		//! slice by fixing one dimension
		template<int dim1 = 0> // default is to fix the first dimension
		__host__ __device__ __forceinline__ ndarray_ref <type, dims-1> subdim(int i_dim1) const{
			static_assert(dim1 >= 0 && dim1 < dims, "bad");
			static_assert(dims > 1, "1-subdim requires dimension 2 or bigger");
			int size[dims - 1];
			int strideb[dims - 1];
			int k = 0;
			type * p = ptr();
			for (int i = 0; i < dims; ++i){
				if (i == dim1){ // fixed dimension
					p += stride(dim1)*i_dim1;
					continue;
				}
				size[k] = this->size(i);
				strideb[k] = _stride_bytes[i];
				++k;
			}
			ndarray_ref<type, dims - 1 > r(p, size, strideb);
			return r;
		}
		//! ndarray_ref by fixing two dimensions
		template<int dim1, int dim2>
		__host__ __device__ __forceinline__ ndarray_ref <type, dims-2> subdim(int i_dim1, int i_dim2) const{
			static_assert(dims > 2, "2-subdim requires dimension 3 or bigger");
			static_assert(dim1 >= 0 && dim1 < dims,"bad");
			static_assert(dim2 >= 0 && dim2 < dims,"bad");
			static_assert(dim1 != dim2,"bad");
			intn<dims-2> size;
			intn<dims-2> strideb;
			type * p = ptr();
			int k = 0;
			for (int i = 0; i < dims; ++i){
				if (i == dim1){
					p += stride(dim1)*i_dim1;
					continue;
				}
				if (i == dim2){
					p += stride(dim2)*i_dim2;
					continue;
				}
				size[k] = this->size(i);
				strideb[k] = _stride_bytes[i];
				++k;
			}
			return ndarray_ref<type, dims-2>(p, size, strideb);
		}
	};
}

/*
template <typename Kernel, typename... Args> void launch(dim3 dimBlock, dim3 dimGrid, Kernel kernel, Args... args){
	kernel <<< dimGrid, dimBlock >>>(args...);
}
*/

//__CUDA_ARCH__is always undefined when compiling host code, steered by nvcc or not
//__CUDA_ARCH__is only defined for the device code trajectory of compilation steered by nvcc
