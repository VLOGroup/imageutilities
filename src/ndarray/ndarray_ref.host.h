#pragma once

#include "ndarray_ref.kernel.h"
#include "error.h"
#include "error.kernel.h"
#include "type_name.h"

#ifndef  __CUDA_ARCH__
//#include <typeinfo>

#include <iostream>
#endif

template<typename compound_type> struct type_expand{
	typedef compound_type s_type;
	static const int n = 1;
};

#include "type_expand_cuda.h"

#ifndef  __CUDA_ARCH__
#define runtime_check_this(expression) if(!(expression))throw error_stream().set_file_line(__FILE__,__LINE__) << "Runtime check failed: " << #expression << *this << "\n"
#else
#define runtime_check_this(expression) stream_eater()
#endif

bool is_ptr_device_accessible(void * ptr);
bool is_ptr_host_accessible(void * ptr);
int ptr_access_flags(void * ptr);


// forward declaration of classes in iu from which conversion is provided, include "ndarray_iu.h"
namespace iu
{
	template<typename type, unsigned int> class LinearHostMemory;
	template<typename type, unsigned int> class LinearDeviceMemory;
	template<typename type, class Allocator> class ImageGpu;
	template<typename type, class Allocator> class ImageCpu;
	template<typename type, class Allocator> class VolumeGpu;
	template<typename type, class Allocator> class VolumeCpu;
	template<typename type> class TensorGpu;
	template<typename type> class TensorCpu;
	template<class iu_class> class proxy;
};

// forward declaration of mxArray, for matlab interface
struct mxArray_tag;

/*
template<typename T, typename U> struct member_ptr_t{
	typedef U T::* type;
	type ptr;
	member_ptr_t(type p):ptr(p){};
	operator type () {return ptr;};
};

template<typename U> struct member_ptr_t<float,U>{
	typedef void type;
};

template<typename U> struct member_ptr_t<int,U>{
	typedef void type;
};
 */
//____________________________flags______________________________________________________

struct ndarray_flags{
public:
	enum access{no_access = 0, host_only = 1, device_only = 2, host_device = 3};
protected:
	union{
		struct{
			unsigned int _access:2;//! access bits
			// depricated // unsigned int _linear_dim:4; //! which dimension has contiguous linear memory access
		} flag_bits;
		int flags;
	};
public:
	//! initialized default
	ndarray_flags(){
		set_defaults();
	}
	ndarray_flags(const ndarray_flags & b) = default;
	explicit ndarray_flags(int policy){
		set_defaults();
		flag_bits._access = policy;
	};
	ndarray_flags & operator = (const ndarray_flags & b) = default;
	void set_defaults(){
		flag_bits._access = no_access;
		//flag_bits._linear_dim = max_dimensions();
	}
//	static int max_dimensions(){
//		return 15; // for the 4 bit linear_dim above
//	}
//	int linear_dim() const{
//		return flag_bits._linear_dim;
//	}
//	template<int dim = 0>
//	bool is_linear_dim() const{
//		return flag_bits._linear_dim == dim;
//	}
	bool access_valid() const {
		return flag_bits._access != no_access;
	}
	int access() const {
		return flag_bits._access;
	}
	void set_access(int new_access){
		flag_bits._access = new_access;
	};
	bool host_allowed() const {
		return flag_bits._access == host_only || flag_bits._access == host_device;
	}
	bool device_allowed() const {
		return flag_bits._access == device_only || flag_bits._access == host_device;
	}
//	bool first_dim_is_linear() const {
//		return flag_bits._linear_dim == 0;
//	}
};

template<typename type, int dims> class ndarray_ref;
template<typename type, int dims, typename System> struct ndarray_iterator;
template<typename type, int dims, typename System> struct ndarray_iterator_over;

namespace base2{
	//_________________________ndarray_ref___________________________________________________
	//! part of general functionality, prior to specializations for communication with specific iu classes
	//! this is a __host__ only derived class
	template<typename type, int dims> class ndarray_ref : public kernel::ndarray_ref<type, dims>, public ndarray_flags{
	private:
		typedef kernel::ndarray_ref<type, dims> parent;
		typedef ::ndarray_ref<type, dims> return_t;
		return_t & self();
		const return_t & self() const;
	public: // inherited methods
		using parent::ptr;
		using parent::size;
		using parent::stride;
		using parent::stride_bytes;
		using parent::linear_stride_bytes;
	public: // constructors
		//! default uninitialized
		ndarray_ref() = default;
		//! default copy
		ndarray_ref(const ndarray_ref<type, dims> & x) = default;
		//! from a pointer
		ndarray_ref(type * const __beg, const intn<dims> & size, const intn<dims> & stride_bytes, int access_policy)
		:parent(__beg, size, stride_bytes), ndarray_flags(access_policy){
		}
		//! from a pointer, assuming linear arrangement
		ndarray_ref(type * const __beg, const intn<dims> & size, int access_policy){
			set_linear_ref(__beg, size, access_policy);
		}
		//! from a pointer, assuming linear arrangement, auto access flags from pointer attributes
		ndarray_ref(type * const __beg, const intn<dims> & size){
			set_linear_ref(__beg, size);
		}
	public: //____________ initializers
		//! default operator =
		ndarray_ref<type, dims> & operator = (const ndarray_ref<type, dims> & x) = default;
		//! from a pointer and shape
		ndarray_ref & set_ref(type * const __beg, const intn<dims> & size, const intn<dims> & stride_bytes, int access_policy);
		//! from a pointer assuming a linear layout
		template<bool order_ascending = true>
		ndarray_ref<type, dims> & set_linear_ref(type * const __beg, const intn<dims> & size, int access_policy);
		//! from a pointer assuming a linear layout and auto access flags from pointer attributes
		template<bool order_ascending = true>
		ndarray_ref & set_linear_ref(type * p, const intn<dims> & size);// defined later, pointer checking;

		// special constructors
		/*
		template<class Allocator, int D = dims, class = typename std::enable_if<D==2>::type >
		ndarray_ref & set_ref(const iu::ImageCpu<type, Allocator> & x);
		template<class Allocator, int D = dims, class = typename std::enable_if<D==2>::type >
		ndarray_ref(const iu::ImageCpu<type, Allocator> & x){
			set_ref(x);
		}
		*/
	public:
		//! helper conversion to the kernel base
		__HOSTDEVICE__ kernel::ndarray_ref<type, dims> & kernel(){ return *this; };
		//! helper conversion to the kernel base
		__HOSTDEVICE__ const kernel::ndarray_ref<type, dims> & kernel() const { return *this; };
	public: // additional shape / size functions
		//! shape
		shapen<dims> shape() const;
		//! The number of elements
		int numel() const;
		//! size of the memory support in bytes
		size_t size_bytes()const;
		//! check has the same shape as another ndarray_ref
		bool same_shape(const ndarray_ref<type, dims> & x) const;
		//! whether addresses contiguous memory (can be converted to linear memory)
		bool is_linear_ascending() const;
		//------------deleted---------// todo: make safe with negative strides
		//! whether strides are ascending
		bool strides_ascending() const = delete;
		//! whether strides are descending
		bool strides_descending() const = delete;
		//! get dimension ordering from strides - in increasing value of strides
		intn<dims> dims_order() const = delete;
		//! how many bytes padded on a given dimension (for the dimension with the smallest stride results in padded bytes per element)
		int dim_padding(int dim) const = delete;
		//! check strides are consistent (cover the size without overlap)
		bool strides_consistent() const = delete;
		//! check alignment of all dimensions, typically 4B and 16B alignments are needed (same as to check alignment of the smallest stride)
		bool aligned(int bytes) const = delete;
		//! check alignment on a given dimensions
		template<int dim> bool aligned(int bytes) const = delete;
		//-----------------------------
	public: // checked element access
		type * __restrict__ ptr(const intn<dims> & ii) const;
		//! pointer access: ptr(1,5,3)
		template<typename A0, typename... AA>
		type * __restrict__ ptr(A0 a, AA... aa) const;
		//! element access: operator()(1,5,3)
		template<typename A0, typename... AA>
		type & operator ()(A0 a, AA... aa) const;
		type & operator ()(const intn<dims> & ii) const;
		//! last element
		type * last()const;
		//! end = pointer to element after the last
		type * end() const;
	protected: // flags initialization
		void find_linear_dim();
	public:
		//! construct from the base class
		ndarray_ref(const kernel::ndarray_ref<type, dims> & x, int access_policy) : parent(x), ndarray_flags(access_policy){
//			find_linear_dim();
		}
	};

	//_________________________base::ndarray_ref_____________________________________________________
	//! from a pointer and shape
	template<typename type, int dims>
	ndarray_ref<type, dims> & ndarray_ref<type, dims>::set_ref(type * const __beg, const intn<dims> & size, const intn<dims> & stride_bytes, int access_policy){
		parent::set_ref(__beg, size, stride_bytes);
		set_access(access_policy);
		find_linear_dim();
		return *this;
	}

	//! initialize from a pointer assuming a linear layout
	template<typename type, int dims>
	template<bool order_ascending>
	ndarray_ref<type, dims> & ndarray_ref<type, dims>::set_linear_ref(type * const __beg, const intn<dims> & size, int access_policy){
		set_ref(__beg, size, parent::template linear_stride_bytes<order_ascending>(size), access_policy);
		return *this;
	}

	template<typename type, int dims>
	template<bool order_ascending>
	ndarray_ref<type, dims> & ndarray_ref<type, dims>::set_linear_ref(type * const __beg, const intn<dims> & size){
		set_linear_ref<order_ascending>(__beg, size, ptr_access_flags(__beg));
		return *this;
	}

	//! shape
	template<typename type, int dims>
	shapen<dims> ndarray_ref<type, dims>::shape() const{
		return shapen<dims>(size(), stride_bytes());
	}

	//! The number of elements
	template<typename type, int dims>
	int ndarray_ref<type, dims>::numel() const {
		return size().prod();
	}

	//! size of the memory support in bytes
	template<typename type, int dims>
	size_t ndarray_ref<type, dims>::size_bytes()const{
		int d = stride_bytes().max_idx();
		return size_t(size(d)) * stride_bytes(d);
	}

	//! check has the same shape as another ndarray_ref
	template<typename type, int dims>
	bool ndarray_ref<type, dims>::same_shape(const ndarray_ref<type, dims> & x) const{
		return (size() == x.size() && stride_bytes() == x.stride_bytes());
	}

	template<typename type, int dims>
	type * __restrict__ ndarray_ref<type, dims>::ptr(const intn<dims> & ii) const{
		for (int d = 0; d<dims; ++d) runtime_check_this(ii[d] >= 0 && ii[d] < size(d)) << "ii=" << ii << "\n";
		return parent::ptr(ii);
	}

	template<typename type, int dims>
	template<typename A0, typename...AA>
	inline type * __restrict__ ndarray_ref<type, dims>::ptr(A0 a0, AA... aa) const { // zero entries will be optimized out
		return ptr(intn<dims>(a0, aa...));
	}

	template<typename type, int dims>
	template<typename A0, typename...AA>
	inline type & ndarray_ref<type, dims>::operator ()(A0 a0, AA... aa) const {
		runtime_check(host_allowed());
		return *ptr(a0, aa...);
	}

	template<typename type, int dims>
	type & ndarray_ref<type, dims>::operator ()(const intn<dims> & ii) const {
		runtime_check(host_allowed());
		return *ptr(ii);
	}

	template<typename type, int dims>
	type * ndarray_ref<type, dims>::last()const{
		type * p = parent::begin();
		for (int i = 0; i < dims; ++i){
			//p += (size(i) - 1)*this->template stride<char>(i);
			p += (size(i) - 1)*stride(i);
		}
		return p;
	}

	//! end = pointer to element after the last
	template<typename type, int dims>
	type * ndarray_ref<type, dims>::end() const{
		//return last() + this->stride(0);
		return ptr() + size(dims-1)*this->stride(dims-1);
	}	

	template<typename type, int dims>
	void ndarray_ref<type, dims>::find_linear_dim(){
//		int d = stride_bytes().min_abs_idx();
//		if (stride_bytes(d) == intsizeof(type)){
//			flag_bits._linear_dim = d;
//		} else{
//			flag_bits._linear_dim = max_dimensions();
//		};
	}
/*
	template<typename type, int dims>
	intn<dims> ndarray_ref<type, dims>::dims_order() const {
		return stride_bytes().sort_idx();
	}
	//! how many bytes padded on a given dimension (for the dimension with the smallest stride results in padded bytes per element)
	template<typename type, int dims>
	int ndarray_ref<type, dims>::dim_padding(int dim) const {
		intn<dims> o = stride_bytes().sort_idx(); // increasing order
		// multiply out all preceding dimensions
		size_t lin_size_bytes = sizeof(type);
		for (int d = 0; d < o[dim]; ++d){ //padding of earlier deimensions
			lin_size_bytes *= size(o[d]);
		}
		// padding of dimension o[dim]
		return stride_bytes(dim) - lin_size_bytes;
	}
	//! check strides are consistent (cover the size without overlap)
	template<typename type, int dims>
	bool ndarray_ref<type, dims>::strides_consistent() const {
		intn<dims> o = stride_bytes().sort_idx(); // ascending ordering
		if (stride_bytes(o[0]) < sizeof(type)) return false; // smallest stride must cover the element size
		for (int d = 1; d < dims; ++d){
			int sz_from_stride = stride_bytes(o[d]) / stride_bytes(o[d - 1]);
			if (sz_from_stride < size(o[d - 1])) return false; // stride ratio is smaller than the size
		}
		return true;
	}

	template<typename type, int dims>
	bool ndarray_ref<type, dims>::strides_ascending() const{
		for (int d = 0; d < dims-1; ++d){
			if (stride_bytes(d) > stride_bytes(d + 1)) return false;
		};
		return true;
	}

	template<typename type, int dims>
	bool ndarray_ref<type, dims>::strides_descending() const{
		for (int d = 0; d < dims - 1; ++d){
			if (stride_bytes(d) < stride_bytes(d + 1)) return false;
		};
		return true;
	}

*/
	template<typename type, int dims>
	bool ndarray_ref<type, dims>::is_linear_ascending() const{
		if(stride_bytes(0) != sizeof(type) ) return false;
		for (int d = 0; d < dims-1; ++d){
			if(! (stride_bytes(d+1) == stride_bytes(d)*size(d)) ) return false;
		};
		return true;
	}

/*
	//! check alignment of all dimensions, typically 4B and 16B alignments are needed (same as to check alignment of the smallest stride)
	template<typename type, int dims>
	bool ndarray_ref<type, dims>::aligned(int bytes) const {
		for (int d = 0; d < dims; ++d){
			if (stride_bytes(d) % bytes != 0) return false;
		}
		return true;
	}
	//! check alignment on a given dimensions
	template<typename type, int dims>
	template<int dim>  bool ndarray_ref<type, dims>::aligned(int bytes) const {
		return (stride_bytes(dim) % bytes == 0);
	}
	*/
}

namespace special2{ // specializations depending on dimensions -- will have different additional constructors
	// generic case
	template<typename type, int dims> class ndarray_ref : public base2::ndarray_ref < type, dims > {
		typedef base2::ndarray_ref < type, dims> parent;
	public:
		// inherit constructors
		//using base2::ndarray_ref < type, dims >::ndarray_ref;
		inherit_constructors(ndarray_ref, parent)
		ndarray_ref() = default;
	};
	// 1D array -- nothing special
	template<typename type> class ndarray_ref<type, 1> : public base2::ndarray_ref < type, 1>{
		typedef  base2::ndarray_ref < type, 1> parent;
	public:
		// inherit constructors
		//using parent::parent;
		inherit_constructors(ndarray_ref, parent)
		ndarray_ref() = default;
		using parent::set_ref;
		//reverse conversions
		//operator iu::LinearHostMemory1d<type>();
		//operator iu::LinearDeviceMemory1d<type>();
	};
	// 2D array
	template<typename type> class ndarray_ref<type, 2> : public base2::ndarray_ref < type, 2>{
		typedef base2::ndarray_ref < type, 2> parent;
	public:
		// inherit constructors
		//using parent::parent;
		inherit_constructors(ndarray_ref, parent)
		//using parent::operator =;
		using parent::set_ref;

		ndarray_ref() = default;
		// special constructors
		template<class Allocator>
		ndarray_ref & set_ref(const iu::ImageCpu<type, Allocator> & x);
		template<class Allocator>
		ndarray_ref & set_ref(const iu::ImageGpu<type, Allocator> & x);
		template<class Allocator>
		ndarray_ref(const iu::ImageCpu<type, Allocator> & x){
			set_ref(x);
		}
		template<class Allocator>
		ndarray_ref(const iu::ImageGpu<type, Allocator> & x){
			set_ref(x);
		}
		/*
		//reverse conversions
		template<class Allocator>
		operator iu::ImageCpu<type, Allocator>();
		template<class Allocator>
		operator iu::ImageGpu<type, Allocator>();
		 */
	};
	// 3D array
	template<typename type> class ndarray_ref<type, 3> : public base2::ndarray_ref < type, 3>{
		typedef base2::ndarray_ref < type, 3> parent;
	public:
		// inherit constructors
		//using parent::parent;
		inherit_constructors(ndarray_ref, parent)
		//using parent::operator =;
		using parent::set_ref;
		ndarray_ref() = default;
		// special constructors
		template<class Allocator>
		ndarray_ref & set_ref(const iu::VolumeCpu<type, Allocator> & x);
		template<class Allocator>
		ndarray_ref & set_ref(const iu::VolumeGpu<type, Allocator> & x);

		template<class Allocator>
		ndarray_ref(const iu::VolumeCpu<type, Allocator> & x){
			set_ref(x);
		}
		template<class Allocator>
		ndarray_ref(const iu::VolumeGpu<type, Allocator> & x){
			set_ref(x);
		}
		template<class Allocator>
		ndarray_ref(const iu::VolumeCpu<type, Allocator> * x){
			set_ref(x);
		}
		template<class Allocator>
		ndarray_ref(const iu::VolumeGpu<type, Allocator> * x){
			set_ref(x);
		}
		/*
		//reverse conversions
		template<class Allocator>
		operator iu::VolumeCpu<type, Allocator>();
		template<class Allocator>
		operator iu::VolumeGpu<type, Allocator>();
		 */
	};
	// 4D array
	template<typename type> class ndarray_ref<type, 4> : public base2::ndarray_ref < type, 4>{
		typedef base2::ndarray_ref < type, 4> parent;
	public:
		// inherit constructors
		//using parent::parent;
		inherit_constructors(ndarray_ref, parent)
		//using parent::operator =;
		using parent::set_ref;
		ndarray_ref() = default;
		// special constructors
		ndarray_ref & set_ref(const iu::TensorCpu<type> & x);
		ndarray_ref & set_ref(const iu::TensorGpu<type> & x);
		ndarray_ref(const iu::TensorCpu<type> & x){
			set_ref(x);
		}
		ndarray_ref(const iu::TensorGpu<type> & x){
			set_ref(x);
		}
		ndarray_ref(const iu::TensorCpu<type> * x){
			set_ref(x);
		}
		ndarray_ref(const iu::TensorGpu<type> * x){
			set_ref(x);
		}
		/*
		//reverse conversions
		operator iu::TensorCpu<type>();
		operator iu::TensorGpu<type>();
		 */
	};
}

namespace special3{ //! specialisation on data type: struct type allows expansion
	template<typename type, int dims, bool is_class> class ndarray_ref;
	//elementary type
	template<typename type, int dims> class ndarray_ref<type,dims,false> : public special2::ndarray_ref < type, dims > {
		typedef special2::ndarray_ref < type, dims > parent;
	public:
		// inherit constructors
		//using parent::parent;
		inherit_constructors(ndarray_ref, parent)
		ndarray_ref() = default;
	};
	//non-elementary type
	template<typename type, int dims> class ndarray_ref<type,dims,true> : public special2::ndarray_ref < type, dims > {
		typedef special2::ndarray_ref < type, dims > parent;
	public:
		// inherit constructors
		//using parent::parent;
		inherit_constructors(ndarray_ref, parent)
		ndarray_ref() = default;
		//! slice a struct member from type
		template<typename U>
		::ndarray_ref<U,dims> subtype(U type::*member)const;
		//! expand struct as a new dimension
		::ndarray_ref<typename type_expand<type>::s_type, dims+1> unpack()const;
	};
}

//_______________________________________________________________________________________________
//struct type_A{
//};

//typedef type_A typeB;

//____________________final______________________________________________________________________
template<typename type, int dims> class ndarray_ref : public special3::ndarray_ref < type, dims, std::is_class<type>::value > {
	typedef special3::ndarray_ref < type, dims, std::is_class<type>::value > parent;
	typedef ::ndarray_ref<type, (dims>1)? dims-1 : 1> decrement_dim_type;
public:
	inherit_constructors(ndarray_ref, parent)
	ndarray_ref() = default;

	//using parent::operator =;
	using parent::access;
	using parent::ptr;
	using parent::size;
	using parent::stride_bytes;
	using parent::set_ref;

public:

	//! from LinearHostMemory1d and size
	ndarray_ref(const iu::LinearHostMemory<type, 1> & x, const intn<dims> & size);
	//! from LinearDeviceMemory1d and size
	ndarray_ref(const iu::LinearDeviceMemory<type, 1> & x, const intn<dims> & size);
    //ndarray_ref(const iu::LinearHostMemory<type, 1> * x, const intn<dims> & size);
    //ndarray_ref(const iu::LinearDeviceMemory<type, 1> * x, const intn<dims> & size);

	// special constructors from LinearDeviceMemory ND
	ndarray_ref & set_ref(const iu::LinearHostMemory<type, dims> & x);
	ndarray_ref & set_ref(const iu::LinearDeviceMemory<type, dims> & x);
	ndarray_ref(const iu::LinearHostMemory<type, dims> & x){
		set_ref(x);
	}
	ndarray_ref(const iu::LinearDeviceMemory<type, dims> & x){
		set_ref(x);
	}
	ndarray_ref(const iu::LinearHostMemory<type, dims> * x){
		set_ref(x);
	}
	ndarray_ref(const iu::LinearDeviceMemory<type, dims> * x){
		set_ref(x);
	}
	//! from mxArray * -- include mex_io.h
	ndarray_ref(const mxArray_tag *);
public: // operations
	//! reshape to get descending order of strides - last index fastest
	ndarray_ref<type, dims> reshape_descending() const = delete;
	//! reshape to get ascending order of strides - first index fastest
	ndarray_ref<type, dims> reshape_ascending() const = delete;
	//! reshape to a new size / dimension
	template<int dims2>
	ndarray_ref<type, dims2> reshape(const intn<dims2> & sz);
	//! reshape alias
	template<typename... Args>
	ndarray_ref<type, sizeof...(Args)> reshape(Args... args){
		return reshape(intn<sizeof...(Args)>{args...});
	}
public: // recast and slicing
	//! reinterpret same data as a different type (no type conversion)
	template<typename type2> ndarray_ref<type2, dims> recast()const;
	//! slice by fixing 1 dimension
	template<int dim1 = 0> ndarray_ref<type, dims - 1> subdim(int i_dim1) const;
	//! slice by fixing 2 dimensions
	template<int dim1, int dim2> ndarray_ref <type, dims - 2> subdim(int i_dim1, int i_dim2) const;
	// ! transpose (only the shape, for transposing data see ndarray_op.h)
	ndarray_ref<type, dims> transpose()const{return transp(); };
	// ! alias to transpose
	ndarray_ref<type, dims> transp()const;
	//! permute other two dimensions
	//template<int dim1, int dim2>  ndarray_ref<type, dims> permute_dims()const;
	ndarray_ref<type, dims> swap_dims(int dim1, int dim2) const;
	//! add a new virtual dimension (assocoated stride is zero)
	template<int ndim> ndarray_ref<type, dims+1> new_dim(int ndim_size) const;
	//! virtually crop to a smaller region
	ndarray_ref<type, dims> subrange(intn<dims> origin, intn<dims> new_size) const;
	//! permute dimensions according to substitution p
	ndarray_ref<type, dims> permute_dims(intn<dims> p) const;
	//! reflection on a given dimension -> negative strides
	ndarray_ref<type, dims> flip_dim(int dim) const;
	//! compress dimensions which can address the same data with fewer strides, e.g. linea memory with ascending strides compressed down to 1D
	ndarray_ref<type,dims> compress_dims()const;
	//! whether dimension is linearly addressable
	bool dim_linear(int d) const;
	//! check whether dimension d can be compressed with dimension d+1 (no gap in padding)
	bool dim_continuous(int d) const;
	//! compress dimensions (d1,d1+1), provided that dim_continuous(d1) is true
	decrement_dim_type compress_dim(int d1)const;
public: // convinience functions
	kernel::ndarray_ref<type,dims> & kernel(){return *this;};
	const kernel::ndarray_ref<type,dims> & kernel()const{return *this;};
public: //iterators
	//! experimental iterators that can be used in trust algorithms, not efficient
	template<typename System> ndarray_iterator<type, dims, System> begin_it() const;
	template<typename System> ndarray_iterator<type, dims, System> end_it() const;
	template<typename System> ndarray_iterator_over<type, dims, System> begin_it1() const;
	template<typename System> ndarray_iterator_over<type, dims, System> end_it1() const;
};
//_________________

//! recast
template<typename type, int dims>
template<typename type2> ndarray_ref<type2, dims> ndarray_ref<type, dims>::recast()const{
	// check: first dimension must be contiguous
	runtime_check_this(sizeof(type) == sizeof(type2) || this->template stride<char>(0) == intsizeof(type));
	return ndarray_ref<type2, dims>(parent::template recast<type2>(), access());
}

/*
//! reinterpret fixed-size vector data as a new dimension
template<typename type, int dims>
template<typename type2, int length> ndarray_ref<type2, dims+1> ndarray_ref<type, dims>::recast()const{
	runtime_check_this(this->template stride<char>(0) == intsizeof(type));
	intn<dims+1> sz2;
	intn<dims+1> st2;
}
 */

/*
template<typename type, int dims>
//template<typename tmember> ndarray_ref<tmember, dims> ndarray_ref<type, dims>::subtype(tmember type::* member)const{
//template<typename tmember, typename tmemberptr> ndarray_ref<tmember, dims> ndarray_ref<type, dims>::subtype(tmemberptr member)const{
template<typename tmember> ndarray_ref<tmember, dims> ndarray_ref<type, dims>::subtype(member_ptr_t<type,tmember> member)const{
	tmember * p2 = &(ptr()->*member);
	return ndarray_ref<tmember, dims>(p2 , size(), stride_bytes() , access());
}
 */

namespace special3{
	template<typename type, int dims>
	template<typename U>
	::ndarray_ref<U,dims>
	ndarray_ref<type,dims,true>::subtype(U type::*member)const{
		U * p2 = &(this->ptr()->*member);
		return ::ndarray_ref<U, dims>(p2 , this->size(), this->stride_bytes() , this->access());
	};

	template<typename type, int dims>
	::ndarray_ref<typename type_expand<type>::s_type, dims+1>
	ndarray_ref<type,dims,true>::unpack()const{
		typedef typename type_expand<type>::s_type type2;
		intn<dims+1> sz2 = this->size().template insert<0>(type_expand<type>::n);
		intn<dims+1> st2 = this->stride_bytes().template insert<0>(sizeof(type2));
		return ::ndarray_ref<type2, dims+1>((type2*)this->ptr(), sz2, st2 , this->access());
	};
};


//! slice by fixing 1 dimension
template<typename type, int dims>
template<int dim1> ndarray_ref<type, dims - 1> ndarray_ref<type, dims>::subdim(int i_dim1) const{
	runtime_check_this(i_dim1 >= 0 && i_dim1 < size(dim1)) << "i=" << i_dim1 << "\n";
	return ndarray_ref<type, dims - 1>(parent::template subdim<dim1>(i_dim1), access());
}

//! subdim
template<typename type, int dims>
template<int dim1, int dim2> ndarray_ref<type, dims - 2> ndarray_ref<type, dims>::subdim(int i_dim1, int i_dim2) const{
	runtime_check_this(i_dim1 >= 0 && i_dim1 < size(dim1)) << "i=" << i_dim1 << "\n";
	runtime_check_this(i_dim2 >= 0 && i_dim2 < size(dim2)) << "i=" << i_dim2 << "\n";
	return ndarray_ref<type, dims - 2>(parent::template subdim<dim1, dim2>(i_dim1, i_dim2), access());
}

/*
//! reshape to get descending order of strides - last index fastest
template<typename type, int dims>
ndarray_ref<type, dims> ndarray_ref<type, dims>::reshape_descending() const {
	intn<dims> o = stride_bytes().sort_idx(); // ascending orser sorting
	intn<dims> sz2;
	intn<dims> strideb2;
	for (int d = 0; d < dims; ++d){
		sz2[dims - d + 1] = size(o[d]);
		strideb2[dims - d + 1] = stride_bytes(o[d]);
	}
	return ndarray_ref<type, dims>(ptr(), sz2, strideb2, access());
}

//! reshape to get ascending order of strides - first index fastest
template<typename type, int dims>
ndarray_ref<type, dims> ndarray_ref<type, dims>::reshape_ascending() const {
	intn<dims> o = stride_bytes().sort_idx(); // ascending orser sorting
	intn<dims> sz2;
	intn<dims> strideb2;
	for (int d = 0; d < dims; ++d){
		sz2[d] = size(o[d]);
		strideb2[d] = stride_bytes(o[d]);
	}
	return ndarray_ref<type, dims>(ptr(), sz2, strideb2, access());
}
*/

//! reshape to a new size / dimension
template<typename type, int dims>
template<int dims2>
ndarray_ref<type, dims2> ndarray_ref<type, dims>::reshape(const intn<dims2> & sz2){
	//throw_error("not implemented");
	//product of sizes must match
	runtime_check(size().prod() == sz2.prod()) << "cannot reshape - different number of elements";
	runtime_check(size() > 0);
	runtime_check(sz2 > 0);
	// but also not any strides are possible, break into smallest product-matching groups
	intn<dims2> st2;
	int d1o = 0;
	int d2o = 0;
	int d1 = 0;
	int d2 = 0;
	int cs1 = size()[d1];
	int cs2 = sz2[d2];
	bool ascending = true;
	bool descending = true;
	while(d1 < dims && d2 < dims2){
		// skip over trivial dimensions
		while(d1 < dims && size()[d1] == 1)++d1;
		while(d2 < dims2 && sz2[d2] == 1)++d2;
		if(cs1 == cs2){ // found a group
			// group strides must be either ascending or descending contiguous
			if(ascending){
				int s = stride_bytes()[d1o];
				int cs = 1;
				for(int d = d2o; d <= d2; ++d){
					st2[d] = cs*s;
					cs = cs*sz2[d];
				};
			}else{ // descending
				runtime_check(descending) << "group strides are not consistent\n" << "sz1=" << size() << "\n sz2=" << sz2 << "\n st1=" << stride_bytes() << "\n range: " << d1o << "-" << d1 << "\n";
				int s = stride_bytes()[d1];
				int cs = 1;
				for(int d = d2; d >= d2o; --d){
					st2[d] = cs*s;
					cs = cs*sz2[d];
				};
			};
			// go to the next group
			if(d1 == dims-1){ //end of size
				runtime_check(d2 == dims2-1);
				break;
			}else{
				runtime_check(d2 < dims2-1) << "sz1=" << size() << "\n sz2=" << sz2 << "\n st1=" << stride_bytes() << "\n range: " << d1o << "-" << d1 << "\n" <<" d2=" << d2 <<"\n";
				++d1;
				++d2;
				d1o = d1;
				d2o = d2;
				cs1 = size()[d1];
				cs2 = sz2[d2];
				ascending = true;
				descending = true;
			};
		}else if(cs1 < cs2){
			// increment d1
			++d1;
			runtime_check(d1 < dims) << "sz1=" << size() << "\n sz2=" << sz2 << "\n st1=" << stride_bytes() << "\n range: " << d1o << "-" << d1 << "\n" <<" d2=" << d2 << "\n cs1="<< cs1 << " cs2=" << cs2;
			// check if ascending
			ascending = ascending && stride_bytes()[d1] % stride_bytes()[d1-1] == 0;
			descending = descending && stride_bytes()[d1-1] % stride_bytes()[d1] == 0;
			cs1 *= size()[d1];
		}else{ // cs1 > cs2
			++d2;
			runtime_check(d2 < dims2);
			cs2 *= sz2[d2];
		};
	};
	return ndarray_ref<type, dims2>(ptr(), sz2, st2, access());
}

// ! transpose (only the shape, for transposing data see ndarray_op.h)
template<typename type, int dims>
ndarray_ref<type, dims> ndarray_ref<type, dims>::transp()const{
	return swap_dims(0, 1);
}

/*
//! permute other two dimensions
template<typename type, int dims>
template<int dim1, int dim2>
ndarray_ref<type, dims> ndarray_ref<type, dims>::permute_dims() const{
	static_assert(dims >= 2, "can only transpose >= 2D");
	ndarray_ref<type, dims> r(*this);
	hd::swap(r.sz[dim1], r.sz[dim2]);
	hd::swap(r._stride_bytes[dim1], r._stride_bytes[dim2]);
	r.find_linear_dim();
	return r;
}
 */

//! permute other two dimensions
template<typename type, int dims>
ndarray_ref<type, dims> ndarray_ref<type, dims>::swap_dims(int dim1, int dim2) const{
	//static_assert(dims >= 2, "can only transpose >= 2D");
	runtime_check(dim1 >=0 && dim1 < dims);
	runtime_check(dim2 >=0 && dim2 < dims);
	ndarray_ref<type, dims> r(*this);
	hd::swap(r.sz[dim1], r.sz[dim2]);
	hd::swap(r._stride_bytes[dim1], r._stride_bytes[dim2]);
	r.find_linear_dim();
	return r;
}

//! add a new virtual dimension (assocoated stride is zero)
template<typename type, int dims>
template<int ndim> ndarray_ref<type, dims+1> ndarray_ref<type, dims>::new_dim(int ndim_size) const{
	static_assert(ndim>=0 && ndim <= dims, "bad ndim");
	intn<dims+1> sz2;
	intn<dims+1> st2;
	for(int d=0; d< dims; ++d){
		int d1 = d>=ndim? d+1:d;
		sz2[d1] = size(d);
		st2[d1] = stride_bytes(d);
	};
	sz2[ndim] = ndim_size;
	st2[ndim] = 0;
	return ndarray_ref<type, dims+1>(ptr(),sz2,st2,access());
};

//! virtually crop to a smaller region
template<typename type, int dims>
ndarray_ref<type, dims> ndarray_ref<type, dims>::subrange(intn<dims> origin, intn<dims> new_size) const{
	type *p = ptr(origin);
	return ndarray_ref<type, dims>(p,new_size,stride_bytes(),access());
};

//! permute dimensions according to substitution p
template<typename type, int dims>
ndarray_ref<type, dims> ndarray_ref<type, dims>::permute_dims(intn<dims> p) const{
	intn<dims> sz2;
	sz2 = -1;
	intn<dims> st2;
	for(int d=0; d<dims; ++d){
		runtime_check(p[d] >= 0 && p[d] < dims) << "dimensions not in range\n";
		sz2[d] = size(p[d]);
		st2[d] = stride_bytes(p[d]);
	};
	runtime_check(sz2 >=0) << "p= " << p << " is not a proper permutation\n";
	return ndarray_ref<type, dims>(ptr(),sz2,st2,access());
}

//! reflection on a given dimension -> negative strides
template<typename type, int dims>
ndarray_ref<type, dims> ndarray_ref<type, dims>::flip_dim(int dim) const{
	runtime_check(dim>=0 && dim < dims);
	intn<dims> st2 = stride_bytes();
	type * p = ptr() + (size(dim)-1)*this->stride(dim);
	st2[dim] = -st2[dim];
	return ndarray_ref<type, dims>(p,size(),st2,access());
}

//! compress_dims
template<typename type, int dims>
ndarray_ref<type, dims> ndarray_ref<type, dims>::compress_dims()const{
	const ndarray_ref<type, dims> & x = *this;
	ndarray_ref<type, dims> r = x;
	r.size() = 0;
	r.stride_bytes() = 0;
	int dims1 = 0;
	int s = x.size(0);
	int st = x.stride_bytes(0);
	for(int d = 1; d< dims; ++d){
		if(x.size(d)==1 || x.stride_bytes(d) == st*s){// no padding from d-1 to d -- can compress
			s *= x.size(d); // multiply out and treat linearly
			continue;
		}else{// index is discontinuous - cannot compress
			r.size()[dims1] = s;
			r.stride_bytes()[dims1] = st;
			st = x.stride_bytes(d);
			s = x.size(d);
			++dims1;
		};
	};
	r.size()[dims1] = s;
	r.stride_bytes()[dims1] = st;
	++dims1; // compressed dimensions
	//std::cout << "compressed to dims1=" <<dims1 <<": " << r << "\n";
	return r;
}

//! whether dimension is linearly addressable
template<typename type, int dims>
bool ndarray_ref<type, dims>::dim_linear(int d) const{
	return stride_bytes(d) == sizeof(type);
};

//! test whether dimension d is continuous
/* true if can go continuously from d to d+1 and thus compress d
 * in case size(d)== 1 || size(d+1)==1, the respective stride does not matter, condition is true
 * it is also true if both strides are zero
 */
template<typename type, int dims>
bool ndarray_ref<type, dims>::dim_continuous(int d) const {
	if(d == dims-1) return false;
	runtime_check(d>=0 && d < dims-1);
	return size(d)==1 || size(d+1)==1 || stride_bytes(d+1) == stride_bytes(d)*size(d);// can go continuously from d to d+1
}

//! compress dimensions (d1,d1+1), provided that dim_continuous(d1) is true
template<typename type, int dims>
typename ndarray_ref<type, dims>::decrement_dim_type ndarray_ref<type, dims>::compress_dim(int d1)const{
	runtime_check(dim_continuous(d1));
	const ndarray_ref<type, dims> & x = *this;
	auto sz = x.size().erase(d1+1);
	auto st = x.stride_bytes().erase(d1+1);
	sz[d1] = sz[d1] * x.size(d1+1);
	return decrement_dim_type(x.ptr(), sz, st, x.access());
}

/*
//__CRTP_definitions_______________________
namespace base2{
	template<typename type, int dims>
	ndarray_ref<type, dims>::return_t & ndarray_ref<type, dims>::self(){
		//now the type return_t is fully known to the compiler and it is derived from this, static_cast is safe
		return static_cast<ndarray_ref<type, dims>::return_t & >(*this);
	};
	template<typename type, int dims>
	const ndarray_ref<type, dims>::return_t & ndarray_ref<type, dims>::self() const{
		return static_cast<const ndarray_ref<type, dims>::return_t &>(*this);
	};
};
 */

//_________________________________external_____________________________
//______________________________________________________________________
template<typename type, int dims> ndarray_ref<type,dims> make_ndarray_ref(type * p, intn<dims> size, int access = ndarray_flags::no_access){
	if(access == ndarray_flags::no_access){
		return ndarray_ref<type,dims>(p,size); // extract from the pointer
	}else{
		return ndarray_ref<type,dims>(p,size,access);
	};
}
//____________________________________________________________________


//_____________________kernel_________________


namespace kernel{
	template<typename type, int dims>
	__HOST__ ndarray_ref<type,dims>::ndarray_ref(const ::ndarray_ref<type,dims> & derived){
		*this = derived;
	}

	template<typename type, int dims>
	__HOST__ ndarray_ref<type, dims> & ndarray_ref<type,dims>::operator = (const ::ndarray_ref<type, dims> & derived){
		if(!derived.device_allowed()){
			throw_error("entering kernel for this array is not permitted") << derived;
		};
		//std::cout << "array entering kernel: " << derived << "\n";
		const ndarray_ref<type,dims> * base = &derived;
		*this = *base;
		return *this;
	}
}


//_________________________flags__________

template <typename tstream> void print_flags(tstream & ss, const ndarray_flags & ff){
	ss << "access: ";
	switch(ff.access()){
		case ndarray_flags::no_access : ss << "no_access";break;
		case ndarray_flags::host_only : ss << "host";break;
		case ndarray_flags::device_only : ss << "device";break;
		case ndarray_flags::host_device : ss << "host, device"; break;
		default: throw_error("unexpected case");
	};
	//ss << "; linear_dim: " << ff.linear_dim();
}

template <typename type, int dims, typename tstream> void print_array(tstream & ss, const ndarray_ref<type,dims> & a){
#ifndef  __CUDA_ARCH__
	//ss << "\n ndarray_ref<" << typeid(type).name() << "," << dims << ">:" << "ptr="<<a.ptr() << ", size=" << a.size() << ", strides_b=" << a.stride_bytes();
	ss << "\n ndarray_ref<" << type_name<type>() << "," << dims << ">:" << "ptr="<<a.ptr() << ", size=" << a.size() << ", strides_b=" << a.stride_bytes();
	const ndarray_flags & ff = a;
	ss << ", " << ff;
#endif
}

inline error_stream & operator << (error_stream & ss, const ndarray_flags & ff){
	print_flags(ss,ff);
	return ss;
}

template <typename type, int dims> error_stream & operator << (error_stream & ss, const ndarray_ref<type,dims> & a){
	print_array(ss,a);
	return ss;
}

inline std::ostream& operator << (std::ostream & ss, const ndarray_flags & ff){
	print_flags(ss,ff);
	return ss;
}

template <typename type, int dims> std::ostream & operator << (std::ostream & ss, const ndarray_ref<type,dims> & a){
	print_array(ss,a);
	return ss;
}
