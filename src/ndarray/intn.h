#pragma once

#include "defines.h"

#ifdef  __CUDA_ARCH__
#include <cuda.h>
#include "error.kernel.h"
#else
#include <algorithm>
#include "error.h"
#include <cstddef>
#include <type_traits>
#endif

//________________________________intn______________________________________________
//! specializations of intn<n> for lower dimensions
namespace special{
	//! generic dimensions - array storage
	template<int n> struct intn{
	public:
		typedef int array_type[n];
		array_type v;
		__HOSTDEVICE__ intn() = default;
	protected://__________
		__HOSTDEVICE__ array_type & as_array(){ return v; }
		__HOSTDEVICE__ const array_type & as_array()const{ return v; }
		__HOSTDEVICE__ int & V(int i){
			// instead of return v[i] make explicit loop: when i is known at compile time - optimized out, when i is dynamic should optimize as a switch
			for(int j = 0; j<n; ++j ){
				if(j==i) return v[j];
			};
			//runtime_check(false) << "index failed";
			return v[0];
		};
		__HOSTDEVICE__ const int & V(int i)const{
			return const_cast<intn<n>&>(*this).V(i);
		};
	};
	//------------------
	//! 1D specialization
	template<> struct intn<1>{
	public:
		typedef int array_type[1];
	public:
		int width;
		static const int height = 1;
		static const int depth = 1;
		__HOSTDEVICE__ intn() = default;
		//__HOSTDEVICE__ intn(int a0){V(0) = a0;};
		// intn<1> is convertible to int
		__HOSTDEVICE__ operator int & (){return V(0);}
		__HOSTDEVICE__ operator const int & ()const {return V(0);}
	protected://__________
		__HOSTDEVICE__ array_type & as_array(){
			return *(array_type*)&width;
		}
		__HOSTDEVICE__ const array_type & as_array()const{
			return *(const array_type *)&width;
		}
		__HOSTDEVICE__ int & V(int i){return width;};
		__HOSTDEVICE__ const int & V(int i)const{return width;};
	};
	//------------------
	//! 2D specialization
	template<> struct intn<2>{
	public:
		typedef int array_type[2];
	public:
		int width;
		int height;
		static const int depth = 1;
		__HOSTDEVICE__ intn() = default;
	protected://__________
		__HOSTDEVICE__ array_type & as_array(){
			static_assert(offsetof(intn<2>,height)==sizeof(int), "struct linear layout assumption failed");
			return *(array_type*)&width;
		}
		__HOSTDEVICE__ const array_type & as_array()const{
			return const_cast<intn<2> *>(this)->as_array();
		}

		__HOSTDEVICE__ int & V(int i){
			switch(i){
				case 1: return height;
				default: return width;
			};
			//return as_array()[i];
		}
		//__HOSTDEVICE__ const int & V(int i)const{return as_array()[i];};
		__HOSTDEVICE__ const int & V(int i)const{
			return const_cast<intn<2>&>(*this).V(i);
		}
	};
	//------------------
	//! 3D specialization
	template<> struct intn<3>{
	public:
		typedef int array_type[3];
	public:
		int width;
		int height;
		int depth;
		__HOSTDEVICE__ intn() = default;
	protected://__________
		__HOSTDEVICE__ array_type & as_array(){
			static_assert(offsetof(intn<3>,height)==sizeof(int), "struct linear layout assumption failed");
			static_assert(offsetof(intn<3>,depth)==2*sizeof(int), "struct linear layout assumption failed");
			return *(array_type*)&width;
		}
		__HOSTDEVICE__ const array_type & as_array()const{
			return const_cast<intn<3> *>(this)->as_array();
		}
		//__HOSTDEVICE__ int & V(int i){return as_array()[i];};
		__HOSTDEVICE__ int & V(int i){
			switch(i){
				case 2: return depth;
				case 1: return height;
				default: return width;
			};
			//return as_array()[i];
		}
		//__HOSTDEVICE__ const int & V(int i)const{return as_array()[i];};
		__HOSTDEVICE__ const int & V(int i)const{
			return const_cast<intn<3>&>(*this).V(i);
		};
	};
}

template<int n> struct intn;
// final
template<int n> struct intn : public special::intn<n>{
	typedef special::intn<n> parent;
	typedef typename parent::array_type array_type;
	typedef ::intn<(n>1)? n-1: 1> decrement_n_type;
public:
	//using parent::parent;
	//inherit_constructors(intn, parent)
	using parent::V;
	using parent::as_array;
#ifdef __CUDA_ARCH__
	__HOSTDEVICE__ intn() = default; //uninitialized
#else
	__HOST__ intn(){ //0-initialized
		for (int i = 0; i < n; ++i)V(i) = 0;
	};
#endif
	//! default copy ctr
	__HOSTDEVICE__ intn(const intn<n> & b) = default;
	//! construct from initializer list, e.g. {1,2,3}
	/*!
	 * example: intn<3>{12, 2, 3};
	 */
	template<typename type2, class = typename std::enable_if<std::is_integral<type2>::value>::type>
	__HOSTDEVICE__ intn(std::initializer_list<type2> list){
		//static_assert(list.size() == n, "size missmatch");
		auto a = list.begin();
		for(int i=0; i<n; ++i){
			if(i < int(list.size())){
				V(i) = (int)*a;
			}else{
				V(i) = 0;
			};
			++a;
		};
	}
	//! construct from variadic list -- allows implicit element type conversion (e.g .sfrom short, unsigned, etc.)
	/*!
	 * example: intn<3>(12, 2, 3);
	 */
	template<typename A0, typename... Args, class = typename std::enable_if<std::is_integral<A0>::value>::type>
	__HOSTDEVICE__ intn(A0 a0, Args... args) : intn(std::initializer_list<int>( {int(a0), int(args)...} )){
		static_assert(sizeof...(Args)==n-1,"size missmatch"); // check number of arguments is matching
	}

	//! constructor from other array type, can be C array, std::vector, etc.
	template<class other, class = typename std::enable_if<std::is_class<other>::value>::type>
	explicit intn(const other & x){
		for(int i=0; i< n; ++i){
			(*this)[i] = x[i];
		};
	}

public://__________initializers
	__HOSTDEVICE__ intn<n> & operator = (const array_type & x){
		for (int i = 0; i < n; ++i)V(i) = x[i];
		return *this;
	}
	//! default operator =
	__HOSTDEVICE__ intn<n> & operator = (const intn<n> & x) = default;

	//! copy constructor
	__HOSTDEVICE__ intn(const array_type & x){
		(*this) = x;
	}
	//! x = value
	__HOSTDEVICE__ intn<n> & operator = (const int val){
		for (int i = 0; i < n; ++i)V(i) = val;
		return *this;
	}

	//! fill(value)
	__HOSTDEVICE__ void fill(const int val){
		(*this) = val;
	}

	//! convert to C++ array
	explicit __HOSTDEVICE__ operator array_type &(){ return as_array(); };

	//! create increasing sequence
	static __HOSTDEVICE__ intn<n> enumerate(){
		intn<n> r;
		for (int i = 0; i < n; ++i)r[i] = i;
		return r;
	}
public: //________________ element access
	__HOSTDEVICE__ int * begin(){ return &V(0); };
	__HOSTDEVICE__ const int * begin()const { return &V(0); };
	__HOSTDEVICE__ int * ptr(){ return &V(0); };
	__HOSTDEVICE__ const int * ptr()const { return &V(0); };
	__HOSTDEVICE__ int * end(){ return begin() + n; };
	__HOSTDEVICE__ int const * end()const { return begin() + n; };
	__HOSTDEVICE__ int & operator[](const int i){ return V(i); };
	__HOSTDEVICE__ const int & operator[](const int i) const { return V(i); };

public:
	__HOSTDEVICE__ long long prod() const {
		long long r = 1;
		for (int i = 0; i < n; ++i) r *= V(i);
		return r;
	}
	__HOSTDEVICE__ long long numel() const {
		return prod();
	}
public: // mapping between multiindex and linear index
	//! index_to_integer is a 1-to-1 mapping of a multiindex in the octant [0 *this] to an integer in the interval [0 prod()-1]
	/*!
	 * The function integer_to_index defined on the range [0 prod()-1] is the inverse map
	 * The purpose of these functions is to allow to compress some dimensions into e.g. a single GridDim
	 * index_to_integer matches the linear memory arrangement with ascending strides irrespective of the ndarray_ref strides
	 * dim1 is a threshold if the index / size is interpreted as shorter that n
	 */
	__HOSTDEVICE__ int index_to_integer(const intn<n> & ii, int dim1 = n) const{
		int r = ii[dim1-1];
		for(int d = n-2; d >= 0; --d){
			if(d > dim1-2) continue;
			r = r * (*this)[d] + ii[d];
		};
		return r;
	}
	//! integer_to_index is a 1-to-1 mapping of a multiindex in the octant [0 *this] to an integer in the interval [0 prod()-1]
	__HOSTDEVICE__ intn<n> integer_to_index(int i, int dim1 = n) const{
		intn<n> ii;
		for(int d = 0; d < n-1; ++d){
			if(d < dim1){
				int a = i / (*this)[d]; // quotient
				int b = i % (*this)[d]; // reminder
				ii[d] = b;
				i = a;
			};
		};
		ii[n-1] = i;
		return ii;
	}
public: // min max and sorting
	//! max element
	__HOSTDEVICE__ int max() const {
		static_assert(n>0,"bad");
		int val = V(0);
		for(int i = 1; i<n; ++i){
			if(V(i) > val) val = V(i);
		};
		return val;
	}
	//! max element index
	__HOSTDEVICE__ int max_idx() const {
		static_assert(n>0,"bad");
		int val = V(0);
		int idx = 0;
		for(int i = 1; i<n; ++i){
			if(V(i) > val){
				val = V(i);
				idx = i;
			};
		};
		return idx;
	}
	//! min element index
	__HOSTDEVICE__ int min_idx() const {
		static_assert(n>0,"bad");
		int val = V(0);
		int idx = 0;
		for(int i = 1; i<n; ++i){
			if(V(i) < val){
				val = V(i);
				idx = i;
			};
		};
		return idx;
	}

	//! min element index
	__HOSTDEVICE__ int min_abs_idx() const {
		static_assert(n>0,"bad");
		int val = abs(V(0));
		int idx = 0;
		for(int i = 1; i<n; ++i){
			if(abs(V(i)) < val){
				val = abs(V(i));
				idx = i;
			};
		};
		return idx;
	}

	//! sort in ascending order
	intn<n> sort() const {
		intn<n> x = *this;
#ifndef  __CUDA_ARCH__
		std::stable_sort(x.begin(), x.end());
#endif
		return x;
	}

	//! sorting indicies
	intn<n> sort_idx()const{
		intn<n> idx;
#ifndef  __CUDA_ARCH__
		for (int d = 0; d < n; ++d){idx[d] = d;};
		std::stable_sort(idx.begin(), idx.end(), [&](const int & i, const int & j){return (*this)[i] < (*this)[j]; });
#else
		idx = 0; //shut up the warning
#endif
		return idx;
	}

public: //__________cut and permute
	//! reverse the order of elements
	__HOSTDEVICE__  intn<n> rev() const{
		intn<n> r;
		for (int i = 0; i < n; ++i){
			r[i] = (*this)[n - i - 1];
		};
		return r;
	}

	//! append element in the end
	__HOSTDEVICE__ intn<n+1> cat(int a) const{
		intn<n+1> r;
		for (int i = 0; i < n; ++i){
			r[i] = (*this)[i];
		};
		r[n] = a;
		return r;
	}

	//! insert element at j
	template<int j>
	__HOSTDEVICE__ intn<n+1> insert(int a) const{
		intn<n+1> r;
		for (int i = 0; i < n; ++i){
			int i1 = (i<j)? i : i+1;
			r[i1] = (*this)[i];
		};
		r[j] = a;
		return r;
	}

	//! remove element at j
	template<int j>
	__HOSTDEVICE__ decrement_n_type erase() const{
		decrement_n_type r;
		for (int i = 0; i < n; ++i){
			if(i<j || n==1){
				r[i] = (*this)[i];
			}else if(i>j){
				r[i-1] = (*this)[i];
			};
		};
		return r;
	}

	//! remove element at j
	__HOSTDEVICE__ decrement_n_type erase(int j) const{
		//static_assert(n > 1,"bad");
		runtime_check(j >=0 && j< n);
		decrement_n_type r;
		for (int i = 0; i < n; ++i){
			if(i<j || n==1){
				r[i] = (*this)[i];
			}else if(i>j){
				r[i-1] = (*this)[i];
			};
		};
		return r;
	}

public: // _________l-value math operators
	//! a+=b
	__HOSTDEVICE__ intn<n> & operator += (const intn<n> & x){
		for (int i = 0; i < n; ++i){
			(*this)[i] += x[i];
		};
		return *this;
	}
	//! a-=b
	__HOSTDEVICE__ intn<n> & operator -= (const intn<n> & x){
		for (int i = 0; i < n; ++i){
			(*this)[i] -= x[i];
		};
		return *this;
	}
	//! a*=b
	__HOSTDEVICE__ intn<n> & operator *= (const int val){
		for (int i = 0; i < n; ++i){
			(*this)[i] *= val;
		};
		return *this;
	}
	//! a/=val
	__HOSTDEVICE__ intn<n> & operator /= (const int val){
		for (int i = 0; i < n; ++i){
			(*this)[i] /= val;
		};
		return *this;
	}
public:// const math operators
	//! a*b
	__HOSTDEVICE__ intn<n> operator * (const int val) const {
		return (intn<n>(*this)) *= val;
	}
	//! a/val
	__HOSTDEVICE__ intn<n> operator / (const int val)const {
		return (intn<n>(*this)) /= val;
	}
	//! a+b
	__HOSTDEVICE__ intn<n> & operator + (const intn<n> & x) const{
		return (intn<n>(*this)) += x;
	}
	//! a-b
	__HOSTDEVICE__ intn<n> & operator - (const intn<n> & x) const{
		return (intn<n>(*this)) += x;
	}

	//! multiplication by floating point with rounding
	template<typename type2, class = typename std::enable_if<std::is_floating_point<type2>::value>::type >
	__HOSTDEVICE__ intn<n> & operator *= (const type2 & factor) {
		for (int i = 0; i < n; ++i){
			(*this)[i] = int((*this)[i] * factor + 0.5);
		};
		return *this;
	}

	//! multiplication by floating point with rounding
	template<typename type2, class = typename std::enable_if<std::is_floating_point<type2>::value>::type >
	intn<n> operator * (const type2 & factor) const {
		return (intn<n>(*this)) *= factor;
	}

	//! division by floating point with rounding
	template<typename type2, class = typename std::enable_if<std::is_floating_point<type2>::value>::type >
	intn<n> operator / (const type2 & factor) const{
		double invFactor = 1 / factor; // can be INF
		return (intn<n>(*this)) *= invFactor;
	}

	//! exclusive cumulative product (postfix form is obtained by setting prefix = false)
	/*! example: (2,2,3,2) -> (1,2,4,12) in prefix, and -> (12,6,2,1) in postfix
	 */
	template<bool prefix = true>
	__HOSTDEVICE__ intn<n> prefix_prod_ex() const {
		intn<n> r;
		int a = 1;
		if(prefix){
			for(int i=0;i < n; ++i){
				r[i] = a;
				if(i<n) a*= (*this)[i];
			};
		}else{
			for(int i=n-1;i >= 0; --i){
				r[i] = a;
				if(i>0) a*= (*this)[i];
			};
		};
		return r;
	}

	__HOSTDEVICE__ bool operator == (const intn<n> & x) const {
		for (int i = 0; i < n; ++i){
			if((*this)[i] != x[i]) return false;
		};
		return true;
	}

	__HOSTDEVICE__ bool operator != (const intn<n> & x) const {
		return !(*this == x);
	}

	//! a< b, element-wise
	__HOSTDEVICE__ bool operator < (const intn<n> & x) const {
		for (int i = 0; i < n; ++i){
			if(!((*this)[i] < x[i])) return false;
		};
		return true;
	}
	//! a>=b, element-wise
	__HOSTDEVICE__ bool operator >= (const intn<n> & x) const {
		for (int i = 0; i < n; ++i){
			if(!((*this)[i] >= x[i])) return false;
		};
		return true;
	}
	//! a>=val
	__HOSTDEVICE__ bool operator >= (const int val) const {
		for (int i = 0; i < n; ++i){
			if(!((*this)[i] >= val)) return false;
		};
		return true;
	}
	//! a < val
	__HOSTDEVICE__ bool operator < (const int val) const {
		for (int i = 0; i < n; ++i){
			if(!((*this)[i] < val)) return false;
		};
		return true;
	}
	//! a > val
	__HOSTDEVICE__ bool operator > (const int val) const {
		for (int i = 0; i < n; ++i){
			if(!((*this)[i] > val)) return false;
		};
		return true;
	}
};

//__HOSTDEVICE__ bool operator < (int i, const intn<1> & x){
//	return i < int(x);
//}

//! stream << x, print vector to stream
template<int n, typename tstream>
//template<int n, typename tstream, class = typename std::enable_if< (std::is_convertible<tstream,std::ostream>::value || std::is_convertible<tstream,host_stream>::value) >::type >
//template<int n, typename tstream, class = typename std::enable_if<std::is_convertible<tstream,std::ostream>::value>::type >
tstream & operator << (tstream & ss, intn<n> a){
	ss << "(";
	for(int i=0; i<n; ++i){
		ss << a[i];
		if (i<n) ss <<",";
	};
	ss << ")";
	return ss;
}
//! c = min(a,b), element-wise
template<int n>
__HOSTDEVICE__ intn<n> min(const intn<n> & a, const intn<n> & b){
	intn<n> r;
	for(int i=0;i<n;++i){
		r[i] = a[i]<b[i]? a[i] : b[i];
	};
	return r;
}

