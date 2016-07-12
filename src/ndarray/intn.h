#ifndef intn_h
#define intn_h

#include "defines.h"

#ifndef  __CUDA_ARCH__
#include <algorithm>
#endif

//________________________________intn______________________________________________
namespace base{
	//! struct with array int[n] and some operations on it
	template<int n> struct intn{
	public:
		typedef int array_type[n];
		array_type v;
	public://__________constructors
		__host__ __device__ __forceinline__ intn() = default; //uninitialized
		//! default copy ctr
		__host__ __device__ __forceinline__ intn(const intn<n> & b) = default;
		//! construct from a list -> improve to a variadic template
		__host__ __device__ __forceinline__ intn(int a0, int a1 = 0, int a2 = 0, int a3 = 0){
			v[0] = a0;
			if (n > 1)v[1] = a1;
			if (n > 2)v[n > 2? 2 : 0] = a2;
			if (n > 3)v[n > 3? 3 : 0] = a3;
		}
		__host__ __device__ __forceinline__ intn<n> & operator = (const intn<n> & x){
			(*this) = x.v;
			return *this;
		}
	};
}

//specializations

namespace special{
	template<int n> struct intn : public base::intn<n>{
		typedef base::intn<n> parent;
	public:
		using parent::parent;
		using parent::operator =;
		__host__ __device__ __forceinline__ intn() = default;
	};

	template<> struct intn<1> : public base::intn<1>{
		typedef base::intn<1> parent;
	public:
		using parent::parent;
		using parent::operator =;
		__host__ __device__ __forceinline__ intn() = default;
		//__host__ __device__ __forceinline__ operator int ()const {return v[0];}
		__host__ __device__ __forceinline__ operator int & (){return v[0];}
		__host__ __device__ __forceinline__ operator const int & ()const {return v[0];}
	};
}

// final
template<int n> struct intn : public special::intn<n>{
	typedef special::intn<n> parent;
	typedef typename parent::array_type array_type;
public:
	using parent::parent;
	using parent::v;
	__host__ __device__ __forceinline__ intn() = default;

public://__________initializers
	__host__ __device__ __forceinline__ intn<n> & operator = (const array_type & x){
		for (int i = 0; i < n; ++i)v[i] = x[i];
		return *this;
	}
	//! default operator =
	__host__ __device__ __forceinline__ intn<n> & operator = (const intn<n> & x){
		(*this) = x.v;
		return *this;
	}
	__host__ __device__ __forceinline__ intn(const array_type & x){
		(*this) = x;
	}
	__host__ __device__ __forceinline__ intn<n> & operator = (const int val){
		for (int i = 0; i < n; ++i)v[i] = val;
		return *this;
	}
	//! convert to C++ array
	explicit __host__ __device__ __forceinline__ operator array_type &(){ return v; };
public: //________________ element access
	__host__ __device__ __forceinline__ int * begin(){ return v; };
	__host__ __device__ __forceinline__ const int * begin()const { return v; };
	__host__ __device__ __forceinline__ int * end(){ return begin() + n; };
	__host__ __device__ __forceinline__ int const * end()const { return begin() + n; };
	__host__ __device__ __forceinline__ int & operator[](const int i){ return v[i]; };
	__host__ __device__ __forceinline__ const int & operator[](const int i) const { return v[i]; };

public:
	__host__ __device__ __forceinline__ long long prod() const {
		long long r = 1;
		for (int i = 0; i < n; ++i) r *= v[i];
		return r;
	}
public:
	//! max element
	__host__ __device__ __forceinline__ int max() const {
		static_assert(n>0,"bad");
		int val = v[0];
		for(int i = 1; i<n; ++i){
			if(v[i] > val) val = v[i];
		};
		return val;
	}
	//! max element index
	__host__ __device__ __forceinline__ int max_idx() const {
		static_assert(n>0,"bad");
		int val = v[0];
		int idx = 0;
		for(int i = 1; i<n; ++i){
			if(v[i] > val){
				val = v[i];
				idx = i;
			};
		};
		return idx;
	}
	//! min element index
	__host__ __device__ __forceinline__ int min_idx() const {
		static_assert(n>0,"bad");
		int val = v[0];
		int idx = 0;
		for(int i = 1; i<n; ++i){
			if(v[i] < val){
				val = v[i];
				idx = i;
			};
		};
		return idx;
	}
	//! sort in ascending order
	__host__ intn<n> sort() const {
		intn<n> x = *this;
		std::sort(x.begin(), x.end());
		return x;
	};
	//! sorting indicies
	__host__ intn<n> sort_idx()const{
		intn<n> idx;
		for (int d = 0; d < n; ++d){idx[d] = d;};
		std::sort(idx.begin(), idx.end(), [&](const int & i, const int & j){return (*this)[i] < (*this)[j]; });
		return idx;
	}
public: //__________cut and permute
	//! reverse the order of elements
	__host__ __device__ __forceinline__  intn<n> rev() const{
		intn<n> r;
		for (int i = 0; i < n; ++i){
			r[i] = (*this)[n - i - 1];
		};
		return r;
	}
	//! append element in the end
	__host__ __device__ __forceinline__ intn<n+1> cat(int a) const{
		intn<n+1> r;
		for (int i = 0; i < n; ++i){
			r[i] = (*this)[i];
		};
		r[n] = a;
		return r;
	}
	//! insert element at j
	template<int j>
	__host__ __device__ __forceinline__ intn<n+1> insert(int a) const{
		intn<n+1> r;
		for (int i = 0; i < n; ++i){
			int i1 = (i<j)? i : i+1;
			r[i1] = (*this)[i];
		};
		r[j] = a;
		return r;
	}
	//! remove element at i
	template<int j>
	__host__ __device__ __forceinline__ intn<n - 1> erase() const{
		static_assert(n > 1,"bad");
		static_assert(j >=0 && j< n,"bad");
		intn<n - 1> r;
		for (int i = 0; i < n; ++i){
			if (i!=j) r[i - (i>j) ] = (*this)[i];
		};
		return r;
	}
public: // _________l-value math operators
	__host__ __device__ __forceinline__ intn<n> & operator += (const intn<n> & x){
		for (int i = 0; i < n; ++i){
			(*this)[i] += x[i];
		};
		return *this;
	}
	__host__ __device__ __forceinline__ intn<n> & operator *= (const int val){
		for (int i = 0; i < n; ++i){
			(*this)[i] *= val;
		};
		return *this;
	}
	__host__ __device__ __forceinline__ intn<n> & operator /= (const int val){
		for (int i = 0; i < n; ++i){
			(*this)[i] /= val;
		};
		return *this;
	}
public:// const math operators
	__host__ __device__ __forceinline__ intn<n> operator * (const int val) const {
		return (intn<n>(*this)) *= val;
	}

	__host__ __device__ __forceinline__ intn<n> operator / (const int val)const {
		return (intn<n>(*this)) /= val;
	}

	__host__ __device__ __forceinline__ intn<n> & operator + (const intn<n> & x) const{
		return (intn<n>(*this)) += x;
	}

	//! exclusive cumulative product (postfix form is obtained by setting prefix = false)
	/*! For example: (2,2,3,2) -> (1,2,4,12) in prefix, and -> (12,6,2,1) in postfix
	 */
	template<bool prefix = true>
	__host__ __device__ __forceinline__ intn<n> prefix_prod_ex() const {
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

	__host__ __device__ __forceinline__ bool operator == (const intn<n> & x) const {
		for (int i = 0; i < n; ++i){
			if((*this)[i] != x[i]) return false;
		};
		return true;
	}
	__host__ __device__ __forceinline__ bool operator < (const intn<n> & x) const {
		for (int i = 0; i < n; ++i){
			if(!((*this)[i] < x[i])) return false;
		};
		return true;
	}
	__host__ __device__ __forceinline__ bool operator >= (const intn<n> & x) const {
		for (int i = 0; i < n; ++i){
			if(!((*this)[i] >= x[i])) return false;
		};
		return true;
	}
};

//__host__ __device__ __forceinline__ bool operator < (int i, const intn<1> & x){
//	return i < int(x);
//}

template<int n, typename tstream>
__host__ __device__ __forceinline__ tstream & operator << (tstream & ss, intn<n> a){
	ss << "(";
	for(int i=0; i<n; ++i){
		ss << a[i];
		if (i<n) ss <<",";
	};
	ss << ")";
	return ss;
}


#endif
