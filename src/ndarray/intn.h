#ifndef intn_h
#define intn_h

#include "defines.h"

#ifndef  __CUDA_ARCH__
#include <algorithm>
#include <error.h>
#endif

template<typename T, typename U> constexpr size_t offsetOf(U T::*member){
	return (char*)&((T*)nullptr->*member) - (char*)nullptr;
}

//________________________________intn______________________________________________
//! specializations of intn<n> for lower dimensions
namespace special{
	//! generic dimensions - array storage
	template<int n> struct intn{
	public:
		typedef int array_type[n];
		array_type v;
	protected://__________
		__host__ __device__ __forceinline__ array_type & as_array(){ return v; }
		__host__ __device__ __forceinline__ const array_type & as_array()const{ return v; }
		__host__ __device__ __forceinline__ int & V(int i){return v[i];};
		__host__ __device__ __forceinline__ const int & V(int i)const{return v[i];};
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
		__host__ __device__ __forceinline__ intn() = default;
		__host__ __device__ __forceinline__ intn(int a0){V(0) = a0;};
		// intn<1> is convertible to int
		__host__ __device__ __forceinline__ operator int & (){return V(0);}
		__host__ __device__ __forceinline__ operator const int & ()const {return V(0);}
	protected://__________
		__host__ __device__ __forceinline__ array_type & as_array(){
			return *(array_type*)&width;
		}
		__host__ __device__ __forceinline__ const array_type & as_array()const{
			return const_cast<intn<1> *>(this)->as_array();
		}
		__host__ __device__ __forceinline__ int & V(int i){return as_array()[i];};
		__host__ __device__ __forceinline__ const int & V(int i)const{return as_array()[i];};
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
	protected://__________
		__host__ __device__ __forceinline__ array_type & as_array(){
			static_assert(offsetof(intn<2>,height)==sizeof(int), "struct linear layout assumption failed");
			return *(array_type*)&width;
		}
		__host__ __device__ __forceinline__ const array_type & as_array()const{
			return const_cast<intn<2> *>(this)->as_array();
		}
		__host__ __device__ __forceinline__ int & V(int i){return as_array()[i];};
		__host__ __device__ __forceinline__ const int & V(int i)const{return as_array()[i];};
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
	protected://__________
		__host__ __device__ __forceinline__ array_type & as_array(){
			static_assert(offsetof(intn<3>,height)==sizeof(int), "struct linear layout assumption failed");
			static_assert(offsetof(intn<3>,depth)==2*sizeof(int), "struct linear layout assumption failed");
			return *(array_type*)&width;
		}
		__host__ __device__ __forceinline__ const array_type & as_array()const{
			return const_cast<intn<3> *>(this)->as_array();
		}
		__host__ __device__ __forceinline__ int & V(int i){return as_array()[i];};
		__host__ __device__ __forceinline__ const int & V(int i)const{return as_array()[i];};
	};
}

// final
template<int n> struct intn : public special::intn<n>{
	typedef special::intn<n> parent;
	typedef typename parent::array_type array_type;
public:
	using parent::parent;
	using parent::V;
	using parent::as_array;
#ifdef __CUDA_ARCH__
	__host__ __device__ __forceinline__ intn() = default; //uninitialized
#else
	__host__ __forceinline__ intn(){ //0-initialized
		for (int i = 0; i < n; ++i)V(i) = 0;
	};
#endif
	//! default copy ctr
	__host__ __device__ __forceinline__ intn(const intn<n> & b) = default;
	//! construct from a list -> improve to a variadic template
	__host__ __device__ __forceinline__ intn(int a0, int a1, int a2 = 0, int a3 = 0){
		V(0) = a0;
		if (n > 1)V(1) = a1;
		if (n > 2)V(n > 2? 2 : 0) = a2;
		if (n > 3)V(n > 3? 3 : 0) = a3;
	}
public://__________initializers
	__host__ __device__ __forceinline__ intn<n> & operator = (const array_type & x){
		for (int i = 0; i < n; ++i)V(i) = x[i];
		return *this;
	}
	//! default operator =
	__host__ __device__ __forceinline__ intn<n> & operator = (const intn<n> & x){
		for (int i = 0; i < n; ++i)V(i) = x[i];
		return *this;
	}
	__host__ __device__ __forceinline__ intn(const array_type & x){
		(*this) = x;
	}
	__host__ __device__ __forceinline__ intn<n> & operator = (const int val){
		for (int i = 0; i < n; ++i)V(i) = val;
		return *this;
	}
	//! convert to C++ array
	explicit __host__ __device__ __forceinline__ operator array_type &(){ return as_array(); };
public: //________________ element access
	__host__ __device__ __forceinline__ int * begin(){ return &V(0); };
	__host__ __device__ __forceinline__ const int * begin()const { return &V(0); };
	__host__ __device__ __forceinline__ int * end(){ return begin() + n; };
	__host__ __device__ __forceinline__ int const * end()const { return begin() + n; };
	__host__ __device__ __forceinline__ int & operator[](const int i){ return V(i); };
	__host__ __device__ __forceinline__ const int & operator[](const int i) const { return V(i); };

public:
	__host__ __device__ __forceinline__ long long prod() const {
		long long r = 1;
		for (int i = 0; i < n; ++i) r *= V(i);
		return r;
	}
public:
	//! max element
	__host__ __device__ __forceinline__ int max() const {
		static_assert(n>0,"bad");
		int val = V(0);
		for(int i = 1; i<n; ++i){
			if(V(i) > val) val = V(i);
		};
		return val;
	}
	//! max element index
	__host__ __device__ __forceinline__ int max_idx() const {
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
	__host__ __device__ __forceinline__ int min_idx() const {
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
	//! remove element at j
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

	intn<n> operator * (const double factor) const {
		intn<n> r;
		for (int i = 0; i < n; ++i){
			(*this)[i] = int((*this)[i] * factor + 0.5);
		};
		return r;
	}

	intn<n> operator / (const double factor) const{
		double invFactor = 1 / factor; // can be INF
		return (*this) *= invFactor;
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

	__host__ __device__ __forceinline__ bool operator != (const intn<n> & x) const {
		return !(*this == x);
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
	__host__ __device__ __forceinline__ bool operator >= (const int val) const {
		for (int i = 0; i < n; ++i){
			if(!((*this)[i] >= val)) return false;
		};
		return true;
	}
	__host__ __device__ __forceinline__ bool operator < (const int val) const {
		for (int i = 0; i < n; ++i){
			if(!((*this)[i] < val)) return false;
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
