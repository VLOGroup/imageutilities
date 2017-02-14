#pragma once

#include "ndarray_ref.host.h"
#include <tuple>

#define HOSTDEVICE __host__ __device__ __forceinline__

namespace nd{
	// make functor

	template<typename F, F * f, typename... AA> struct functor_b{
	    HOSTDEVICE void operator()(AA&&... aa) const {
	        f(aa...);
	    }
	};

	template<typename F, F & f, typename... AA> struct functor_b1{
	    HOSTDEVICE void operator()(AA&&... aa) const {
	        f(aa...);
	    }
	};

	template<typename F, F * f> struct functor_parse{

	    template<typename... AA>
	    static functor_b<F, f, AA...> make_functor(void (*fp)(AA...) ){
	        return functor_b<F, f, AA...>();
	    }
	};

	template<typename F, F & f> struct functor_parse1{
	    template<typename... AA>
	    static functor_b1<F, f, AA...> make_functor(void (&fp)(AA...) ){
	        return functor_b1<F, f, AA...>();
	    }
	};

	template<typename F, F* f> struct functor : public decltype ( functor_parse<F,f>::make_functor(f) ) {
		typedef decltype ( functor_parse<F,f>::make_functor(f) ) parent;
		using parent::operator ();
	};

	template<typename F, F* f>
	auto make_functor() -> decltype( functor_parse<F,f>::make_functor(f) ){
		return functor_parse<F,f>::make_functor(f);
	};

	template<typename F, F & f>
		auto make_functor1() -> decltype( functor_parse1<F,f>::make_functor(f) ){
			return functor_parse1<F,f>::make_functor(f);
		};

	// helper functions
	template<typename type, int dims> type array_type( const kernel::ndarray_ref<type, dims> &){
		return type();
	};

	template<typename type, int dims> constexpr int array_dims( const kernel::ndarray_ref<type, dims> &){
		return dims;
	};

	template<typename type, int dims> intn<dims> array_tindex( const kernel::ndarray_ref<type, dims> &){
		return intn<dims>();
	};

	template<typename T> struct array_parse{
		typedef decltype(array_type(T())) type;
		typedef decltype(array_tindex(T())) tindex;
		static constexpr int dims = T::my_dims; //array_dims(T());
		//static constexpr int dims = T::my_dims; //array_dims(T());
		typedef kernel::ndarray_ref<type, dims> tkernel;
	};

	//! transform using a free scalar function
	template<int dims, typename T0, void (*f)(T0 &, const T0&)>
		__attribute__((noinline))
		void transform(ndarray_ref<T0,dims> & a, const ndarray_ref<T0,dims>& a1);
	/*
	template<int dims, typename T0, typename... TT, void (*f)(T0 &, const TT&... )>
	__attribute__((noinline))
	void transform(ndarray_ref<T0,dims> & a, const ndarray_ref<TT,dims>&... aa);
	*/
	//void transform( void (*f)(T0 &, const TT&... ), ndarray_ref<T0,dims> & a, const ndarray_ref<TT,dims>&... aa);

/*
	//! transform using a free function
	template<typename A0, typename... AA>
	__attribute__((noinline))
	void transform( void (*f)( typename array_parse<A0>::type &, const typename array_parse<AA>::type&... ), A0 & a, const AA&... aa);
*/

	//! transform using a free function recieving index and full arrays
	template<typename A0, typename... AA>
	__attribute__((noinline))
	void transform( void (*f)( const typename array_parse<A0>::tindex &, typename array_parse<A0>::tkernel &, const typename array_parse<AA>::tkernel&... ), A0 & a, const AA&... aa);

	//! transform using a functor
	template <typename F, typename A0, typename... AA>
	__attribute__((noinline))
	void transform(const F & f, A0 & a, const AA&... b);

	/*
	template<typename A0, typename... AA>
	void test_transform(A0 & a, const AA&... aa){
		const typename array_parse<A0>::tindex & i = a.size();
		typename array_parse<A0>::tkernel & k = a.kernel();
	};
	 */


	/*
	//! transform using a free function recieving 2 array elements
	template<int dims1, typename T1, int dims2, typename T2> __attribute__((noinline))
			void transform( void (*f)( T1&, const T2&), ndarray_ref<T1, dims1> & a, const ndarray_ref<T2, dims2> & b);

	//! transform using a free function recieving 2 array elements
	template<int dims1, typename T1, int dims2, typename T2> __attribute__((noinline))
			void transform( void (*f)( T1&, const T2&), ndarray_ref<T1, dims1> & a, const ndarray_ref<T2, dims2> & b);

	//! transform using an object function recieving 1 array element
	template <int dims1, typename F, typename T1> __attribute__((noinline)) void
			transform(const F & f, ndarray_ref<T1, dims1> & a);

	//! transform using an object function recieving 3 array element
	template <int dims1, typename F, typename T1> __attribute__((noinline)) void
			transform(const F & f, ndarray_ref<T1, dims1> & a, const ndarray_ref<T1, dims1> & b, const ndarray_ref<T1, dims1> & c);

	//! transform using a free function recieving full index and 2 arrays
	template <int dims1, int dims2, typename T1> __attribute__((noinline)) void
			transform( void (*f)(const intn<dims1> & T1&, const kernel::ndarray_ref<T1, dims1> &, const kernel::ndarray_ref<T1, dims2> &), ndarray_ref<T1, dims1> & a, const ndarray_ref<T1, dims2> & b);

	 */
};
