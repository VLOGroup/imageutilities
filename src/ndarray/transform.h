#pragma once

#include "ndarray_ref.host.h"
#include <tuple>

namespace nd{

	// helper functions
	template<typename type, int dims> type array_type( const kernel::ndarray_ref<type, dims> &){
		return type();
	};

	template<typename type, int dims> constexpr int array_dims(const kernel::ndarray_ref<type, dims> &){
		return dims;
	};

	template<typename type, int dims> intn<dims> array_tindex( const kernel::ndarray_ref<type, dims> &){
		return intn<dims>();
	};

	template<typename T> struct array_parse{
		typedef decltype(array_type(T())) type;
		typedef decltype(array_tindex(T())) tindex;
		static constexpr int dims = T::my_dims;
		typedef kernel::ndarray_ref<type, dims> tkernel;
	};


	//! transform using a functor
	template<typename F, typename A0, typename... AA>
	__attribute__((noinline))
	void transform(const F & f, A0 & a, const AA&... aa);


};
