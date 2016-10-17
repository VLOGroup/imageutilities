#pragma once

#include "intn.h"
#include "transform.h"
#include <thrust/tuple.h>

#include "ndarray/error_cuda.h"

namespace device_op1{
	inline constexpr int divup(int x, int y){
		return (x-1)/(y)+1;
	}

	inline constexpr int roundup(int x, int y){
		return divup(x,y)*y;
	}

	// declaration of device operations
	template<int dims> struct for_each_dims{
	public:
		template<typename Func>
		static inline void apply(const intn<dims> & r, Func & func); // implementation through a kernel launch in tarry_op.cu
	};

	template<int dims, typename Func> inline void for_each(const intn<dims> & r, Func & func){
		for_each_dims<dims>::apply(r,func);
	}


	//! 1D kernel
	template< typename Func>
	__global__ void //__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
	for_each_1D_kernel(intn<1> size, Func func){
		int i = threadIdx.x + blockDim.x * blockIdx.x;
		if(i < size){
			func(i);
		};
	}

	//! 1D launch
	template<>
	template<typename Func>
	void for_each_dims<1>::apply(const intn<1> & size, Func & func){
		dim3 dimBlock(32 * 32, 1, 1);
		dim3 dimGrid(divup(size[0], dimBlock.x), 1, 1);
		for_each_1D_kernel <<< dimGrid, dimBlock >>>(size, func);
		cudaDeviceSynchronize();
		cuda_check_error();
	}


	//! 2D kernel
	template< typename Func>
	__global__ void //__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
	for_each_2D_kernel(intn<2> size, Func func){
		intn<2> ii = intn<2>(threadIdx.x + blockDim.x * blockIdx.x, threadIdx.y + blockDim.y * blockIdx.y);
		if(ii < size){
			func(ii);
		};
	}

	//! 2D launch
	template<>
	template<typename Func>
	void for_each_dims<2>::apply(const intn<2> & size, Func & func){
		dim3 dimBlock(32, 32, 1);
		dim3 dimGrid(divup(size[0], dimBlock.x), divup(size[1], dimBlock.y), 1);
		for_each_2D_kernel <<< dimGrid, dimBlock >>>(size, func);
		cudaDeviceSynchronize();
		cuda_check_error();
	}

/*
	template <typename Ts, int k> struct param_tuple;
	template <typename Ts> struct param_tuple<Ts,0>{
		typedef decltype(   thrust::make_tuple(std::get<0>(Ts()).kernel())  ) type;
		static type apply(Ts & t){
			return thrust::make_tuple(std::get<0>(t).kernel());
		};
	};
	template <typename Ts, int k = std::tuple_size<Ts>::value -1 > struct param_tuple{
		typedef typename param_tuple<Ts, k-1>::type subtype;
		typedef decltype(   thrust::tuple_cat(subtype(), thrust::make_tuple(std::get<k>(Ts()).kernel()) )   ) type;
		static type apply(Ts & t){
			return thrust::tuple_cat(param_tuple<Ts, k-1>(t), thrust::make_tuple(std::get<k>(t).kernel()) );
		};
	};
*/

	/*
	template < uint N >
	struct apply_funcV{
		template < typename F, typename... ArgsT, typename... Args >
		__host__ __device__ static void applyTuple( const F & f, const std::tuple<ArgsT...>& t, Args... args ){
			apply_func<N-1>::applyTuple( f, t, std::get<N-1>(t), args... );
		}
	};

	template <> struct apply_func<0>{
		template < typename F, typename... ArgsT, typename... Args >
		__host__ __device__ static void applyTuple(const F & f, const std::tuple<ArgsT...>& ,  Args... args ){
			f( args... );
		}
	};
*/

	template<typename... Args> thrust::tuple<Args...> convert_tuple(const std::tuple<Args...> & t){
		constexpr int N = std::tuple_size<std::tuple<Args...> >::value;
		typedef thrust::tuple<Args...> R;
		return host_detail::apply_func<N,R>::applyTuple(thrust::make_tuple<Args...>, t);
	}

	template < uint N >
	struct apply_func_thrust{
		template < typename F, typename... ArgsT, typename... Args >
		__host__ __device__ static __forceinline__ void applyTuple( const F & f, const thrust::tuple<ArgsT...>& t, Args... args ){
			apply_func_thrust<N-1>::applyTuple( f, t, thrust::get<N-1>(t), args... );
		}
	};

	template <> struct apply_func_thrust<0>{
		template < typename F, typename... ArgsT, typename... Args >
		__host__ __device__ static __forceinline__ void applyTuple(const F & f, const thrust::tuple<ArgsT...>& /* t */,  Args... args ){
			f( args... );
		}
	};

	/*
	template < uint N, typename R = void >
	struct tuple_for_each_thrust{
		template <typename F, typename... ArgsT, typename... Args >
		__host__ __device__ static R applyTuple( const F & f, const std::tuple<ArgsT...>& t, Args... args ){
			auto a = f(thrust::get<N-1>(t));
			return tuple_for_each_thrust<N-1,R>::applyTuple( f, t, a, args... );
		}
	};

	template <typename R> struct tuple_for_each_thrust<0, R>{
		template <typename F, typename... ArgsT, typename... Args >
		__host__ __device__ static R applyTuple(const F & f, const std::tuple<ArgsT...>& ,  Args... args ){
			return thrust::make_tuple( args... );
		}
	};
*/

	template < uint N, typename R = void >
	struct ptr_tuple_thrust{
		template <typename F, typename... ArgsT, typename... Args >
		__host__ __device__ __forceinline__ static R applyTuple( const F & f, const thrust::tuple<ArgsT...>& t, Args... args ){
			auto a = thrust::get<N-1>(t).kernel().ptr(f);
			return ptr_tuple_thrust<N-1,R>::applyTuple( f, t, a, args... );
		}
	};

	template <typename R> struct ptr_tuple_thrust<0, R>{
		template <typename F, typename... ArgsT, typename... Args >
		__host__ __device__ __forceinline__ static R applyTuple(const F & f, const thrust::tuple<ArgsT...>& /* t */,  Args... args ){
			return thrust::make_tuple( args... );
		}
	};

	#pragma hd_warning_disable
	template<int dims, class Ts, typename F> struct bind_functor_args{
		static constexpr int N = std::tuple_size<Ts>::value;
		typedef intn<dims> tindex;
		typedef decltype(  convert_tuple(Ts() )  ) params_type;
		typedef typename transform_tuple<dims, Ts>::ptr_tuple<tindex> ptr_tuple; // element accessor
		typedef decltype(   ptr_tuple::apply(Ts(), tindex() )   ) ptr_tuple_ret_t; // std tuple of pointers
		typedef decltype(   convert_tuple(ptr_tuple_ret_t())   ) ptr_tuple_ret_t_thrust; // thrust tuple of pointers
		//
		params_type params;
		F f;
		bind_functor_args(const Ts & t, const F & _f):f(_f){
			params = convert_tuple(t);
		};
		__device__ __forceinline__ void operator () (const tindex & ii){
			auto pp = ptr_tuple_thrust<N, ptr_tuple_ret_t_thrust>::applyTuple(ii, params);
			apply_func_thrust<N>::applyTuple(f,pp);
		}
	};


	/*
	// export instances of this function
	template<int dims, class Ts, typename F> // tuple
	void apply_to_tuple(intn<dims> size, Ts & t, F f){
		constexpr int N = std::tuple_size<Ts>::value;
		typedef intn<dims> tindex;
		// convert std::tuple to thrust::tuple
		auto t_t = convert_tuple(t);
		typedef typename transform_tuple<dims, Ts>::ptr_tuple<tindex> ptr_tuple; // element accessor
		typedef decltype(   ptr_tuple::apply(t, tindex() )   ) ptr_tuple_ret_t; // std tuple of pointers
		typedef decltype(   convert_tuple(ptr_tuple_ret_t())   ) ptr_tuple_ret_t_thrust; // thrust tuple of pointers

		auto func = [=] __device__ (const tindex & ii){
			//auto pp = ptr_tuple::apply(t_t, ii); // evaluate all pointers at index ii
			auto pp = ptr_tuple_thrust<N, ptr_tuple_ret_t_thrust>::applyTuple(ii, t_t);
			apply_func_thrust<N>::applyTuple(f,pp);
		};

		for_each(size, func );
	}
	*/

	/*
	template<int dims, typename F, class Args...> // tuple unpacked
	void apply_to_tuple_1(intn<dims> size, F f, Args... & args){
		auto t_t = thrust::make_tuple(args...);
		constexpr int N = std::tuple_size<Ts>::value;
		typedef intn<dims> tindex;
		tindex ii;

		auto t_t = apply_func<N>::applyTuple(thrust::make_tuple<Ts::>, t); // thrust tuple
		typedef typename transform_tuple<dims,decltype(t_t)>::ptr_tuple<tindex> ptr_tuple;

		auto func = [=] __device__ (const tindex & ii){
			auto pp = ptr_tuple::apply(t_t, ii); // evaluate all pointers at index ii
			apply_func<N>::applyTuple(f,pp);
		};

		for_each(size, func );
	}
	*/
};
