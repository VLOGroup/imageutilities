#pragma once

#include <cuda.h>
#include "transform.h"

#include <tuple>
#include "type_name.h"
#include "ndarray/error_cuda.h"

// based on http://stackoverflow.com/a/17426611/410767 by Xeo
namespace std  // WARNING: at own risk, otherwise use own namespace
{
	template <size_t... Ints>
	struct index_sequence
	{
		using type = index_sequence;
		using value_type = size_t;
		static constexpr std::size_t size() { return sizeof...(Ints); }
	};

	// --------------------------------------------------------------

	template <class Sequence1, class Sequence2>
	struct _merge_and_renumber;

	template <size_t... I1, size_t... I2>
	struct _merge_and_renumber<index_sequence<I1...>, index_sequence<I2...>>
	: index_sequence<I1..., (sizeof...(I1)+I2)...>
	{ };

	// --------------------------------------------------------------

	template <size_t N>
	struct make_index_sequence
			: _merge_and_renumber<typename make_index_sequence<N/2>::type,
			  typename make_index_sequence<N - N/2>::type>
			{ };

			template<> struct make_index_sequence<0> : index_sequence<> { };
			template<> struct make_index_sequence<1> : index_sequence<0> { };
}

namespace nd{
//template<typename A0, typename... AA> std::tuple<AA...> tuple_tail_type(const std::tuple<A0, AA...> & t){
//		return std::tuple<AA...>();
//	};

	//--------------- functor kernel wrap-----------------------------------
	//--------------- little __device__ tuple-------------------------------
	template<typename A0, typename... AA> struct tuple{
	    A0 a;
	    tuple<AA...> tail;
	    tuple() = default;
	    tuple(A0 _a, AA... aa):a(_a), tail(aa...){};
	    //tuple(const std::tuple<A0, AA...> & t):a(std::get<0>(t)), tail(tuple_tail(t) ){};
	};

	template<typename A0> struct tuple<A0>{
	    A0 a;
	    tuple(A0 _a):a(_a){};
	    //tuple(const std::tuple<A0> & t):a(std::get<0>(t)){};
	};

	/*
	template<typename A0, typename... AA> tuple<A0, AA...> make_tuple(A0 a0, AA... aa){
	    return tuple<A0, AA...>(a0, aa...);
	}
	*/

	/*
	template<typename... AA, std::size_t... II> tuple<typename std::decay<AA>::type...> make_tuple(std::tuple<AA...> & t, std::index_sequence<II...> ii){
		return tuple<typename std::decay<AA>::type...>(std::get<II>(t)...);
	}
	*/

	/*
	template<typename... AA> tuple<AA...> make_tuple(std::tuple<AA...> & t){
		return make_tuple(t, touple_index(t));
	}
	*/

	template<typename F, typename A0, typename... AA, typename... BB>
	auto __HOSTDEVICE__ tuple_call(const F & f, tuple<A0, AA...> & p, BB&... bb) -> decltype( tuple_call(f, p.tail, bb..., p.a) ){
		return tuple_call(f, p.tail, bb..., p.a);
	}

	template<typename F, typename A0, typename... BB>
	auto __HOSTDEVICE__ tuple_call(const F & f, tuple<A0> & p, BB&... bb) -> decltype( f(bb..., p.a) ){
	    return f(bb..., p.a);
	}

	template<typename F, typename M, typename A0, typename... AA, typename... BB>
	__HOSTDEVICE__ void tuple_call_m(const F & f, const M & m, const tuple<A0, AA...> & p, BB&... bb){
		tuple_call_m(f, m, p.tail, bb..., m(p.a));
	}

	template<typename F, typename M, typename A0, typename... BB>
	__HOSTDEVICE__ void tuple_call_m(const F & f, const M & m, const tuple<A0> & p, BB&... bb ){
	    f(bb..., m(p.a));
	}


	//---------- scalar functor kernel wrap -----------------------------------
	template<int dims> struct index_accessor{
		intn<dims> ii;
		template<typename A>
		__HOSTDEVICE__ typename array_parse<A>::type & operator ()(const A & a) const {
			return *a.ptr(ii);
		}
	};

	template<int dims, typename F, bool indexed, typename... AA > struct functor_bind{
		F f; // functor
		tuple< typename std::decay<decltype( AA().kernel() )>::type... > tt; // tuple of kernels
		functor_bind(const F & _f, AA... aa):f(_f), tt(aa.kernel()...){
		}
		__HOSTDEVICE__ const intn<dims> & size()const {
			return tt.a.size();
		}
		void __HOSTDEVICE__ operator ()(const intn<dims>  & ii)const{
			index_accessor<dims> m{ii};
			tuple_call_m(f, m, tt);
		}
	};


	//---------- indexed functor kernel wrap -----------------------------------
	template<int dims, typename F, typename... AA> struct functor_bind<dims,F, true, AA...>{
		F f; // functor
		tuple< typename std::decay<decltype( AA().kernel() )>::type... > tt; // tuple of kernels
	public:
		functor_bind(const F & _f, AA... aa):f(_f), tt(aa.kernel()...){
		}
		__HOSTDEVICE__ const intn<dims> & size()const {
			return tt.a.size();
		}
		void __HOSTDEVICE__ operator ()(const intn<dims>  & ii){
			tuple_call(f, tt, ii);
		}
	};


	template<typename... AA>
	auto tuple_index(const std::tuple<AA...> &) -> decltype( std::make_index_sequence<sizeof...(AA)>() ){
		return std::make_index_sequence<sizeof...(AA)>();
	}

	template<int dims, typename F, bool f_indexed, typename... AA, std::size_t... II>
	functor_bind<dims, F, f_indexed, AA...>
	tuple_expand_and_bind(const F & _f, const std::tuple<AA...> & t, std::index_sequence<II...> ii){
		return functor_bind<dims, F, f_indexed, AA...>(_f, std::get<II>(t)... );
	}


	/*
	template<int dims, typename F, typename... AA, bool f_indexed>
	functor_bind<dims, F, AA..., f_indexed>
	tuple_expand_and_bind(const F & _f, const std::tuple<AA...> & t){
		auto ind = std::make_index_sequence<sizeof...(AA)>();
		return tuple_expand_and_bind(_f, t, ind);
	}
	*/

	//------------------------------------------------------------------------

	namespace device_op1{
		inline constexpr int divup(int x, int y){
			return (x-1)/(y)+1;
		}

		inline constexpr int roundup(int x, int y){
			return divup(x,y)*y;
		}

		// dimensions selector
		template<int dims> struct for_each_dims{
		public:
			template<typename Func>
			static inline void apply(const intn<dims> & r, Func & func);
		};

		template<int dims, typename Func> inline void for_each(const intn<dims> & r, Func & func){
			for_each_dims<dims>::apply(r,func);
		}
		//! 1D kernel
		template< typename Func>
		__global__ void //__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
		for_each_1D_kernel(Func func){
			int i = threadIdx.x + blockDim.x * blockIdx.x;
			if(i < func.size()){
				func(i);
			};
		}

		//! 1D launch
		template<>
		template<typename Func>
		void for_each_dims<1>::apply(const intn<1> & size, Func & func){
			dim3 dimBlock(32 * 32, 1, 1);
			dim3 dimGrid(divup(size[0], dimBlock.x), 1, 1);
			for_each_1D_kernel <<< dimGrid, dimBlock >>>(func);
			cudaDeviceSynchronize();
			cuda_check_error();
		}

		struct layout{
			int r;
			int f_over;
		};

		//! 2D kernel
		template< typename Func>
		__global__ void //__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
		for_each_2D_kernel(Func func, const layout L){
			//intn<2> ii = intn<2>(threadIdx.x + blockDim.x * blockIdx.x, threadIdx.y + blockDim.y * blockIdx.y);
			int x = threadIdx.x + blockDim.x * blockIdx.x;
			int y = threadIdx.y + blockDim.y * blockIdx.y;
			int i = x + L.r * threadIdx.y; // 2D index + correction per column
			int j = y;
			if(i >= L.f_over){ // threads overflow to the next column
				i-= L.f_over;
				++j;
			}
			if(i < func.size()[0] && j < func.size()[1]){
				intn<2> ii(i,j);
				func(ii);
			};
		}

		//! 2D launch
		template<>
		template<typename Func>
		void for_each_dims<2>::apply(const intn<2> & size, Func & func){
			dim3 dimBlock(32, 32, 1);
			// chose layout
			const int WARP_SIZE = 32;
			if(size[0] <  WARP_SIZE){ // small dim0
				// pick BY such that
			}
			int r = (size[0]) % dimBlock.x; // folding over reminder
			// coverage in dimBlock.y
			//int W = dimBlock.x + r; //
			dim3 dimGrid(divup(size[0], dimBlock.x), divup(size[1], dimBlock.y), 1);
			layout L;
			L.r = r;
			for_each_2D_kernel <<< dimGrid, dimBlock >>>(func, L);
			cudaDeviceSynchronize();
			cuda_check_error();
		}


/*
		template<int dims, typename F, typename...AA> void apply_to_tuple(intn<dims> size, const F & f, const std::tuple<AA...> & t){
			for_each_dims<dims>::apply(size, f,  )
		};
*/

	}


	template<typename policy, typename... Args> struct transform_tuple;
	template<typename policy, typename... Args> std::ostream & operator << (std::ostream && ss, transform_tuple<policy, Args...> & tt);

	struct policy_flatten{
		static constexpr bool indexed = false;
		static constexpr bool scalar = true;
	};

	struct policy_indexed{
		static constexpr bool indexed = true;
		static constexpr bool scalar = false;
	};


	template<typename policy, typename... Args> struct transform_tuple{
		std::tuple<Args...> t;
		typedef std::tuple<Args...> Ts;
		typedef typename std::tuple_element<0, Ts>::type A0;
		static constexpr int N = std::tuple_size<Ts>::value;
		static constexpr int dims = A0::my_dims;

		transform_tuple(Args... aa):t(aa...){
		}

		transform_tuple(const Ts & _t):t(_t){
		}

		template <int i = N-1> bool device_allowed(){
			bool r = std::get<i>(t).device_allowed();
			if(i>0) r = r && device_allowed<(i > 0 ? i-1: 0)>();
			return r;
		}

		template <int i = N-1> bool host_allowed(){
			bool r = std::get<i>(t).host_allowed();
			if(i>0) r = r && host_allowed<(i > 0 ? i-1: 0)>();
			return r;
		}

		template <int i = N-1> bool dim_continuous(int d){
			bool r = std::get<i>(t).dim_continuous(d);
			if(i>0) r = r && dim_continuous<(i > 0 ? i-1: 0)>(d);
			return r;
		}

		template <int k = N-1> void swap_dims(int d1, int d2){
			auto & a = std::get<k>(t);
			a = a.swap_dims(d1,d2);
			/*
			if(d2 < d1){
				std::swap(d1,d2); // reorder so that d1 < d2
			};
			constexpr dims = decltype(a)::my_dims;
			if(d2 < dims){
				a = a.swap_dims(d1,d2);
			}else if(d1 < dims){
				// d2 is outr of bouds -> create permutation
				intn<dims> o;
				for(int d = 0; d<dims;++d){
					o[d] = d;
				};
				o = o.erase<d1>().insert<dims-1>(d1);
				a = a.permute_dims(o);
			};
			*/
			if(k>0) swap_dims<(k > 0 ? k-1: 0)>(d1,d2);
		}

		template <int k = N-1> void permute_dims(const intn<dims> & o){
			auto & a = std::get<k>(t);
			a = a.permute_dims(o);
			if(k>0) permute_dims<(k > 0 ? k-1: 0)>(o);
		}

		template <int k = N-1> void print(std::ostream & ss){
			if(k == 0) ss << "tuple:";
			if(k>0) print<(k > 0 ? k-1: 0)>(ss);
			auto & a = std::get<k>(t);
			ss << a;
			if(k == N-1) ss << "\n";
		}

		template <typename U, int k> struct compress;
		template <typename U> struct compress<U,0>{
			typedef decltype( std::make_tuple(std::get<0>(Ts()).compress_dim(0)) ) type;
			static type apply(Ts & t, U dc){
				return std::make_tuple(std::get<0>(t).compress_dim(dc));
			};
		};
		template <typename U, int k = N-1> struct compress{
			typedef typename compress<U, k-1>::type subtype;
			typedef decltype(   std::tuple_cat(subtype(), std::make_tuple(std::get<k>(Ts()).compress_dim(0)) )   ) type;
			static type apply(Ts & t, U dc){
				return std::tuple_cat( compress<U, k-1>::apply(t, dc), std::make_tuple(std::get<k>(t).compress_dim(dc)) );
			};
		};

		//! compressed dimensions dc,dc+1 and returns resulting tuple
		auto compress_dim(int dc) -> decltype( make_transform(policy(), compress<int>::apply(t,dc)) ){
			return make_transform( policy() , compress<int>::apply(t,dc) );
		}

		template <int k = 0> bool reorder(){
			if(k==0){
				int i = std::get<0>(t).stride_bytes().min_idx();// fastest dim according to get<0> <- output array
				if(i!=0){
					swap_dims(0, i);
				};
			}else{
				auto & a = std::get<k>(t);
				int i = a.stride_bytes().min_idx();// fastest dim according to input k
				if(i > 1 && a.dim_linear(i)){ // contiguous
					swap_dims(1, i);
					return true;
				};
			}
			bool dim1_used = false;
			if(k<N-1)dim1_used = dim1_used || reorder<(k < N-1 ? k+1: N-1)>();
			if(k==0){
				// fix dim0 and dim1 if dim1_used, sort the reminder
				auto & a = std::get<0>(t);
				intn<dims> st = a.stride_bytes();
				st[0] = 0; if(dim1_used) st[1] = 0;
				intn<dims> o = st.sort_idx(); // ascending order sorting (stable)
				permute_dims(o);
			};
			return dim1_used;
		}


		template <typename F> void apply(F f, policy_flatten p){
			std::cout << "applying " << type_name(f) << " to ";
			//std::cout << t;
			print(std::cout);
			//
			reorder();
			//
			// flatten dimensions that are continuous for the whole tuple and forward operation there
			{constexpr int d1 = 0;
			if(dim_continuous(d1)) return compress_dim(d1).apply(f, p);
			}
			{constexpr int d1 = dims>2? 1: 0;
			if(dim_continuous(d1)) return compress_dim(d1).apply(f, p);
			}
			{constexpr int d1 = dims>3? 2: 0;
			if(dim_continuous(d1)) return compress_dim(d1).apply(f, p);
			}
			{constexpr int d1 = dims>4? 3: 0;
			if(dim_continuous(d1)) return compress_dim(d1).apply(f, p);
			}
			static_assert(dims <=5,"add implementation");

			// dispatch host / device
			std::cout << "dispatching " << type_name(f) << " to ";
			print(std::cout);

			if(device_allowed()){
				apply_on_device(f);
			}else{
				runtime_check(host_allowed());
				apply_on_host(f);
			};
		}

		template <typename F> void apply(F f, policy_indexed p){
			std::cout << "applying " << type_name(f) << " to ";
			print(std::cout);
			// dispatch host / device
			std::cout << "dispatching " << type_name(f) << " to ";
			//
			if(device_allowed()){
				apply_on_device(f);
			}else{
				runtime_check(host_allowed());
				apply_on_host(f);
			};
		}

		template <typename F> void apply_on_host(F f){

		}

		template <typename F> void apply_on_device(F f){
			auto fb = tuple_expand_and_bind<dims,F, policy::indexed, Args...>(f, t, tuple_index(t));
			intn<dims> size = std::get<0>(t).size();
			nd::device_op1::for_each_dims<dims>::apply(size, fb);
		}
	};

	template<typename... TT> std::ostream & operator << (std::ostream && ss, transform_tuple<TT...> & tt){
		tt.print(ss);
		return ss;
	};

	template<typename policy, typename... Args> transform_tuple<policy, Args...> make_transform(policy p, const Args&... args){
		return transform_tuple<policy, typename std::decay<Args>::type...>(args...);
	}

	template<typename policy, typename... Args> transform_tuple<policy, Args...> make_transform(policy p, const std::tuple<Args...> & _t){
			return transform_tuple<policy, typename std::decay<Args>::type...>(_t);
	}


	/*
	template<typename... Args> transform_tupler<Args...> transform_tuple(Args... aa){

	};

	 */


	//---------- functor parsers ---------------------------------------
	template<typename F, typename FB, typename... AA> std::tuple<AA...> functor_args(const F & f, void (FB::* method)(AA... )const ){
		return std::tuple<AA...>();
	}

	template<typename F, typename FB, typename... AA> std::tuple<AA...> functor_args(const F & f, void (FB::* method)(AA... )){
		return std::tuple<AA...>();
	}

	template<typename F, int dims> struct functor_parse{
		typedef decltype( functor_args(*((F*)0), &F::operator() ) ) args_tuple;
		typedef typename std::tuple_element<0, args_tuple >::type f_first;
		typedef const intn<dims> & tindex;
		static constexpr bool indexed = std::is_same<f_first, tindex >::value;
		template<typename A0, typename... AA> struct matches_indexed_args{
			static constexpr bool value = std::is_same< args_tuple, std::tuple<tindex, typename std::decay<decltype(A0().kernel())>::type &, const typename std::decay<decltype(AA().kernel())>::type &... > >::value;
		};
		template<typename A0, typename... AA> struct matches_scalar_args{
			static constexpr bool value = std::is_same< args_tuple, std::tuple<typename A0::my_type&, const typename AA::my_type & ...> >::value;
		};
	};

	//---------- transform facade --------------------------------------
	//------------------------------------------------------------------
	//! transform using a functor
	template <typename F, typename A0, typename... AA>
	void transform(const F & f, A0 & a, const AA&... aa){
		constexpr int dims1 = array_parse<A0>::dims;
		constexpr bool indexed = functor_parse<F,dims1>::indexed;
		typedef typename std::conditional<indexed, policy_indexed, policy_flatten>::type policy;
		static_assert( !indexed || functor_parse<F,dims1>::template matches_indexed_args<A0,AA...>::value , "Indexed functor must take kernels of arguments of the transform");
		static_assert(  indexed || functor_parse<F,dims1>::template matches_scalar_args<A0,AA...>::value , "Scalar functor must take elements of arguments of the transform");
		make_transform(policy(), a, aa...).apply(f, policy());
	}

};
