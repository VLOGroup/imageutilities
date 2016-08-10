#pragma once

#include <tuple>
#include "ndarray_ref.host.h"
#include "type_name.h"

namespace device_op1{
/*
	template<int dims, class Ts, typename F> struct for_each_tuple{
		static void apply(intn<dims> size, const Ts & t, const F & f);
	};

	template<int dims, class Ts, typename F> // tuple
	void apply_to_tuple(intn<dims> size, const Ts & t, const F & f){
		for_each_tuple<dims,Ts,F>::apply(size,t,f);
	}

	*/

	// export instances of this function
	template<int dims, class Ts, typename F>
	__attribute__((noinline)) void apply_to_tuple(intn<dims> size, const Ts & t, const F & f);
};



/*
namespace detail
{
    template<int... Is>
    struct seq { };

    template<int N, int... Is>
    struct gen_seq : gen_seq<N - 1, N - 1, Is...> { };

    template<int... Is>
    struct gen_seq<0, Is...> : seq<Is...> { };
}

namespace detail
{
    template<typename T, typename F, int... Is>
    void for_each_method(T&& t, F f, seq<Is...>)
    {
        auto l = { (std::get<Is>(t).f(), 0)... };
    }
}
 */

/*
template <int n, typename F, int i=0> void for_unroll(F && f){
	f();
	if(i<n){
		for_unroll<n, F, (i<n ? i+1: n)>(f);
	};
};
 */

/*
template <typename F> void for_unroll_foo(F && f){
	f(0);
};
 */

/*
template <typename Ts, typename F, int i=0> void for_tuple(Ts & t, F  f){
	f(std::get<i>(t));
	constexpr int N = std::tuple_size<Ts>::value;
	if(i<N){
		for_tuple<Ts, F, (i < N ? i+1: N)>(t, f);
	};
};

struct get_size{
	template<class T>
	auto operator()(T&& obj) const -> decltype(std::forward<T>(obj).size()) {
		return std::forward<T>(obj).size();
	}
};
 */

namespace host_detail{

	template < uint N, typename R = void >
	struct apply_func{
		template <typename F, typename... ArgsT, typename... Args >
		__host__ __device__ static R applyTuple( const F & f, const std::tuple<ArgsT...>& t, Args... args ){
			return apply_func<N-1,R>::applyTuple( f, t, std::get<N-1>(t), args... );
		}
	};

	template <typename R> struct apply_func<0, R>{
		template <typename F, typename... ArgsT, typename... Args >
		__host__ __device__ static R applyTuple(const F & f, const std::tuple<ArgsT...>& /* t */,  Args... args ){
			return f( args... );
		}
	};
};

//
template<int dims, class Ts> struct transform_tuple{
	Ts t; // work on tuple
	intn<dims> size;
	static constexpr int N = std::tuple_size<Ts>::value;

	transform_tuple(const Ts & _t):t(_t){
		size = std::get<0>(t).size();
	}

	template <int i = N-1> bool same_size(){
		bool r = std::get<i>(t).size() == std::get<0>(t).size();
		if(i>0) r = r && same_size<(i > 0 ? i-1: 0)>();
		return r;
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

	auto compress_dim(int dc) -> decltype(make_transform_tup(compress<int>::apply(t,dc))){
		return make_transform_tup(compress<int>::apply(t,dc));
	}

	template <typename U, int k> struct ptr_tuple;

	template <typename U> struct ptr_tuple<U,0>{
		//typedef typename std::tuple_element<0,Ts>::type::my_type type;
		typedef decltype(   std::make_tuple(std::get<0>(Ts() ).ptr( U() ))   ) type;
		//ptr_tuple(U ii){};
		static type apply(const Ts & t, const U & ii){
			return std::make_tuple(std::get<0>(t).kernel().ptr(ii));
		};
	};
	template <typename U, int k = N-1> struct ptr_tuple{
		typedef typename ptr_tuple<U, k-1>::type subtype;

		//typedef typename std::tuple_element<k,Ts>::type::my_type added_type;
		//typedef decltype(   std::tuple_cat(subtype(), std::make_tuple(added_type()))   ) type;
		//typedef decltype(  ptr_tuple<U, k-1>::apply(Ts(), U())  ) subtype;

		typedef decltype(   std::tuple_cat( subtype(), std::make_tuple(std::get<k>(Ts() ).ptr(U() )) )   ) type;
		//ptr_tuple(U ii){};
		static type apply(const Ts & t, const U & ii){
			return std::tuple_cat( ptr_tuple<U, k-1>::apply(t, ii), std::make_tuple(std::get<k>(t).ptr(ii)) );
		};
	};

	/*
	auto ptr_tuple(int ii) -> decltype(make_transform(compress<int>::apply(t,dc))){
		return make_transform(compress<int>::apply(t,dc));
	}
	 */

	/*
	template <int dc, int k> struct compress;
	template <int dc> struct compress<dc,0>{
		Ts & t;
		//static constexpr int N = std::tuple_size<Ts>::value;
		compress(Ts & _t):t(_t){};
		auto apply() -> decltype(std::make_tuple(std::get<0>(t).compress_dim(dc))){
			return std::make_tuple(std::get<0>(t).compress_dim(dc));
		}
	};

	template <int dc, int k = N-1> struct compress{
		Ts & t;
		compress(Ts & _t):t(_t){};
		auto apply() -> decltype(std::tuple_cat(compress<dc,k-1>(t).apply(dc)), std::make_tuple(std::get<k>(t).compress_dim(dc)) ){
			return std::tuple_cat(compress<dc,k-1>(t).apply(dc)), std::make_tuple(std::get<k>(t).compress_dim(dc));
		}
	};
	 */

	template <typename F> void apply(F f);
	template <typename F> void apply_on_host(F f);
	template <typename F> void apply_on_device(F f);
};

template<int dims, class Ts> std::ostream & operator << (std::ostream && ss, transform_tuple<dims,Ts> & tt){
	tt.print(ss);
	return ss;
};


#pragma hd_warning_disable
template<int dims, class Ts>
template <typename F>
void transform_tuple<dims, Ts>::apply(F f){
	std::cout << "applying " << type_name(f) << " to ";
	//std::cout << t;
	print(std::cout);
	// reorder
	int i = std::get<0>(t).stride_bytes().min_idx();// fastest dim according to get<0> <- output array
	if(i!=0){
		swap_dims(0, i);
	};
	bool dim1_used = false;
	if(!dim1_used){
		constexpr int k = (N> 1)? 1: 0;
		auto & a = std::get<k>(t);
		int i = a.stride_bytes().min_idx();// fastest dim according to input 1
		if(i > 1 && a.dim_linear(i)){ // contiguous
			swap_dims(1, i);
			dim1_used = true;
		};
	};

	if(!dim1_used){ // input 1 is aligned with output
		constexpr int k = (N> 2)? 2: 0;
		auto & a = std::get<k>(t);
		int i = a.stride_bytes().min_idx();// fastest dim according to input 2
		if(i > 1 && a.dim_linear(i)){ // contiguous
			swap_dims(1, i);
			dim1_used = true;
		};
	};

	if(!dim1_used){ // input 1 and 2 are aligned with output
		constexpr int k = (N> 3)? 3: 0;
		auto & a = std::get<k>(t);
		int i = a.stride_bytes().min_idx();// fastest dim according to input 2
		if(i > 1 && a.dim_linear(i)){ // contiguous
			swap_dims(1, i);
			dim1_used = true;
		};
	};

	// fix dim0 and dim1 if dim1_used, sort the reminder
	auto & a = std::get<0>(t);
	intn<dims> st = a.stride_bytes();
	st[0] = 0; if(dim1_used) st[1] = 0;
	intn<dims> o = st.sort_idx(); // ascending order sorting (stable)
	permute_dims(o);
	//
	// flatten dimensions that are continuous for the whole tuple and forward operation there
	{constexpr int d1 = 0;
	if(dim_continuous(d1)) return compress_dim(d1).apply(f);
	}
	{constexpr int d1 = dims>2? 1: 0;
	if(dim_continuous(d1)) return compress_dim(d1).apply(f);
	}
	{constexpr int d1 = dims>3? 2: 0;
	if(dim_continuous(d1)) return compress_dim(d1).apply(f);
	}
	{constexpr int d1 = dims>4? 3: 0;
	if(dim_continuous(d1)) return compress_dim(d1).apply(f);
	}
	static_assert(dims <=5,"add implementation");

	// dispatch host / device
	std::cout << "dispatching " << type_name(f) << " to ";
	print(std::cout);

	if(device_allowed()){
		intn<dims> size = std::get<0>(t).size();
		device_op1::apply_to_tuple(size, t, f);
	}else{
		runtime_check(host_allowed());
		apply_on_host(f);
	};
}

namespace host_op1{
	template<int dims, typename Func>
	void for_each(const intn<dims> & size, Func func){
		intn<dims> ii;
		for (ii = 0;;){
			// check not exceeding ranges
			if (ii[dims-1] >= size[dims-1]) break;
			// perform operation
			func(ii);
			// increment iterator
			for (int d = 0; d < dims; ++d){
				if (++ii[d] >= size[d] && d < dims-1 ){
					ii[d] = 0;
				} else break;
			};
		};
	}
};

template<int dims, class Ts>
template <typename F>
void transform_tuple<dims, Ts>::apply_on_host(F f){
	auto func = [&] (const intn<dims> & ii){
		//
		auto pp = ptr_tuple< intn<dims> >::apply(t, ii); // evaluate all pointers at index ii
		// host_detail::apply_func<N>::applyTuple(f,pp);
	};
	intn<dims> size = std::get<0>(t).size();
	host_op1::for_each(size, func);
}

/*
// declaration of device operations
namespace device_op1{
	template<int dims, typename Func> struct for_each{
	public:
		static inline void apply(const intn<dims> & r, Func & func); // implementation through a kernel launch in tarry_op.cu
	};
};
 */

/*
template <typename type, int dims> struct binary_op : public transform<ndarray_ref<type, dims>, ndarray_ref<type, dims>, ndarray_ref<type, dims> >{
	typedef transform<ndarray_ref<type, dims>, ndarray_ref<type, dims>, ndarray_ref<type, dims> > parent;
	type c1, c2; // pool of consts
	typedef type * __restrict__ ptrt;
	__host__ __device__ binary_op(ndarray_ref<type, dims> & d, ndarray_ref<type, dims> & a, ndarray_ref<type, dims> & b, type _c1=0, type _c2=0)
	:c1(_c1), c2(_c2), parent(std::make_tuple(d,a,b)){};
	__host__ __device__ void add(ptrt d, const ptrt a, const ptrt b){
 *d = *a + *b;
	}
};
 */



/*
	template <int dc> struct compress{
		static auto apply() -> decltype(std::tie(std::get<N-1>(t).template compress_dim<dc>())){
			return std::tie(std::get<N-1>(t).template compress_dim<dc>();
		}
	};
 */


//	bool same_size(){
//detail::gen_seq<sizeof...(Ts)>()
//		boost::fusion::for_each(t, [&](auto && a){f = f && (a.size() == size);});
//tuple_and<get_size> foo;
//boost::fusion::for_each(t, foo);
//		bool flag = true;
//		constexpr int N = std::tuple_size<Ts>::value;
//		for_unroll<3,[&](int i){ flag = flag && (std::get<0>(t).size() == size);}>();
//		for_tuple(t, [&](int i){ flag = flag && (std::get<i>(t).size() == size);});
//		for_unroll_foo([&](int i){ flag = flag && (std::get<0>(t).size() == size);});
//		for_unroll_foo<[&](int i){ flag = flag && (std::get<0>(t).size() == size);}>();
//		return flag;
//	}
//	};


template<typename type, int dims> constexpr int get_dims(ndarray_ref<type, dims> t){
	return dims;
};

template<class Ts, int i> constexpr int tup_dims(){
	//return std::remove_reference<decltype( std::get<i>(*(Ts*)(0)) )>::type::my_dims;
	typedef typename std::tuple_element<i,Ts>::type ith_type;
	//typedef typename std::remove_reference<decltype(std::get<i>(Ts()))>::type ith_type;
	return ith_type::my_dims;
	//return get_dims(first_type());
}

template<class Ts> transform_tuple<tup_dims<Ts,0>(), Ts> make_transform_tup(const Ts & t){
	return transform_tuple<tup_dims<Ts,0>(), Ts>(t);
};

/*
template<typename... Args> auto make_transform(const std::tuple<Args...> & t) -> decltype(   make_transform_tup(t)   ){
	return make_transform_tup(t);
};
*/

template<typename... Args> auto make_transform(const Args&... args) -> decltype(   make_transform_tup(std::make_tuple(args...))   ){
	return make_transform_tup(std::make_tuple(args...));
};


template<typename type_out, typename type_in, int dims> struct unary_op{
	type_in c1, c2; // pool of consts
	tstride<char> offset[dims];
	unary_op(type_in _c1=0, type_in _c2=0):c1(_c1),c2(_c2){};
	__host__ __device__ __forceinline__ type_out & copy(type_out & dest, const type_in __restrict__ * a){
		dest = (type_out)*a;
		return dest;
	}
};

template<typename type, int n> struct typen{
	type v[n];
	__host__ __device__ type & operator[](int i){return v[i];};
	__host__ __device__ const type & operator[](int i)const{return v[i];};
};

template<typename type, int dims> struct grad{
	tstride<char> di[dims];
	typedef typen<type,dims> grad_type;
	grad(const ndarray_ref<type, dims> & a){
		for(int d=0;d<dims;++d){
			di[d] = a.stride(d);
		};
	};
	__host__ __device__ __forceinline__ void operator()(grad_type * __restrict__ dest, const type * __restrict__ a) const {
		type v1 = *a;
		for(int dim = 0; dim < dims; ++dim){
			type v2 = *(a + di[dim]);
			(*dest)[dim] = v2 - v1;
		};
	}
};


template<typename type> struct binary_op{
	typedef type * __restrict__ ptrt;
	//binary_op(type _c1=0, type _c2=0):c1(_c1),c2(_c2){};
	struct add{
		__host__ __device__ __forceinline__ void operator()(ptrt d, const ptrt a, const ptrt b) const {
			*d = *a + *b;
		}

		__host__ __device__ __forceinline__ void operator()(type & d, const type & a, const type & b){
			d = a + b;
		}
	};

	struct add_r{
		__host__ __device__ __forceinline__ void operator()(type & d, const type & a, const type & b)const{
			d = a + b;
		}
	};

	struct madd2{
		type c1, c2;
		madd2(type _c1, type _c2):c1(_c1),c2(_c2){};
		__host__ __device__ __forceinline__ void operator()(ptrt d, const ptrt a, const ptrt b) const {
			*d = (*a)*c1 + (*b)*c2;
		}
	};
};

/*
template<typename O, typename R, typename... T> struct method_functor{
    typedef R (*function_type)(T...);
    function_type function;
    ) obj;
public:
    method_functor(O & x, function_type f): obj(x),function(f){
    }

    R operator() (T&&... a){
        R r = obj.function(std::forward<T>(a)...);
        return r;
    }
};

template<typename O, typename R, typename... T> method_functor<O,R,T...> wrap_method(O & o, method_functor<O,R,T...>::function_type f){
	return method_functor<O,R,T...>(o,f);
};
 */

void test_transform();

/*

template<class T> void binary_op(){
	runtime_check(a.size() == b.size());
	runtime_check(a.size() == d.size());
	// reorder
	int i = a.stride_bytes().min_idx();// fastest dim according to input 1 <- output array
	if(i!=0){
		a = a.swap_dims(0,i);
		b = b.swap_dims(0,i);
		d = d.swap_dims(0,i);
	};
	i = b.stride_bytes().min_idx();// fastest dim according to input 2
	bool dim1_used = false;
	if(i > 1 && b.stride_bytes(i) == sizeof(type)){ // contiguous
		a = a.swap_dims(1,i);
		b = b.swap_dims(1,i);
		d = d.swap_dims(1,i);
		dim1_used = true;
	}else{// input2 is ok, chose dim1 for output
		i = d.stride_bytes().min_idx();// fastest dim according to output
		if(i>1){
			a = a.swap_dims(1,i);
			b = b.swap_dims(1,i);
			d = d.swap_dims(1,i);
			dim1_used = true;
		};
	};
	// fix dim0 and dim1 if dim1_used, sort the reminder
	intn<dims> st = a.stride_bytes();
	st[0] = 0; if(dim1_used) st[1] = 0;
	intn<dims> o = st.sort_idx(); // ascending order sorting (stable)
	a = a.permute_dims(o);
	b = b.permute_dims(o);
	d = d.permute_dims(o);
	//
	// flatten dimensions continuous for the whole tuple
	for_unroll<>([&]-> bool (constexpr int i){
		if(a.dims_continuous(i) && b.dims_continuous(i) && d.dims_continuous(i)){
			binary_op<type,dims-1,OP>(a.compress_dim<i>(), b.compress_dim<i>(), c.compress_dim<i>(), op);
			return true;
		};
		return false;
	});
	// dispatch host / device
	if(a.device_allowed() && b.device_allowed()){
		runtime_check(d.device_allowed());
		device_op::binary_op(a,b,dest,op);
	}else{
		runtime_check(a.host_allowed());
		runtime_check(b.host_allowed());
		runtime_check(d.host_allowed());
		host_op::binary_op(a,b,dest,op);
	};
}
 */

template<typename... Args> struct ptr_functor{
	typedef void (*f_type)(Args&...);
	f_type f;
	ptr_functor(const f_type & _f):f(_f){};
	void operator()(Args*... pargs)const{
		f(*pargs...);
	}
};

/*
template <typename T>
struct func_args{ };

template<typename Res, typename... Args>
struct func_args<Res (Args...)>{
	typedef Res (*f_type)(Args...);
	func_args
};

*/


template<typename... ArgsF, typename... Args> void nd_transform(void (*f)(ArgsF...), const Args&... args){
	auto tup = std::make_tuple(args...);
	auto transform = make_transform_tup(tup);
	auto ff = ptr_functor<typename std::decay<ArgsF>::type...>(f);
	transform.apply(ff);
}
/*
template<typename T, typename... ArgsF> std::tuple<ArgsF...> parse_method(T* pObj, void (T::*f)( ArgsF... )){
	return std::tuple<ArgsF...>();
}
*/

template<typename F, typename... ArgsF> struct wrap_functor{
	F f;
	__host__ __device__ __forceinline__ wrap_functor(const F & _f):f(_f){};
	__host__ __device__ __forceinline__ void operator()(ArgsF * __restrict__ ... pargs)const{
		f(*pargs...);
	}
};

template<typename F, typename... ArgsF> wrap_functor<F, typename std::decay<ArgsF>::type...> wrap_method(const F & f, void (F::*method)( ArgsF... )const){
	return wrap_functor<F, typename std::decay<ArgsF>::type...>(f);
}

//! case when f arguments are pointers
template<typename F, typename... ArgsF> F wrap_method(const F & f, void (F::*method)( ArgsF*... )const){
	return f;
}

/*
template<typename F, typename... ArgsF> wrap_functor<F, typename std::decay<ArgsF>::type...> wrap_method(const F & f, void (F::*method)( ArgsF... )){
	return wrap_functor<F, typename std::decay<ArgsF>::type...>(f);
}
*/

template<typename F, typename... Args> void nd_transform(const F & f, const Args&... args){
	auto tup = std::make_tuple(args...);
	auto transform = make_transform_tup(tup);

	//auto ff = wrap_functor<typename std::decay<ArgsF>::type...>(f);
	auto ff = wrap_method(f, &F::operator() );

	transform.apply(ff);
}



