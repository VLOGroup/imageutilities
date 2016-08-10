#include "transform.cuh"

namespace device_op1{

	/*
	// export instances of this
	template<int dims, class Ts, typename F>
	void for_each_tuple<dims,Ts,F>::apply(intn<dims> size, const Ts & t, const F & f){
		bind_functor_args<dims,Ts,F> ff(t, f);
		for_each(size, ff);
	};
	*/

	template<int dims, class Ts, typename F>
	void apply_to_tuple(intn<dims> size, const Ts & t, const F & f){
		bind_functor_args<dims,Ts,F> ff(t, f);
		for_each(size, ff);
	}
};

template<typename type>
void __host__ __device__ add(type * a, type * b, type * c){
	*a = *b + *c;
};

template<typename type>
void __host__ __device__ add1(type & a, type & b, type & c){
	a = b + c;
};

void instanciate_transform(){
	ndarray_ref<int,2> A1;
	ndarray_ref<int,2> A2;
	ndarray_ref<int,2> A3;
	auto tup = std::make_tuple(A1.transpose(),A2,A3);
	//const int N = std::remove_reference<decltype(std::get<0>(tup))>::type::my_dims;
	auto transform = make_transform_tup(tup);
	bool same = transform.same_size();
	//transform.apply(binary_op<int>::add());
	//transform.apply(binary_op<int>::madd2(2.0f, 3.0f));

	nd_transform(binary_op<int>::add(), A1, A2, A3);
	nd_transform(add1<int>, A1, A2, A3);
	nd_transform(binary_op<int>::add_r(), A1, A2, A3);

	make_transform(A1,A1,A1).apply([]__device__(int * a, int * b, int * c){*a = *b + *c; });

	ndarray_ref<typen<int,2>, 2> G;
	make_transform(G,A1).apply(grad<int,2>(A1));



	//
	//nd_transform([](int & a, int b, int c){ a = b+c; }, A1.transpose(), A2.ref(), A3.ref());
}

namespace transform_cu_instantiate{
	template<typename type, int dims>
	void ttest(){
		ndarray_ref<type, dims> a;
		auto tup = std::make_tuple(a,a,a);
		auto transform = make_transform_tup(tup);
		//transform.apply(typename binary_op<type>::add());
		//transform.apply(typename binary_op<type>::madd2(1.0f,1.0f));
	};

	template<typename type>
	void tdtest(){
		ttest<type,1>();
		ttest<type,2>();
		//ttest<type,3>();
		//ttest<type,4>();
	};

	template<int dims>
	void dtest(){
		/*
	// conversions
	ndarray_ref<float, dims> a;
	ndarray_ref<int, dims> b;
	a << b;
	b << a;
	a << a;
	b << b;
		 */
	};

	void test(){
		tdtest<float>();
		tdtest<int>();
		// conversions
		dtest<1>();
		dtest<2>();
		//dtest<3>();
		//dtest<4>();
	};
};
