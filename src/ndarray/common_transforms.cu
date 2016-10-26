#include <cuda.h>

#include "detail/transform.cuh"
#include "common_transforms.h"
#include "ndarray_mem.h"

using namespace nd;

template<typename type, int dims> void test_all(){
	ndarray<type, dims> a;
	intn<dims> sz; sz = 100;
	a.template create<memory::GPU>(sz);
//    add(a,a,a);
//	add(a,a);
//	add(a, type(0));
//	madd2(a,a,a,type(0), type(0));
	ndarray_ref<type, dims+1> g;
	grad(g,a);
}

template<int dims> void test_d(){
	test_all<float,dims>();
//	test_all<int,dims>();
}

void test_transform(){
	test_d<1>();
	//test_d<2>();
	//test_d<3>();
	//test_d<4>();
};
