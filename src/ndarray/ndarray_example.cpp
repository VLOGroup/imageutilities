#include <cuda_runtime_api.h>
#include "ndarray_mem.h"
#include "ndarray_iu.h"
#include "ndarray_op.h"

#include "ndarray_example.h"

// managed memory test
void test_1(){
	ndarray_ref<float,3> a;
	float * x = new float[1000];
	a.set_linear_ref(x, {10, 10, 10});
	for (int i = 0; i < a.size().prod(); ++i){
		x[i] = i;
	};
	ndarray<float, 3> b;
	b.create<memory::GPU_managed>(10, 10, 10);
	b << 1.0f; // b is managed -> device-side function allowed
	b += a; // a is host and b is managed -> host-side function allowed
	//
	ndarray<float, 2> c;
	c.create<memory::GPU_managed>(10, 10);
	call_my_kernel(c, b); //c is device and b is managed -> device-side function allowed
	//
	std::cout << c(10, 10) << "\n"; // accessing managed memory from host
	delete x;
}


void foo(ndarray_ref<float,2> && x){
	// do something
}


template<typename type, int dims> void template_foo(ndarray_ref<type, dims> & x){
	// do something general
}
/*
template<typename type, int dims> void template_foo(ndarray_ref<float, 3> && x){
	//template_foo(x);
}
*/


//! Image utilities interfaces
void test_2(){

		{
			iu::LinearDeviceMemory<float> L1;
			iu::LinearHostMemory<float> L2;

			const ndarray_ref<float, 1> & x1 =  L1;
			//x1 *= 0.5f;
			ndarray_ref<float, 1>(L1) *= 0.5f;
			ndarray_ref<float, 1> x2 = L2;

		};
		{
			iu::ImageCpu_32f_C1 I1(200,200);
			iu::ImageGpu_32f_C3 I2(100,100);

			ndarray_ref<float, 2> x1 = I1;
			ndarray_ref<float3, 2> x2 = I2;
			x1.subrange({0,0},{100,100}) += I2.ref().subtype(&float3::y); // slice out member y of struct float3
			//
			foo(I1);
		};
		{
			iu::ImageCpu_32f_C1 I1(100,200);
			iu::VolumeCpu_32f_C1 V1(10,100,100);
			iu::VolumeGpu_32f_C3 V2(10,200,100);

			ndarray_ref<float, 3> x1 = V1;
			ndarray_ref<float3, 3> x2 = V2;
			x1 += x1;
			x2.subtype(&float3::x) += I1.ref().transp().new_dim<0>(x2.size(0));
			//
			template_foo(x1);
		};
		{
			iu::TensorCpu<float> V1(2, 5,7,13);
			iu::TensorGpu<int>   V2(7,13,2,5);

			V2.ref() << V1.ref().permute_dims({2,3,0,1});
		};
}

//! type conversions
void test_3(){
}

/*
float test_warn(long long x){
	int n = 100;
	double p = 7.0/3.0f;
	float * bla = new float[n];
	int z = 8;
	bla[2] = x;
	//z = 0.5f/ 0.0f;
	z = int(bla[z]);
	return z;
}
*/

int main(){
	test_1();
	test_2();
	//std::cout << test_warn(13);
	return 0;
}
