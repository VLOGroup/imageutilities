#include <cuda_runtime_api.h>
#include "ndarray.h"
#include "ndarray_iu.h"
#include "ndarray_example.h"

// managed memory test
void test_1(){
	ndarray_ref<float,3> a;
	float * x = new float[1000];
	a.set_linear_ref(x, {10, 10, 10}); //,ndarray_flags::host_only);
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

	//cuda_check_error();
	//
	//c << b.subdim<0>(0);
	cudaDeviceSynchronize();
	int r = (int)c(9, 9);
	std::cout << r << "\n"; // accessing managed memory from host
	if(r != 5500) slperror("test failed");
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
void test_1D(){
	iu::LinearDeviceMemory<float> L1(100);
	iu::LinearHostMemory<float> L2(100);

	ndarray_ref<float, 1> x1 =  L1;
	x1 << 0.5f;
	L1.ref() *= 0.5f;
	const ndarray_ref<float, 1> & x2 = L2;
	L2.ref() += x2;
}

void test_2D(){
	{
		iu::ImageCpu_32f_C1 I1(200,200);
		iu::ImageGpu_32f_C3 I2(100,100);
		ndarray_ref<float, 2> x1 = I1;
		ndarray_ref<float3, 2> x2 = I2;
	}
	{
		auto I = new iu::ImageGpu_32f_C1(100,100);
		ndarray_ref<float, 2> x1;
		x1 = *I;
		delete I;
	}
	iu::ImageGpu_32f_C1 I1(200,200);
	iu::ImageGpu_32f_C3 I2(100,100);
	I1.ref() << 1.0f;
	ndarray_ref<float, 3> I2x = I2.ref().unpack(); // array of size [3 x 100 x 100]
	I2x << 2.0f;
	I2.ref().subtype(&float3::x) << 1.0f;
	I1.ref().subrange({0,0},{100,100}) += I2.ref().subtype(&float3::y); // slice out member y of struct float3
	foo(I1); // implicit conversion, for non-templated function
}

void test_3D(){
	iu::ImageGpu_32f_C1 I1(5,7);
	iu::VolumeGpu_32f_C1 V1(3,5,5);
	iu::VolumeGpu_32f_C3 V2(3,7,5);

	I1.ref() << 1.0f;

	ndarray_ref<float, 3> x1 = V1;
	ndarray_ref<float3, 3> x2 = V2;
	x1 << 1.0f;
	x1 += x1;
	x2.unpack() << 0.0f;
	x2.subtype(&float3::x) *= I1.ref().transp().new_dim<0>(x2.size(0));
	//
	ndarray<float3,3> r;
	r.create<memory::GPU_managed>(x2);
	r.unpack() << x2.unpack();
	//std::cout << r;
	//print_array("\n r=", r.unpack());
	template_foo(x1);
}

void test_4D(){
	iu::TensorGpu<float> V1(2, 5,7,13);
	iu::TensorGpu<int>   V2(7,13,2,5);
	std::cout << V2;
	std::cout << V2.ref();

	V1.ref() << 0.0f;
	V1.ref().subdim<0,1>(0,0) << 1.0f;

	V2.ref() << V1.ref().permute_dims({2,3,0,1});
	ndarray<float,4> r;
	r.create<memory::GPU_managed>(V2.ref());
	r << V2.ref();
	//std::cout << r;
	print_array("\n r=", r.subrange({0,0,0,0},{2,2,2,2}));
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
	try{
		test_1();
	}catch(const std::exception & e){
		std::cout << e.what() <<"\n";
		throw e;
	};
	test_1D();
	test_2D();
	test_3D();
	test_4D();
	//std::cout << test_warn(13);
	return EXIT_SUCCESS;
}
