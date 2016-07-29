#include <cuda_runtime_api.h>
#include "../../src/ndarray/ndarray.h"
#include "../../src/ndarray/ndarray_iu.h"
#include "ndarray_example.h"

void foo(const ndarray_ref<float,2> & x){
	x(1,1) = 0;
}

void test_IuSize(){
	//-----1D------
	intn<1> a(5);
	a = 3;
	a < 10;
	runtime_check(a.width == 3);
	runtime_check(a.width == a[0]);
	runtime_check(a.height == 1);
	runtime_check(a.depth == 1);
	//-----2D------
	intn<2> b(5,6);
	b >= 0;
	b < intn<2>(10,10);
	runtime_check(b.width ==5);
	runtime_check(b.height ==6);
	runtime_check(b.width ==b[0]);
	runtime_check(b.height ==b[1]);
	runtime_check(b.depth == 1);
	//-----3D------
	intn<3> c(5,6,7);
	c >= 0;
	c < intn<3>(10,10,10);
	c == intn<3>(10,10,10);
	runtime_check(c.width ==5);
	runtime_check(c.height ==6);
	runtime_check(c.depth ==7);
	runtime_check(c.width ==c[0]);
	runtime_check(c.height ==c[1]);
	runtime_check(c.depth ==c[2]);
	c = intn<3>(2,2,2);
	intn<3> d(c);
	d *= 1.5;
	std::cout << "d=" << d << "\n";
	//-----4D------
	intn<4> s(2,3,4,5);
	//s.height; // error
	std::cout <<"s=" << s << "\n";
	//  s=(2,3,4,5,)
	std::cout << "s.erase<1>() = " << s.erase<1>() << "\n";
	//	s.erase<1>() = (2,4,5,)
	std::cout << "s.erase<1>().height = " << s.erase<1>().height << "\n";
	//	s.erase<1>().height = 4
};

void intro_test(){
	//Here a brief overview of ndarray functionality
	ndarray<float,4> A;
	A.create<memory::GPU>({3,3,3,3});
	//A += A;

	//___________________Basics___________________________
	ndarray_ref<float, 2> x; // x is "reference" to a 2D array of floats
	auto y = x; // should be treated like a pointer: can copy, pass by value, etc.
	// Can be attached to any ImageUtilities class
	iu::ImageGpu_32f_C1 I1(100,200);
	x = I1; // ImageCpu uses padded memory, ndarray_ref remembers a separate stride per dimension
	//foo(I1); // exception: host_allowed() failed
	std::cout << "x    size:" << x.size() << "\n";
	// x    size:(100,200,)
	std::cout << "x strides:" << x.stride_bytes() << "\n";
	// x strides:(4,512,)
	// Let's take a volume
	iu::VolumeGpu_32f_C1 V1(3,300,100);
	//          v  new built-in method in all iu classes
	auto z =  V1.ref();
	//^^ z will have type ndarray_ref<float,3> (this is what V1.ref() returns)
	// Complete information abut z is as follows:
	std::cout << "z= " << z << "\n";
	//
	//ndarray_ref<f,3>:ptr=0x604920000, size=(3,300,100,), strides_b=(4,512,153600,), access: device; linear_dim: 0
	//      =float^           address^                               ^strides in bytes        ^remembers it is on device
	//
	// We can slice an array by fixing one dimension
	auto u = z.subdim<0>(0); // fix dimension 0 to 0
	//                ^ dimension to fix
	std::cout << "u= " << u << "\n";
	//ndarray_ref<f,2>:ptr=0x604920000, size=(300,100,), strides_b=(512,153600,), access: device; linear_dim: 15
	// note, the slice may address memory discontinuously, even if VolumeGpu was not using padding
	//
	u = u.transp(); // logical transpose of the referenced shape
	std::cout << "after transpose u= " << u << "\n";
	// ndarray_ref<f,2>:ptr=0x604920000, size=(100,300,), strides_b=(153600,512,), access: device; linear_dim: 15
	//                                         ^^^^^^^ swapped       ^^^^^^^^^^ swapped
	u = u.subrange({0,50},{100,200}); // logically crop to the size 100x200 starting at (0,50)
	std::cout << "after subrange u= " << u << "\n";
	//ndarray_ref<f,2>:ptr=0x604926400, size=(100,200,), strides_b=(153600,512,), access: device; linear_dim: 15
	//                            ^^^^        ^^^^^^^
	// no changes to the data so far, methods of ndarray_ref are all just on the "reference" object
	// not at all like external functions:
	u += x; // we are adding to the cropped slice of V1 the image I1
	// this works because both have the same size (but different strides) and both are on GPU
	//
	// Other currently available (element-wise) functions:
	u << 0.0f; // assign constant
	u << x; // overwrite
	u *= u; // multiply

	//_________________Type Slicing__________________________
	iu::ImageGpu_32f_C3 I2(100,200); // take a 3-channel image
	x = I2.ref().subtype(&float3::x); // get an ndarray_ref to the slice of member "x" of struct float3 in ImageGpu_32f_C3
	std::cout << "x= " << x << "\n";
	//ndarray_ref<f,2>:ptr=0x6057e0000, size=(100,200,), strides_b=(12,1536,), access: device; linear_dim: 15
	//                                                              ^^ jumps over 3 floats
	y = I2.ref().subtype(&float3::y); // member "y"
	std::cout << "y= " << y << "\n";
	//ndarray_ref<f,2>:ptr=0x6057e0004, size=(100,200,), strides_b=(12,1536,), access: device; linear_dim: 15
	//                            ^^^^ offset 1 float
	// of course we can do any operation, like
	x += y;


	//________________Struct Unpacking and Conversion_________
	iu::ImageGpu_32s_C3 I3(100,200); // Image of type int
	// attach a reference
	std::cout << "I3.ref()=" << I3.ref() <<"\n";
	//ndarray_ref<4int3,2>:ptr=0x60582b000, size=(100,200,), strides_b=(12,1536,), access: device; linear_dim: 0
	//              ^^^ int3 struct
	auto w = I3.ref();
	// there is no math on int3:
	// w += w; // <- error, unless we implement int3 += int3 and instantiate a kernel for ndarray_ref<int3,3>
	// but can interpret it as an extra dimension:
	std::cout << "w.unpack()=" << w.unpack() <<"\n";
	//ndarray_ref<i,3>:ptr=0x60582b000, size=(3,100,200,), strides_b=(4,12,1536,), access: device; linear_dim: 15
	//         int^    same pointer^^^        ^added dimension        ^ 1 float stride
	I2.ref().unpack() << I3.ref().unpack(); // unpack both strides and copy with conversion from int to float


	//________________Zero Strides______________________________
	// One more useful logical opration on arrays
	// suppose we like to add Image I1 (iu::ImageGpu_32f_C1) to every channel of I2 (iu::ImageGpu_32f_C3)
	auto q = I1.ref().new_dim<0>(3);
	std::cout << "q=" << q <<"\n";
	// ndarray_ref<f,3>:ptr=0x604820000, size=(3,100,200,), strides_b=(0,4,512,), access: device; linear_dim: 15
	//                                                                 ^ offset along this dimension is zero
	// then it is valid to do:
	I2.ref().unpack() += q;
	// zero strides on the left-hand sides are currently lead to race condition:
	// q += I2.ref().unpack(); // bad


	//________________Permutation_______________________________
	// We've seen logical transposition, it is more general:
	iu::TensorGpu<float> T1(2, 5,7,13);
	iu::TensorGpu<int>   T2(7,13,2,5);
	std::cout << "T2:" << T2.ref() <<"\n";
	//ndarray_ref<i,4>:ptr=0x6058e1000, size=(7,13,2,5,), strides_b=(520,40,20,4,), access: device; linear_dim: 3
	//       TensorGpu uses descending strides layout (pyton arrays)^^^^^^^^^^^^^^
	std::cout << "T1:" << T1.ref() <<"\n";
	//ndarray_ref<f,4>:ptr=0x6058e0000, size=(2,5,7,13,), strides_b=(1820,364,52,4,), access: device; linear_dim: 3
	auto T1p = T1.ref().permute_dims({2,3,0,1});
	std::cout << "T1p:" << T1p <<"\n";
	//ndarray_ref<f,4>:ptr=0x6058e0000, size=(7,13,2,5,), strides_b=(52,4,1820,364,), access: device; linear_dim: 15
	//                                        ^^^^^^^^^              ^^^^^^^^^^^^^
	// can just copy over from a different layout with conversion from float to int
	T2.ref() << T1p;


	//   _______________Kernels_________________________________
	// consider a call to a function with the signature:
	// void call_my_kernel(ndarray_ref<float, 2>  result, const ndarray_ref<float, 3> & data);
	// (see ndarray_example.h)
	//             v implicit conversion creates a temporary object of type ndarray_ref<float, 2>
	call_my_kernel(I1, I2.ref().unpack().permute_dims({1,2,0}));
	//                 ^ some rearrangement of I2 so resulting in a temporary ndarray_ref<float, 3>


	// ________________Attaching to "wild" pointers_____________
	//the usage of new / delete is not recommended - better use container classes, here just to demonstrate how to attach to pointers
	int * p1 = new int[1000];
	// interpret as 3D array:
	//                                               vv  need to specify size, assumes linear memory layout
	std::cout <<"linear 1:" << ndarray_ref<int,3>(p1,{10,10,10}) <<"\n";
	//ndarray_ref<i,3>:ptr=0x166e6a0, size=(10,10,10,), strides_b=(4,40,400,), access: host; linear_dim: 0

	ndarray_ref<int,3> t1;
	t1.set_linear_ref<false>(p1,{10,10,10});
	//                ^ use descending strides layout (default is ascending strides)
	std::cout  <<"linear 2:" << t1 << "\n";
	//ndarray_ref<i,3>:ptr=0x166e6a0, size=(10,10,10,), strides_b=(400,40,4,), access: host; linear_dim: 2
	//                               v  specify custom stride vector, in bytes
	t1.set_ref(p1,{10,10,10},intn<3>(100,1,10)*intsizeof(int),ndarray_flags::host_only);
	//                                                     ^ specify explicitly access type
	std::cout  <<"linear 3:" << t1 << "\n";
	//ndarray_ref<i,3>:ptr=0x115ae70, size=(10,10,10,), strides_b=(400,4,40,), access: host; linear_dim: 1

	// ________________ Host or Device? _____________
	// host/device access is determined using cudaPointerGetAttributes
	int * p2;
	//the usage of cudaMalloc / cudaFree is not recommended - better use container classes, here just to demonstrate how to attach to pointers
	cudaMallocManaged(&p2, 1000*sizeof(int));
	cuda_check_error();
	ndarray_ref<int,3> t2 = ndarray_ref<int,3>(p2,{10,10,10});
	std::cout << t2 <<"\n";
	//ndarray_ref<i,3>:ptr=0x203500000, size=(10,10,10,), strides_b=(4,40,400,), access: host, device; linear_dim: 0
	//                              legal for both: host-side and device-side operations ^^^^^^^^^^^^^
	t1 << 1;  // host-side operation
	t2 << 0;  // resolves to GPU operation -- t2 has device access
	cudaDeviceSynchronize(); // needed for managed memory in CUDA 7.5, otherwise demons come out
	t2 += t1; // resolves to CPU operation -- t2 has device access but not t1
	cudaDeviceSynchronize();
	t2 += t2; // resolves to GPU operation
	cudaDeviceSynchronize();
	std::cout << t2(0,0) << "\n"; // should print 2
	runtime_check(t2(0,0)==2);
	//
	delete p1;
	cudaFree(p2);
}


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
	if(r != 100) slperror("test failed");
	int fl = ptr_access_flags(x);
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
	std::cout << V2.ref() << "\n";

	V1.ref() << 0.0f;
	V1.ref().subdim<0,1>(0,0) << 1.0f;

	V2.ref() << V1.ref().permute_dims({2,3,0,1});
	ndarray<float,4> r;
	r.create<memory::GPU_managed>(V2.ref());
	r << V2.ref();
	//std::cout << r;
	print_array("\n r=", r.subrange({0,0,0,0},{2,2,2,2}),0);
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
	test_IuSize();
	intro_test();
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
