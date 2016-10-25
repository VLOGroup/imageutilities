#include <cuda_runtime_api.h>
#include "ndarray/ndarray.h"
#include "ndarray/ndarray_iu.h"
#include "ndarray_example.h"
#include "ndarray/tests.h"

//#include "ndarray/bit_index.h"
//#include "ndarray/transform.h"

#include "ndarray/common_transforms.h"

//void test_iterator(){
//	thrust::device_vector<int> v(4);
//	thrust::fill(thrust::device, v.begin(), v.end(), 137);
//};

/*
struct type_A{
};

typedef type_A typeB;

struct C{
	template <typename A, typename X = type_B >
	C(const A& a0){
	}
};

int msvc_bug(){
	C c(1);
}
*/


void foo(const ndarray_ref<float,2> & x){
	x(1,1) = 0;
}


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
	std::cout <<"linear 1:" << ndarray_ref<int,3>(p1,intn<3>{10,10,10},ndarray_flags::host_only) <<"\n";
	//ndarray_ref<i,3>:ptr=0x166e6a0, size=(10,10,10,), strides_b=(4,40,400,), access: host; linear_dim: 0

	ndarray_ref<int,3> t1;
	t1.set_linear_ref<false>(p1,{10,10,10},ndarray_flags::host_only);
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
	ndarray_ref<int,3> t2 = ndarray_ref<int,3>(p2,intn<3>{10,10,10});
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
	std::cout << t2(0,0,0) << "\n"; // should print 2
	runtime_check(t2(0,0,0)==2);
	//
	delete p1;
	cudaFree(p2);
}


// managed memory test
void test_1(){
	ndarray_ref<float,3> a;
	float * x = new float[1000];
	a.set_linear_ref(x, {10, 10, 10}, ndarray_flags::host_only);
	for (int i = 0; i < a.size().prod(); ++i){
		x[i] = float(i);
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
	if(r != 100) throw_error("test failed");
	//int fl = ptr_access_flags(x);

	print_array("\n c=", c, 0);
	std:: cout << "c=" << c <<"\n";
	std:: cout << "c.flip_dim(1)=" << c.flip_dim(1) <<"\n";
	c << c.flip_dim(1);
	print_array("\n c.flip_dim(1)=", c,0);
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
	iu::LinearDeviceMemory_32f_C1 L1(100);
	iu::LinearHostMemory_32f_C1 L2(100);

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

	// reshape into 4D
	iu::VolumeGpu_32f_C1 V3(4,8,10);
	auto r4 = V3.ref().reshape(2,2,8,10);
	std::cout << r4 << "\n";

	{
		iu::LinearDeviceMemory<float, 3> L0(iu::Size<3>{4, 8, 10});
		iu::LinearDeviceMemory<float,3> L1{L0.ref()};
		iu::LinearHostMemory<float,4> L2(iu::Size<4>{4,8,10,10});
		L2.ref().subdim<3>(0) << L1.ref();
	}
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
	// reshape
	std::cout << "reshape test\n"; // accessing managed memory from host
	auto r1 = r.reshape(intn<2>{7*13,2*5});
	std::cout << r1 << "\n";
	r1 = r.permute_dims({0,1,3,2}).reshape(intn<2>{7*13,2*5});
	std::cout << r1 << "\n";
	r1 = r.permute_dims({1,0,3,2}).reshape(intn<2>{7*13,2*5});
	std::cout << r1 << "\n";
	auto r2 = r.permute_dims({2,1,3,0}).reshape(2*13,5*7).reshape(2*13*5,7);
	std::cout << r2 << "\n";
	print_array("\n r2=", r2.subrange({0,0},{3,3}),0);
	r2 << r2.flip_dim(1);
	print_array("\n r2.flip_dim(1)=", r2.subrange({0,0},{3,3}),0);
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
	//test_transform();
	//test_bit_index();
	//return 0;
	//test_cuda_constructors();

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
	//
	test_IuSize();
	//
	return EXIT_SUCCESS;
	//test_thrust_iterator_1();
	//test_thrust_iterator_2();
	//std::cout << test_warn(13);
	return EXIT_SUCCESS;
}
