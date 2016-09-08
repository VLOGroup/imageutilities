#include <cuda.h>
#include <assert.h>
//#include <device_functions.h>
#include "./../src/ndarray/ndarray_ref.kernel.h"

// using ndarray_ref inside kernel
	//                                                   v here arguments passed by value
__global__ void my_kernel_1(kernel::ndarray_ref<float, 2> result, kernel::ndarray_ref<float, 3> data){
	//                      ^^^^^^^^^^^^^^^^^^^ the base class of ndarray_ref, all __device__ functionality
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if( x >= result.size(0) || y >= result.size(1))return;
	//              ^method
	if( x >= data.size(0) || y >= data.size(1))return;
	float sum = 0;
	for (int z = 0; z < data.size(2); ++z){ // loop over 3rd dimension
		sum += data(x,y,z);
	//             ^operator()
	};
	result(x, y) = sum;
	//    ^operator()
	float * __restrict__ p_data = data.ptr(x, y, 0); // pointer to the beginning of the data to sum
	//                                 ^ get a raw pointer to data,
	for (int z = 0; z < data.size(2); ++z){
			sum += *p_data;
			p_data += data.stride(2);
	//		               ^ stride along dimension 2, of a special proxy type tstride<char>
	// pointer arithmetics is overloaded, so you don't see that the stride is in bytes
	};
	// can also use some function if you need to:
	kernel::ndarray_ref<float, 2> data_slice = data.subdim<2>(0);
	result(x, y) = data_slice(x,y);
}

#define divup(x,y) ((x-1)/(y)+1)
#define roundup(x,y) (divup(x,y)*y)

#include "ndarray_example.h"
#include "ndarray/ndarray_ref.host.h"

    //                                    v passing by value accepts temporaries
void call_my_kernel(ndarray_ref<float, 2> result, const ndarray_ref<float, 3> & data){
	//                                            ^ const & accepts temporaries too
	runtime_check(result.size(0) == data.size(0));
	runtime_check(result.size(1) == data.size(1));
	dim3 dimBlock(8 , 8, 1);
	dim3 dimGrid(divup(result.size(0), dimBlock.x), divup(result.size(1), dimBlock.y), 1);
	my_kernel_1 <<< dimGrid, dimBlock >>>(result, data);
}

#include "ndarray/ndarray.h"
#include "ndarray/ndarray_print.h"
#include "ndarray/ndarray_iterator.h"
#include "ndarray/ndarray_iterator_over.h"
#include "ndarray/ndarray_op.h"

#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>


template<typename OutIterator, typename InIterator>
__global__ void test_kernel__05__(InIterator beg, OutIterator out, int n){
	int i =  threadIdx.x +blockIdx.x*blockDim.x;
	if(i<n){
		beg += i;
		out += i;
		*out = *beg;
	};
}

void test_iterator_kernel(){
	typedef thrust::device_system_tag System;
	ndarray<float, 3> a;
	a.create<memory::GPU>({500,600,700});
	int B = 256;
	auto in_b = a.begin_it1<System>();
	auto in_e = a.end_it1<System>();
	int n = in_e - in_b; // distance
	int G = (n+B-1)/B;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	test_kernel__05__<<<G,B>>>(in_b,in_b,n);
	cudaDeviceSynchronize();
	cuda_check_error();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime , start, stop);
	printf("test_kernel__05__ time is %f ms\n", elapsedTime);
}


void test_thrust_iterator_1(){
	typedef thrust::device_system_tag System;
	//ndarray<float, 4> a;
	ndarray<float, 3> a;
	a.create<memory::GPU>({40,50,20});
	auto a1 = a.permute_dims({2,1,0});
	//auto a1 = a.ref();
	std::cout << a1 << "a1.numel()=" <<a1.numel() << "\n";
	int i = 0;
	auto it = a1.begin_it1<System>();
	auto end = a1.end_it1<System>();
	for(; it!=end ; ++it, ++i){
		//std::cout<<"i="<<i<<" it.ii="<<it.ii<<" it.ptr="<<&(*it) - a1.ptr() << "\n";
		//runtime_check(it!=end) << it.ii << ":" << end.ii<<"\n";
		runtime_check(it!=end)<< it.ii << (int)(&(*it)-a1.ptr()) << " : " << end.ii << int(&(*end)-a1.ptr()) << "\n";
		//<< a1.ptr(it.ii) << "/" << a1.ptr(end.ii) <<"\n";
		auto it1 = a1.begin_it1<System>();
		runtime_check(it1 + i == it);
		int d = end - it;
		// relaxed assumption on iterator distance (upper bound):
		runtime_check(d >= a1.numel() - i) << d <<" vs. " <<a1.numel() - i << " i="<<i<<"\n";
	};
	//runtime_check(it==end)<<it.ii<<it.offset.value()<<"/"<<end.ii<<end.offset.value();
	std::cout << "Iterator advancing checked" << "\n";
	//test_iterator_kernel();
	//std::cout << "Iterator kernel checked" << "\n";
};

template<typename System>
void test_thrust_iterator_transform(const ndarray_ref<float,3> & a1){
	auto a_beg = a1.begin_it<System>();
	auto a_end = a1.end_it<System>();
	thrust::fill(a_beg,a_end,0.0f);
	thrust::transform(a_beg,a_end,thrust::counting_iterator<float>(1.0f),a_beg,thrust::plus<float>());
	//
	//int d = a_end - a_beg;
	//auto adv = a_beg + d;
	//float s = 0;
	//runtime_check(adv == a_end) << adv.ii << (int)(&*adv-a1.ptr()) << " : " << a_end.ii << int(&*a_end-a1.ptr()) << "\n";
	//cudaDeviceSynchronize();
	//cuda_check_error();
	//cudaDeviceSynchronize();
	//cuda_check_error();

	/*
	s = thrust::reduce(a_beg,a_end);
	int n = a1.numel();
	int S = n*(n+1)/2;
	std::cout << "sum = " << s <<"\n";
	if(s < (1<<24)){ // significant bits
		runtime_check(S == int(s+0.5)) << S << " v.s. " << s;
	}
	*/

	//float s = 0;
	//float s1 = thrust::reduce(v.begin(),v.end());
	//
};

template<typename System>
void test_thrust_iterator_transform1(const ndarray_ref<float,3> & a1){
	auto a_beg = a1.begin_it1<System>();
	auto a_end = a1.end_it1<System>();
	thrust::fill(a_beg,a_end,0.0f);
	thrust::transform(a_beg,a_end,thrust::counting_iterator<float>(1.0f),a_beg,thrust::plus<float>());
	float s = thrust::reduce(a_beg,a_end);
	int n = a1.numel();
	int S = n*(n+1)/2;
	std::cout << "sum = " << s <<"\n";
	if(s < (1<<24)){ // significant bits
		runtime_check(S == int(s+0.5)) << S << " v.s. " << s;
	}

};

void test_3D(ndarray_ref<float,3> a1){
	a1 << 0.0f;
	cudaDeviceSynchronize();
	a1 += 1.0f;
};

template<typename System>
void test_thrust_iterator_transform_mark(const ndarray_ref<float,3> & a1){
	thrust::device_ptr<float> a_beg((float *)a1.ptr());
	thrust::device_ptr<float> a_end = a_beg + a1.numel();
	//float * a_beg = (float *)a1.ptr();
	//float * a_end = a_beg + a1.numel();
	thrust::fill(a_beg,a_end,0.0f);
	thrust::transform(a_beg,a_end,thrust::counting_iterator<float>(1.0f),a_beg,thrust::plus<float>());
	float s = thrust::reduce(a_beg,a_end);
	int n = a1.numel();
	int S = n*(n+1)/2;
	//std::cout << "sum = " << s <<"\n";
	if(s < (1<<24)){ // significant bits
		runtime_check(S == int(s+0.5)) << S << " v.s. " << s;
	};
};


void test_thrust_iterator_2(){
	//{int n=0;
	//for(int n3=1; n3<20; ++n3){
	//	for(int n2=1; n2<40; ++n2){
	//		for(int n=1; n<20; ++n){
	//{{{int n = 9; int n2 = 34; int n3 = 3;
	//{{{int n = 19; int n2 = 25; int n3 = 2;
	{{{
				typedef thrust::device_system_tag System;
				//typedef thrust::host_system_tag System;
				ndarray<float, 3> a;
				a.create<memory::GPU>({500,600,700});
				//a.create<memory::GPU>({5,6,70});
				//a.create<memory::GPU>({n,n+1,n/2});
				//a.create<memory::GPU>({n,n2,n3});
				std::cout<<"iterator test " << a.size() <<"\n";
				for(int d=-2; d<4; ++d){
					//std::cout << "a1 = " << a1 << "\n";
					std::cout<<" test # " << d <<"\n";
					cudaEvent_t start, stop;
					cudaEventCreate(&start);
					cudaEventCreate(&stop);
					cudaEventRecord(start, 0);
					switch(d){
						case -2: test_thrust_iterator_transform_mark<System>(a); break;
						case -1: test_3D(a); break;
						case 0: test_thrust_iterator_transform<System>(a); break;
						case 1: test_thrust_iterator_transform1<System>(a); break;
						//case 1: test_thrust_iterator_transform<System>(a.permute_dims<0,1>()); break;
						case 2: test_thrust_iterator_transform<System>(a.swap_dims(0,2)); break;
						case 3: test_3D(a.swap_dims(0,2)); break;
					};
					cudaDeviceSynchronize();
					cudaEventRecord(stop, 0);
					cudaEventSynchronize(stop);
					float elapsedTime;
					cudaEventElapsedTime(&elapsedTime , start, stop);
					cuda_check_error();
					ndarray<float, 3> b;
					auto a_sub = a.subrange({0,0,0},min(intn<3>({4,5,2}),a.size()));
					//auto a_sub = a.ref();
					b.create<memory::CPU>(a_sub);
					copy_data(b,a_sub);
					print_array("\n a=", b, 0);
					printf("Time is %f ms\n", elapsedTime);
				};
			};
		};
	};
}
