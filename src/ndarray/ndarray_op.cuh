#pragma once
#include "ndarray_ref.kernel.h"

#include "ndarray_op.h"
#include "ndarray/error_cuda.h"
#include "ndarray/ndarray_print.h"

#define blockdim0 8
#define blockdim1 8

#define divup(x,y) ((x-1)/(y)+1)
#define roundup(x,y) (divup(x,y)*y)


namespace device_op{

	//! 1D kernel
	template< typename Func>
	__global__ void //__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
	for_each_1D_kernel(shapen<1> s, Func func){
		int i = threadIdx.x + blockDim.x * blockIdx.x;
		if(i < s.size()){
			func(i);
		};
	}

	//! 2D kernel
	template< typename Func>
	__global__ void //__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
	for_each_2D_kernel(shapen<2> s, Func func){
		intn<2> ii = intn<2>(threadIdx.x + blockDim.x * blockIdx.x, threadIdx.y + blockDim.y * blockIdx.y);
		if(ii < s.size()){
			func(ii);
		};
	}

	//! 3D kernel
	template< typename Func>
	__global__ void //__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
	for_each_3D_kernel(shapen<3> s, Func func){
		intn<3> ii = intn<3>(threadIdx.x + blockDim.x * blockIdx.x, threadIdx.y + blockDim.y * blockIdx.y, threadIdx.z + blockDim.z * blockIdx.z);
		if(ii < s.size()){
			func(ii);
		};
	}

	//! 4D kernel
	template< typename Func>
	__global__ void //__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
	for_each_4D_kernel(shapen<4> s, Func func){
		intn<4> ii = intn<4>(threadIdx.x + blockDim.x * blockIdx.x, threadIdx.y + blockDim.y * blockIdx.y, threadIdx.z + blockDim.z * blockIdx.z, 0);
		if(ii < s.size()){
			for(int & u = ii[3]; u < s.size(3); ++u){
				func(ii);
			};
		};
	}
}

namespace device_op{
	//! 1D kernel launch
	template<>
	template<typename Func>
	void struct_dims<1>::for_each<Func>(const shapen<1> & s, Func func){
		dim3 dimBlock(blockdim0 * blockdim1, 1, 1);
		dim3 dimGrid(divup(s.size(0), dimBlock.x), 1, 1);
		for_each_1D_kernel <<< dimGrid, dimBlock >>>(s, func);
		cudaDeviceSynchronize();
		cuda_check_error();
	}

	//! 2D kernel launch
	template<>
	template<typename Func>
	void struct_dims<2>::for_each<Func>(const shapen<2> & s, Func func){
		//launch<A<Func> > ll;
		//ll.launch_2D(s);
		//launch_2D_a(s, func);
		//auto func_device = [=] __device__ (const intn<2> & ii) {return ii[0];};
		//
		dim3 dimBlock(blockdim0 , blockdim1, 1);
		dim3 dimGrid(divup(s.size(0), dimBlock.x), divup(s.size(1), dimBlock.y), 1);
		for_each_2D_kernel <<< dimGrid, dimBlock >>>(s, func);
		cudaDeviceSynchronize();
		cuda_check_error();
	}

	//! 3D kernel launch
	template<>
	template<typename Func>
	void struct_dims<3>::for_each<Func>(const shapen<3> & s, Func func){
		dim3 dimBlock(blockdim0 , blockdim1, blockdim1);
		dim3 dimGrid(divup(s.size(0), dimBlock.x), divup(s.size(1), dimBlock.y), divup(s.size(2), dimBlock.z));
		for_each_3D_kernel <<< dimGrid, dimBlock >>>(s, func);
		cudaDeviceSynchronize();
		cuda_check_error();
	}

	//! 4D kernel launch
	template<>
	template<typename Func>
	void struct_dims<4>::for_each<Func>(const shapen<4> & s, Func func){
		dim3 dimBlock(blockdim0 , blockdim1, blockdim1);
		dim3 dimGrid(divup(s.size(0), dimBlock.x), divup(s.size(1), dimBlock.y), divup(s.size(2), dimBlock.z));
		for_each_4D_kernel <<< dimGrid, dimBlock >>>(s, func);
		cudaDeviceSynchronize();
		cuda_check_error();
	}

}
