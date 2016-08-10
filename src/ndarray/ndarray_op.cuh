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

	/*
	namespace permuted{
		//! 2D kernel
		template< typename Func>
		__global__ void //__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
		for_each_2D_kernel(const intn<2> S, , Func func){
			intn<2> ii = intn<2>(threadIdx.x + blockDim.x * blockIdx.x, threadIdx.y + blockDim.y * blockIdx.y);
			if(ii < S){
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
	 */
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

	/*
	// packed
	//! 2D kernel launch
	template<>
	template<typename pack, int N, typename Func>
	void struct_dims<2>::transform<pack,N,Func>(Func func, const ndarray_ref<pack,dims> & inputs[N], ndarray_ref<pack,dims> & dest){
		// work towards 128B / 256B mem transactions -- 32 / 64 floats, especially writing is essential
		// layout for large matricies
		// (th32 x float4) x (for4 x th16 x for2) -- full warp coalesced load and float4 write (cache: 16 regs / thread)
		// (th8 x float4) x (th4 x for8)
		dim3 dimBlock;
		intn<dims> size = inputs[0].size();
		// chose X size -- prefer full warp along dim 0
		if(size[0] < WARP_SIZE){ // fold over into y
			dimBlock.x = hob(size[0]); //balpark power of 2
		}else if(size[0] < WARP_SIZE*2){//use smaller x-block
			dimBlock.x = WARP_SIZE/2;
		}else{// use full x-block
			dimBlock.x = WARP_SIZE; // prefer contiguous chunk load
		};
		int XY_FOLDS = WARP_SIZE / dimBlock.x;
		// chose Y size -- XY_FOLDS times a full output warp
		size = dest.

		dim3 dimBlock(blockdim0 , blockdim1, 1);
		dim3 dimGrid(divup(s.size(0), dimBlock.x), divup(s.size(1), dimBlock.y), 1);
		for_each_2D_kernel <<< dimGrid, dimBlock >>>(s, func);
		cudaDeviceSynchronize();
		cuda_check_error();
	}
	 */
};
