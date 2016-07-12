#include <cuda.h>
#include <device_functions.h>
#include "ndarray_example.h"

// using ndarray_ref inside kernel, simple level
__global__ void my_kernel_1(kernel::ndarray_ref<float, 2> result, kernel::ndarray_ref<float, 3> data){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if( x >= result.size(0) || y >= result.size(1))return;
	if( x >= data.size(0) || y >= data.size(1))return;
	float sum = 0;
	for (int z = 0; z < data.size(2); ++z){ // loop over 3rd dimension
		sum += data(x,y,z);
	};
	result(x, y) = sum;
}

// using a pointer to data
__global__ void my_kernel_2(kernel::ndarray_ref<float, 2> result, kernel::ndarray_ref<float, 3> data){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if( x >= result.size(0) || y >= result.size(1))return;
	float sum = 0;
	float * __restrict__ p_data = data.ptr(x, y, 0); // pointer to the beginning of the data to sum
	for (int z = 0; z < data.size(2); ++z){
		sum += *p_data;
		p_data += data.stride(2);
	};
	result(x, y) = sum;
}

/*
// using dslice
__global__ void my_kernel_3(kernel::ndarray_ref<float, 2> result, kernel::ndarray_ref<float, 3> data){
	__shared__ dslice<float, 3> block_data;
	float register_array[10];
	if (threadIdx.x == 0 && threadIdx.y == 0){// one thread creates a per-block shortcut to the data
		block_data = data.offset(blockIdx.x * blockDim.x, blockIdx.y * blockDim.y);
	};
	__syncthreads();
#pragma unroll 10
	for (int z = 0; z < data.size(2); ++z){
		register_array[z] = block_data(threadIdx.x, threadIdx.y, z);
	};
	float sum = 0;
	for (int z = 0; z < data.size(2); ++z){
		sum += register_array[z];
	};
}
*/


#define divup(x,y) ((x-1)/(y)+1)
#define roundup(x,y) (divup(x,y)*y)

#include "ndarray_ref.host.h"

void call_my_kernel(ndarray_ref<float, 2> & result, const ndarray_ref<float, 3> & data){
	runtime_check(result.size(0) == data.size(0));
	runtime_check(result.size(1) == data.size(1));
	dim3 dimBlock(8 , 8, 1);
	dim3 dimGrid(divup(result.size(0), dimBlock.x), divup(result.size(1), dimBlock.y), 1);
	my_kernel_1 <<< dimGrid, dimBlock >>>(result, data);
}
