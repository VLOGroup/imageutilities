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
