#include <cuda_runtime_api.h>
#include <iostream>

__global__ void my_kernel_00(void * p){
}

int main(){

	int * ptr;
	cudaMallocManaged(&ptr, 1000*sizeof(int), cudaMemAttachGlobal);

	my_kernel_00 <<< 1, 1 >>>(0);
	//cudaDeviceSynchronize();

	std::cout << ptr[81] << "\n";

	return 0;
}
