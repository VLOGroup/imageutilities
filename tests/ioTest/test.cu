#ifdef WIN32
// necessary for Intellisense
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

// Device code
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < N)
		C[i] = A[i] + B[i];
}

// Interface functions

namespace cuda
{

void launchTestKernel(float *h_A, float *h_B, float *h_C, int N)
{
	int size = N*sizeof(float);

	// Allocate vectors in device memory
	float* d_A;
	cudaMalloc(&d_A, size);
	float* d_B;
	cudaMalloc(&d_B, size);
	float* d_C;
	cudaMalloc(&d_C, size);

	// Copy vectors from host memory to device memory
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	// Invoke kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	VecAdd << <blocksPerGrid, threadsPerBlock >> >(d_A, d_B, d_C, N);

	// Copy result from device memory to host memory
	// h_C contains the result in host memory
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	// Free device memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

}