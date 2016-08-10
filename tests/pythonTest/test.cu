#include "test.cuh"

#define BLOCK_SIZE 16

__global__ void cuda_kernel(cudaTextureObject_t tex_input, iu::ImageGpu_32f_C1::KernelData result,
                            float parameter)
{
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < result.width_ && y < result.height_)
    {
        result(x,y) = tex2D<float>(tex_input, result.width_-x-0.5f, result.height_-y-0.5f) * parameter;
    }
}


void cuda_function(iu::ImageGpu_32f_C1& input, iu::ImageGpu_32f_C1 &result, float parameter)
{
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(iu::divUp(input.width(), BLOCK_SIZE), iu::divUp(input.height(), BLOCK_SIZE));

    cuda_kernel<<< dimGrid, dimBlock >>> (input.getTexture(), result, parameter);
}
