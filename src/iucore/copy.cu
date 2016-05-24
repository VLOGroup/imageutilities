//#include "coredefs.h"
//#include "memorydefs.h"

#include "../iucore.h"

namespace iuprivate {

//template<typename PixelType, class AllocatorGpu, IuPixelType _pixel_type>
//__global__ void cu_imgToLinearMemory(iu::ImageGpu<PixelType, AllocatorGpu, _pixel_type>::KernelData img, iu::LinearDeviceMemory<PixelType>::KernelData linMem)
__global__ void cu_imgToLinearMemory(iu::ImageGpu_32f_C1::KernelData img, iu::LinearDeviceMemory_32f_C1::KernelData linMem)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x < img.width_ && y < img.height_)
  {
	  linMem(y * img.width_ + x) = img(x, y);
  }
}

//template<typename PixelType, class AllocatorGpu, IuPixelType _pixel_type>
//void copy(const iu::ImageGpu<PixelType, AllocatorGpu, _pixel_type>* src, iu::LinearDeviceMemory<PixelType>* dst)
void cuCopy(const iu::ImageGpu_32f_C1* src, iu::LinearDeviceMemory_32f_C1* dst)
{
	const unsigned int block_size = 16;
	dim3 threadsPerBlock(block_size, block_size);
	dim3 numBlocks(iu::divUp(src->width(), threadsPerBlock.x),
				   iu::divUp(src->height(), threadsPerBlock.y));

	cu_imgToLinearMemory<<<numBlocks, threadsPerBlock>>>(*src, *dst);
	cudaDeviceSynchronize();
}

}
