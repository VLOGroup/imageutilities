
#ifndef IUCORE_IMAGE_ALLOCATOR_GPU_H
#define IUCORE_IMAGE_ALLOCATOR_GPU_H

#include <assert.h>
#include <cuda_runtime.h>
#include "../iucutil.h"

#include "coredefs.h"

namespace iuprivate {

//--------------------------------------------------------------------------
template <typename PixelType>
class ImageAllocatorGpu
{
public:
  static PixelType* alloc(IuSize size, size_t *pitch)
  {
    if ((size.width == 0) || (size.height == 0)) throw IuException("width or height is 0", __FILE__, __FUNCTION__, __LINE__);
    PixelType* buffer = 0;
    IU_CUDA_SAFE_CALL(cudaMallocPitch((void **)&buffer, pitch,
                                         size.width * sizeof(PixelType), size.height));
    return buffer;
  }

  static void free(PixelType *buffer)
  {
    IU_CUDA_SAFE_CALL(cudaFree((void *)buffer));
  }

  static void copy(const PixelType *src, size_t src_pitch, PixelType *dst, size_t dst_pitch, IuSize size)
  {
    IU_CUDA_SAFE_CALL(cudaMemcpy2D(dst, dst_pitch, src, src_pitch,
                          size.width * sizeof(PixelType), size.height,
                          cudaMemcpyDeviceToDevice));
  }
};

} // namespace iuprivate

#endif // IUCORE_IMAGE_ALLOCATOR_GPU_H
