#pragma once

#include <assert.h>
#include <cuda_runtime.h>
#include "coredefs.h"
#include "../iucutil.h"

namespace iuprivate {

//--------------------------------------------------------------------------
template <typename PixelType>
class VolumeAllocatorGpu
{
public:
  static PixelType* alloc(iu::Size<3> size, size_t *pitch)
  {
    if ((size.width==0) || (size.height==0) || (size.depth==0))
      throw IuException("width, height or depth is 0", __FILE__,__FUNCTION__, __LINE__);
    PixelType* buffer = 0;
    cudaError_t status = cudaMallocPitch((void **)&buffer, pitch, size.width * sizeof(PixelType), size.height*size.depth);
    if (buffer == 0) throw std::bad_alloc();
    if (status != cudaSuccess)
      throw IuException("cudaMallocPitch returned error code", __FILE__, __FUNCTION__, __LINE__);

    return buffer;
  }

  static void free(PixelType *buffer)
  {
    IU_CUDA_SAFE_CALL(cudaFree((void *)buffer));
  }

  static void copy(const PixelType *src, size_t src_pitch, PixelType *dst, size_t dst_pitch, iu::Size<3> size)
  {
    IU_CUDA_SAFE_CALL(cudaMemcpy2D(dst, dst_pitch, src, src_pitch,
                                      size.width * sizeof(PixelType), size.height*size.depth,
                                      cudaMemcpyDeviceToDevice));
  }
};

} // namespace iuprivate


