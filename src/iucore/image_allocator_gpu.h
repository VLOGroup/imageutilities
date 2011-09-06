/*
 * Copyright (c) ICG. All rights reserved.
 *
 * Institute for Computer Graphics and Vision
 * Graz University of Technology / Austria
 *
 *
 * This software is distributed WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 * PURPOSE.  See the above copyright notices for more information.
 *
 *
 * Project     : ImageUtilities
 * Module      : Core
 * Class       : ImageAllocatorGpu
 * Language    : C++
 * Description : Image allocation functions for Gpu images.
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUCORE_IMAGE_ALLOCATOR_GPU_H
#define IUCORE_IMAGE_ALLOCATOR_GPU_H

#include <assert.h>
#include <cuda_runtime.h>
#include "iucutil.h"
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
    cudaError_t status = cudaMallocPitch((void **)&buffer, pitch,
                                         size.width * sizeof(PixelType), size.height);
    if (buffer == 0) throw std::bad_alloc();
    if (status != cudaSuccess)
      throw IuException("cudaMallocPitch returned error code", __FILE__, __FUNCTION__, __LINE__);

    return buffer;
  }

  static void free(PixelType *buffer)
  {
    cudaError_t status = cudaFree((void *)buffer);
    if (status != cudaSuccess)
      throw IuException("cudaFree returned error code", __FILE__, __FUNCTION__, __LINE__);
  }

  static void copy(const PixelType *src, size_t src_pitch, PixelType *dst, size_t dst_pitch, IuSize size)
  {
    cudaError_t status;
    status = cudaMemcpy2D(dst, dst_pitch, src, src_pitch,
                          size.width * sizeof(PixelType), size.height,
                          cudaMemcpyDeviceToDevice);
    if (status != cudaSuccess)
      throw IuException("cudaMemcpy2D returned error code", __FILE__, __FUNCTION__, __LINE__);
  }
};

} // namespace iuprivate

#endif // IUCORE_IMAGE_ALLOCATOR_GPU_H
