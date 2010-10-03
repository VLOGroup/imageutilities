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
    IU_ASSERT(size.width * size.height > 0);
    cudaError_t status;
    PixelType* buffer = 0;
    status = cudaMallocPitch((void **)&buffer, pitch,
                             size.width * sizeof(PixelType), size.height);

    IuStatus iu_status = iu::checkCudaErrorState();
    IU_ASSERT(status == cudaSuccess && iu_status == IU_NO_ERROR);

    return buffer;
  }

  static void free(PixelType *buffer)
  {
    cudaError_t status;
    status = cudaFree((void *)buffer);
    IuStatus iu_status = iu::checkCudaErrorState();
    IU_ASSERT(status == cudaSuccess && iu_status == IU_NO_ERROR);
  }

  static void copy(const PixelType *src, size_t src_pitch, PixelType *dst, size_t dst_pitch, IuSize size)
  {
    cudaError_t status;
    status = cudaMemcpy2D(dst, dst_pitch, src, src_pitch,
                          size.width * sizeof(PixelType), size.height,
                          cudaMemcpyDeviceToDevice);
    IuStatus iu_status = iu::checkCudaErrorState();
    IU_ASSERT(status == cudaSuccess && iu_status == IU_NO_ERROR);
  }
};

} // namespace iuprivate

#endif // IUCORE_IMAGE_ALLOCATOR_GPU_H
