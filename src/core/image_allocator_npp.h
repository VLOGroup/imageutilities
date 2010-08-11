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
 * Class       : ImageAllocatorNpp
 * Language    : C++
 * Description : Image allocation functions for Npp images.
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUCORE_IMAGE_ALLOCATOR_NPP_H
#define IUCORE_IMAGE_ALLOCATOR_NPP_H

#include <assert.h>
#include <cuda_runtime.h>
#include <nppi.h>
#include "coredefs.h"

namespace iuprivate {

//--------------------------------------------------------------------------
template <typename PixelType, unsigned int NumChannels>
class ImageAllocatorNpp
{
};


// PARTIAL SPECIALIZATIONS:

//--------------------------------------------------------------------------
template<>
class ImageAllocatorNpp<Npp8u, 1>
{
public:
  static Npp8u* alloc(unsigned int width, unsigned int height, size_t *pitch)
  {
    IU_ASSERT(width * height > 0);

    Npp8u *buffer = nppiMalloc_8u_C1(width, height, reinterpret_cast<int *>(pitch));
    IU_ASSERT(buffer != 0);

    return buffer;
  }

  static void free(Npp8u *buffer)
  {
    nppiFree(buffer);
  }

  static void copy(const Npp8u *src, size_t src_pitch, Npp8u *dst, size_t dst_pitch, IuSize size)
  {
    NppStatus status;
    status = nppiCopy_8u_C1R(src, (int)src_pitch, dst, (int)dst_pitch, size.nppiSize());
    IU_ASSERT(status == NPP_NO_ERROR);
  }
};

//--------------------------------------------------------------------------
template<>
class ImageAllocatorNpp<Npp8u, 2>
{
public:
  static Npp8u* alloc(unsigned int width, unsigned int height, size_t *pitch)
  {
    IU_ASSERT(width * height > 0);

    Npp8u *buffer = nppiMalloc_8u_C2(width, height, reinterpret_cast<int *>(pitch));
    IU_ASSERT(buffer != 0);

    return buffer;
  }

  static void free(Npp8u *buffer)
  {
    nppiFree(buffer);
  }

  static void copy(const Npp8u *src, size_t src_pitch, Npp8u *dst, size_t dst_pitch, IuSize size)
  {
    cudaError_t status;
    status = cudaMemcpy2D(dst, dst_pitch, src, src_pitch, size.width * 2 * sizeof(Npp8u), size.height, cudaMemcpyDeviceToDevice);
    IU_ASSERT(status == cudaSuccess);
  }
};

//--------------------------------------------------------------------------
template<>
class ImageAllocatorNpp<Npp8u, 3>
{
public:
  static Npp8u *alloc(unsigned int width, unsigned int height, size_t *pitch)
  {
    IU_ASSERT(width * height > 0);

    Npp8u *buffer =  nppiMalloc_8u_C3(width, height, reinterpret_cast<int *>(pitch));
    IU_ASSERT(buffer != 0);

    return buffer;
  }

  static void free(Npp8u *buffer)
  {
    nppiFree(buffer);
  }

  static void copy(const Npp8u *src, size_t src_pitch, Npp8u *dst, size_t dst_pitch, IuSize size)
  {
    cudaError_t status;
    status = cudaMemcpy2D(dst, dst_pitch, src, src_pitch, size.width * 3 * sizeof(Npp8u), size.height, cudaMemcpyDeviceToDevice);
    IU_ASSERT(status == cudaSuccess);
  }
};

//--------------------------------------------------------------------------
template<>
class ImageAllocatorNpp<Npp8u, 4>
{
public:
  static Npp8u* alloc(unsigned int width, unsigned int height, size_t *pitch)
  {
    IU_ASSERT(width * height > 0);

    Npp8u *buffer = nppiMalloc_8u_C4(width, height, reinterpret_cast<int *>(pitch));
    IU_ASSERT(buffer != 0);

    return buffer;
  }

  static void free(Npp8u *buffer)
  {
    nppiFree(buffer);
  }

  static void copy(const Npp8u *src, size_t src_pitch, Npp8u *dst, size_t dst_pitch, IuSize size)
  {
    NppStatus status;
    status = nppiCopy_8u_C4R(src, (int)src_pitch, dst, (int)dst_pitch, size.nppiSize());
    IU_ASSERT(status == NPP_NO_ERROR);
  }
};

//--------------------------------------------------------------------------
template<>
class ImageAllocatorNpp<Npp32f, 1>
{
public:
  static Npp32f* alloc(unsigned int width, unsigned int height, size_t *pitch)
  {
    IU_ASSERT(width * height > 0);

    Npp32f *buffer = nppiMalloc_32f_C1(width, height, reinterpret_cast<int *>(pitch));
    IU_ASSERT(buffer != 0);

    return buffer;
  }

  static void free(Npp32f *buffer)
  {
    nppiFree(buffer);
  }

  static void copy(const Npp32f *src, size_t src_pitch, Npp32f *dst, size_t dst_pitch, IuSize size)
  {
    NppStatus status;
    status = nppiCopy_32f_C1R(src, (int)src_pitch, dst, (int)dst_pitch, size.nppiSize());
    IU_ASSERT(status == NPP_NO_ERROR);
  }
};

//--------------------------------------------------------------------------
template<>
class ImageAllocatorNpp<Npp32f, 2>
{
public:
  static Npp32f* alloc(unsigned int width, unsigned int height, size_t *pitch)
  {
    IU_ASSERT(width * height > 0);
    Npp32f *buffer = 0;
    cudaMallocPitch((void **)&buffer, pitch, width * 2 * sizeof(Npp32f), height);
    IU_ASSERT(buffer != 0);

    return buffer;
  }

  static void free(Npp32f *buffer)
  {
    cudaFree((void*)buffer);
  }

  static void copy(const Npp32f *src, size_t src_pitch, Npp32f *dst, size_t dst_pitch, IuSize size)
  {
    cudaError_t status;
    status = cudaMemcpy2D((void *)dst, dst_pitch, (void *)src, src_pitch, size.width * 2 * sizeof(Npp32f), size.height, cudaMemcpyDeviceToDevice);
    IU_ASSERT(status == cudaSuccess);
  }
};

//--------------------------------------------------------------------------
template<>
class ImageAllocatorNpp<Npp32f, 3>
{
public:
  static Npp32f* alloc(unsigned int width, unsigned int height, size_t *pitch)
  {
    IU_ASSERT(width * height > 0);
    Npp32f *buffer = nppiMalloc_32f_C3(width, height, reinterpret_cast<int *>(pitch));
    IU_ASSERT(buffer != 0);

    return buffer;
  }

  static void free(Npp32f *buffer)
  {
    nppiFree(buffer);
  }

  static void copy(const Npp32f *src, size_t src_pitch, Npp32f *dst, size_t dst_pitch, IuSize size)
  {
    cudaError_t status;
    status = cudaMemcpy2D(dst, dst_pitch, src, src_pitch, size.width * 3 * sizeof(Npp32f), size.height, cudaMemcpyDeviceToDevice);
    IU_ASSERT(status == cudaSuccess);
  }
};

//--------------------------------------------------------------------------
template<>
class ImageAllocatorNpp<Npp32f, 4>
{
public:
  static Npp32f* alloc(unsigned int width, unsigned int height, size_t *pitch)
  {
    IU_ASSERT(width * height > 0);
    Npp32f *buffer = nppiMalloc_32f_C4(width, height, reinterpret_cast<int *>(pitch));
    IU_ASSERT(buffer != 0);

    return buffer;
  }

  static void free(Npp32f *buffer)
  {
    nppiFree(buffer);
  }

  static void copy(const Npp32f *src, size_t src_pitch, Npp32f *dst, size_t dst_pitch, IuSize size)
  {
    cudaError_t status;
    status = cudaMemcpy2D(dst, dst_pitch, src, src_pitch, size.width * 4 * sizeof(Npp32f), size.height, cudaMemcpyDeviceToDevice);
    IU_ASSERT(status == cudaSuccess);
  }
};

} // namespace iuprivate

#endif // IUCORE_IMAGE_ALLOCATOR_NPP_H
