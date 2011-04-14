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
 * Class       : none
 * Language    : C
 * Description : Definition of copy functions
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUCORE_COPY_H
#define IUCORE_COPY_H

//
//  W A R N I N G
//  -------------
//
// This file is not part of the IU API.  It exists purely as an
// implementation detail.  This header file may change from version to
// version without notice, or even be removed.
//

#include "coredefs.h"
#include "memorydefs.h"

namespace iuprivate {

/* ****************************************************************************
 *
 * 1D copy
 *
 **************************************************************************** */

// 1D; copy host -> host
template <typename PixelType>
void copy(const iu::LinearHostMemory<PixelType> *src, iu::LinearHostMemory<PixelType> *dst)
{
  memcpy(dst->data(), src->data(), dst->length() * sizeof(PixelType));
}

// 1D; copy device -> device
template <typename PixelType>
void copy(const iu::LinearDeviceMemory<PixelType> *src, iu::LinearDeviceMemory<PixelType> *dst)
{
  cudaError_t status;
  status = cudaMemcpy(dst->data(), src->data(), dst->length() * sizeof(PixelType), cudaMemcpyDeviceToDevice);
  IU_ASSERT(status == cudaSuccess);
}

// 1D; copy host -> device
template <typename PixelType>
void copy(const iu::LinearHostMemory<PixelType> *src, iu::LinearDeviceMemory<PixelType> *dst)
{
  cudaError_t status;
  status = cudaMemcpy(dst->data(), src->data(), dst->length() * sizeof(PixelType), cudaMemcpyHostToDevice);
  IU_ASSERT(status == cudaSuccess);
}

// 1D; copy device -> host
template <typename PixelType>
void copy(const iu::LinearDeviceMemory<PixelType> *src, iu::LinearHostMemory<PixelType> *dst)
{
  cudaError_t status;
  status = cudaMemcpy(dst->data(), src->data(), dst->length() * sizeof(PixelType), cudaMemcpyDeviceToHost);
  IU_ASSERT(status == cudaSuccess);
}

/* ****************************************************************************
 *
 * 2D copy
 *
 **************************************************************************** */

// 2D; copy host -> host
template<typename PixelType, class Allocator>
void copy(const iu::ImageCpu<PixelType, Allocator> *src, iu::ImageCpu<PixelType, Allocator> *dst)
{
  Allocator::copy(src->data(), src->pitch(), dst->data(), dst->pitch(), dst->size());
}

// 2D; copy device -> device
template<typename PixelType, class Allocator>
void copy(const iu::ImageGpu<PixelType, Allocator> *src, iu::ImageGpu<PixelType, Allocator> *dst)
{
  Allocator::copy(src->data(), src->pitch(), dst->data(), dst->pitch(), dst->size());
}

// 2D; copy host -> device
template<typename PixelType, class AllocatorCpu, class AllocatorGpu>
void copy(const iu::ImageCpu<PixelType, AllocatorCpu> *src, iu::ImageGpu<PixelType, AllocatorGpu> *dst)
{
  cudaError_t status;
  unsigned int roi_width = dst->roi().width;
  unsigned int roi_height = dst->roi().height;
  status = cudaMemcpy2D(dst->data(dst->roi().x, dst->roi().y), dst->pitch(),
                        src->data(src->roi().x, src->roi().y), src->pitch(),
                        roi_width * sizeof(PixelType), roi_height,
                        cudaMemcpyHostToDevice);
  IU_ASSERT(status == cudaSuccess);
}

// 2D; copy device -> host
template<typename PixelType, class AllocatorGpu, class AllocatorCpu>
void copy(const iu::ImageGpu<PixelType, AllocatorGpu> *src, iu::ImageCpu<PixelType, AllocatorCpu> *dst)
{
  cudaError_t status;
  unsigned int roi_width = dst->roi().width;
  unsigned int roi_height = dst->roi().height;
  status = cudaMemcpy2D(dst->data(dst->roi().x, dst->roi().y), dst->pitch(),
                        src->data(src->roi().x, src->roi().y), src->pitch(),
                        roi_width * sizeof(PixelType), roi_height,
                        cudaMemcpyDeviceToHost);
  IU_ASSERT(status == cudaSuccess);
}

/* ****************************************************************************
 *
 * 3D copy
 *
 **************************************************************************** */

// 3D; copy host -> host
template<typename PixelType, class Allocator>
void copy(const iu::VolumeCpu<PixelType, Allocator> *src,
          iu::VolumeCpu<PixelType, Allocator> *dst)
{
  Allocator::copy(src->data(), src->pitch(), dst->data(), dst->pitch(), dst->size());
}

// 3D; copy device -> device
template<typename PixelType, class Allocator>
void copy(const iu::VolumeGpu<PixelType, Allocator> *src,
          iu::VolumeGpu<PixelType, Allocator> *dst)
{
  Allocator::copy(src->data(), src->pitch(), dst->data(), dst->pitch(), dst->size());
}

// 3D; copy host -> device
template<typename PixelType, class AllocatorCpu, class AllocatorGpu>
void copy(const iu::VolumeCpu<PixelType, AllocatorCpu> *src,
          iu::VolumeGpu<PixelType, AllocatorGpu> *dst)
{
  cudaError_t status;
  unsigned int roi_width = dst->roi().width;
  unsigned int roi_height = dst->roi().height;
  unsigned int roi_depth =  dst->roi().depth;
  status = cudaMemcpy2D(dst->data(dst->roi().x, dst->roi().y, dst->roi().z), dst->pitch(),
                        src->data(src->roi().x, src->roi().y, dst->roi().z), src->pitch(),
                        roi_width * sizeof(PixelType), roi_height*roi_depth,
                        cudaMemcpyHostToDevice);
  IU_ASSERT(status == cudaSuccess);
}

// 3D; copy device -> host
template<typename PixelType, class AllocatorGpu, class AllocatorCpu>
void copy(const iu::VolumeGpu<PixelType, AllocatorGpu> *src,
          iu::VolumeCpu<PixelType, AllocatorCpu> *dst)
{
  cudaError_t status;
  unsigned int roi_width = dst->roi().width;
  unsigned int roi_height = dst->roi().height;
  unsigned int roi_depth =  dst->roi().depth;
  status = cudaMemcpy2D(dst->data(dst->roi().x, dst->roi().y, dst->roi().z), dst->pitch(),
                        src->data(src->roi().x, src->roi().y, dst->roi().z), src->pitch(),
                        roi_width * sizeof(PixelType), roi_height*roi_depth,
                        cudaMemcpyDeviceToHost);
  IU_ASSERT(status == cudaSuccess);
}


} // namespace iuprivate

#endif // IUCORE_COPY_H
