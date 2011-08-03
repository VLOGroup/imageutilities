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
  if (status != cudaSuccess) throw IuException("cudaMemcpy returned error code", __FILE__, __FUNCTION__, __LINE__);
}

// 1D; copy host -> device
template <typename PixelType>
void copy(const iu::LinearHostMemory<PixelType> *src, iu::LinearDeviceMemory<PixelType> *dst)
{
  cudaError_t status;
  status = cudaMemcpy(dst->data(), src->data(), dst->length() * sizeof(PixelType), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) throw IuException("cudaMemcpy returned error code", __FILE__, __FUNCTION__, __LINE__);
}

// 1D; copy device -> host
template <typename PixelType>
void copy(const iu::LinearDeviceMemory<PixelType> *src, iu::LinearHostMemory<PixelType> *dst)
{
  cudaError_t status;
  status = cudaMemcpy(dst->data(), src->data(), dst->length() * sizeof(PixelType), cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) throw IuException("cudaMemcpy returned error code", __FILE__, __FUNCTION__, __LINE__);
}

/* ****************************************************************************
 *
 * 2D copy
 *
 **************************************************************************** */

// 2D; copy host -> host
template<typename PixelType, class Allocator, IuPixelType _pixel_type>
void copy(const iu::ImageCpu<PixelType, Allocator, _pixel_type> *src,
          iu::ImageCpu<PixelType, Allocator, _pixel_type> *dst)
{
  Allocator::copy(src->data(), src->pitch(), dst->data(), dst->pitch(), dst->size());
}

// 2D; copy device -> device
template<typename PixelType, class Allocator, IuPixelType _pixel_type>
void copy(const iu::ImageGpu<PixelType, Allocator, _pixel_type> *src,
          iu::ImageGpu<PixelType, Allocator, _pixel_type> *dst)
{
  Allocator::copy(src->data(), src->pitch(), dst->data(), dst->pitch(), dst->size());
}

// 2D; copy host -> device
template<typename PixelType, class AllocatorCpu, class AllocatorGpu, IuPixelType _pixel_type>
void copy(const iu::ImageCpu<PixelType, AllocatorCpu, _pixel_type> *src,
          iu::ImageGpu<PixelType, AllocatorGpu, _pixel_type> *dst)
{
  cudaError_t status;
  unsigned int roi_width = dst->roi().width;
  unsigned int roi_height = dst->roi().height;
  status = cudaMemcpy2D(dst->data(dst->roi().x, dst->roi().y), dst->pitch(),
                        src->data(src->roi().x, src->roi().y), src->pitch(),
                        roi_width * sizeof(PixelType), roi_height,
                        cudaMemcpyHostToDevice);
  if (status != cudaSuccess) throw IuException("cudaMemcpy2D returned error code", __FILE__, __FUNCTION__, __LINE__);
}

// 2D; copy device -> host
template<typename PixelType, class AllocatorGpu, class AllocatorCpu, IuPixelType _pixel_type>
void copy(const iu::ImageGpu<PixelType, AllocatorGpu, _pixel_type> *src,
          iu::ImageCpu<PixelType, AllocatorCpu, _pixel_type> *dst)
{
  cudaError_t status;
  unsigned int roi_width = dst->roi().width;
  unsigned int roi_height = dst->roi().height;
  status = cudaMemcpy2D(dst->data(dst->roi().x, dst->roi().y), dst->pitch(),
                        src->data(src->roi().x, src->roi().y), src->pitch(),
                        roi_width * sizeof(PixelType), roi_height,
                        cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) throw IuException("cudaMemcpy2D returned error code", __FILE__, __FUNCTION__, __LINE__);
}

/* ****************************************************************************
 *
 * 3D copy
 *
 **************************************************************************** */

// 3D; copy host -> host
template<typename PixelType, class Allocator, IuPixelType _pixel_type>
void copy(const iu::VolumeCpu<PixelType, Allocator, _pixel_type> *src,
          iu::VolumeCpu<PixelType, Allocator, _pixel_type> *dst)
{
  Allocator::copy(src->data(), src->pitch(), dst->data(), dst->pitch(), dst->size());
}

// 3D; copy device -> device
template<typename PixelType, class Allocator, IuPixelType _pixel_type>
void copy(const iu::VolumeGpu<PixelType, Allocator, _pixel_type> *src,
          iu::VolumeGpu<PixelType, Allocator, _pixel_type> *dst)
{
  Allocator::copy(src->data(), src->pitch(), dst->data(), dst->pitch(), dst->size());
}

// 3D; copy host -> device
template<typename PixelType, class AllocatorCpu, class AllocatorGpu, IuPixelType _pixel_type>
void copy(const iu::VolumeCpu<PixelType, AllocatorCpu, _pixel_type> *src,
          iu::VolumeGpu<PixelType, AllocatorGpu, _pixel_type> *dst)
{
  cudaError_t status;
  unsigned int roi_width = dst->roi().width;
  unsigned int roi_height = dst->roi().height;
  unsigned int roi_depth =  dst->roi().depth;
  status = cudaMemcpy2D(dst->data(dst->roi().x, dst->roi().y, dst->roi().z), dst->pitch(),
                        src->data(src->roi().x, src->roi().y, dst->roi().z), src->pitch(),
                        roi_width * sizeof(PixelType), roi_height*roi_depth,
                        cudaMemcpyHostToDevice);
  if (status != cudaSuccess) throw IuException("cudaMemcpy2D returned error code", __FILE__, __FUNCTION__, __LINE__);
}

// 3D; copy device -> host
template<typename PixelType, class AllocatorGpu, class AllocatorCpu, IuPixelType _pixel_type>
void copy(const iu::VolumeGpu<PixelType, AllocatorGpu, _pixel_type> *src,
          iu::VolumeCpu<PixelType, AllocatorCpu, _pixel_type> *dst)
{
  cudaError_t status;
  unsigned int roi_width = dst->roi().width;
  unsigned int roi_height = dst->roi().height;
  unsigned int roi_depth =  dst->roi().depth;
  status = cudaMemcpy2D(dst->data(dst->roi().x, dst->roi().y, dst->roi().z), dst->pitch(),
                        src->data(src->roi().x, src->roi().y, dst->roi().z), src->pitch(),
                        roi_width * sizeof(PixelType), roi_height*roi_depth,
                        cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) throw IuException("cudaMemcpy2D returned error code", __FILE__, __FUNCTION__, __LINE__);
}


} // namespace iuprivate

#endif // IUCORE_COPY_H
