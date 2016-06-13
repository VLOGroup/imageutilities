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
#include "../iucutil.h"

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
  IU_SIZE_CHECK(src, dst);
  memcpy(dst->data(), src->data(), dst->length() * sizeof(PixelType));
}

// 1D; copy device -> device
template <typename PixelType>
void copy(const iu::LinearDeviceMemory<PixelType> *src, iu::LinearDeviceMemory<PixelType> *dst)
{
  IU_SIZE_CHECK(src, dst);
  IU_CUDA_SAFE_CALL(cudaMemcpy(dst->data(), src->data(), dst->length() * sizeof(PixelType), cudaMemcpyDeviceToDevice));
}

// 1D; copy host -> device
template <typename PixelType>
void copy(const iu::LinearHostMemory<PixelType> *src, iu::LinearDeviceMemory<PixelType> *dst)
{
  IU_SIZE_CHECK(src, dst);
  IU_CUDA_SAFE_CALL(cudaMemcpy(dst->data(), src->data(), dst->length() * sizeof(PixelType), cudaMemcpyHostToDevice));
}

// 1D; copy device -> host
template <typename PixelType>
void copy(const iu::LinearDeviceMemory<PixelType> *src, iu::LinearHostMemory<PixelType> *dst)
{
  IU_SIZE_CHECK(src, dst);
  IU_CUDA_SAFE_CALL(cudaMemcpy(dst->data(), src->data(), dst->length() * sizeof(PixelType), cudaMemcpyDeviceToHost));
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
  IU_SIZE_CHECK(src, dst);
  Allocator::copy(src->data(), src->pitch(), dst->data(), dst->pitch(), dst->size());
}

// 2D; copy device -> device
template<typename PixelType, class Allocator, IuPixelType _pixel_type>
void copy(const iu::ImageGpu<PixelType, Allocator, _pixel_type> *src,
          iu::ImageGpu<PixelType, Allocator, _pixel_type> *dst)
{
  IU_SIZE_CHECK(src, dst);
  Allocator::copy(src->data(), src->pitch(), dst->data(), dst->pitch(), dst->size());
}

// 2D; copy host -> device
template<typename PixelType, class AllocatorCpu, class AllocatorGpu, IuPixelType _pixel_type>
void copy(const iu::ImageCpu<PixelType, AllocatorCpu, _pixel_type> *src,
          iu::ImageGpu<PixelType, AllocatorGpu, _pixel_type> *dst)
{
  IU_SIZE_CHECK(src, dst);
  IU_CUDA_SAFE_CALL(cudaMemcpy2D(dst->data(), dst->pitch(),
                        src->data(), src->pitch(),
                        src->width() * sizeof(PixelType), src->height(),
                        cudaMemcpyHostToDevice));
}

// 2D; copy device -> host
template<typename PixelType, class AllocatorGpu, class AllocatorCpu, IuPixelType _pixel_type>
void copy(const iu::ImageGpu<PixelType, AllocatorGpu, _pixel_type> *src,
          iu::ImageCpu<PixelType, AllocatorCpu, _pixel_type> *dst)
{
  IU_SIZE_CHECK(src, dst);
  IU_CUDA_SAFE_CALL(cudaMemcpy2D(dst->data(), dst->pitch(),
                        src->data(), src->pitch(),
                        src->width() * sizeof(PixelType), src->height(),
                        cudaMemcpyDeviceToHost));
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
  IU_SIZE_CHECK(src, dst);
  Allocator::copy(src->data(), src->pitch(), dst->data(), dst->pitch(), dst->size());
}

// 3D; copy device -> device
template<typename PixelType, class Allocator, IuPixelType _pixel_type>
void copy(const iu::VolumeGpu<PixelType, Allocator, _pixel_type> *src,
          iu::VolumeGpu<PixelType, Allocator, _pixel_type> *dst)
{
  IU_SIZE_CHECK(src, dst);
  Allocator::copy(src->data(), src->pitch(), dst->data(), dst->pitch(), dst->size());
}

// 3D; copy host -> device
template<typename PixelType, class AllocatorCpu, class AllocatorGpu, IuPixelType _pixel_type>
void copy(const iu::VolumeCpu<PixelType, AllocatorCpu, _pixel_type> *src,
          iu::VolumeGpu<PixelType, AllocatorGpu, _pixel_type> *dst)
{
  IU_SIZE_CHECK(src, dst);
  IU_CUDA_SAFE_CALL(cudaMemcpy2D(dst->data(), dst->pitch(),
                        src->data(), src->pitch(),
                        src->width() * sizeof(PixelType), src->height()*src->depth(),
                        cudaMemcpyHostToDevice));
}

// 3D; copy device -> host
template<typename PixelType, class AllocatorGpu, class AllocatorCpu, IuPixelType _pixel_type>
void copy(const iu::VolumeGpu<PixelType, AllocatorGpu, _pixel_type> *src,
          iu::VolumeCpu<PixelType, AllocatorCpu, _pixel_type> *dst)
{
  IU_SIZE_CHECK(src, dst);
  IU_CUDA_SAFE_CALL(cudaMemcpy2D(dst->data(), dst->pitch(),
                        src->data(), src->pitch(),
                        src->width() * sizeof(PixelType), src->height()*src->depth(),
                        cudaMemcpyDeviceToHost));
}

template<typename PixelType, class AllocatorCpu, IuPixelType _pixel_type>
void copy(const iu::ImageCpu<PixelType, AllocatorCpu, _pixel_type> *src, iu::LinearHostMemory<PixelType> *dst)
{
  IU_SIZE_CHECK(src, dst);
	PixelType *dstData = dst->data();
	for(int y = 0; y < src->height(); ++y)
	{
		for(int x = 0; x < src->width(); ++x)
		{
			dstData[x + y * src->width()] = *(src->data(x, y));
		}
	}
}

// only declaration
void copy(const iu::ImageGpu_32f_C1* src, iu::LinearDeviceMemory_32f_C1* dst);


} // namespace iuprivate

#endif // IUCORE_COPY_H
