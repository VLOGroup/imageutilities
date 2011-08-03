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
 * Module      : IPP-Connector
 * Class       : none
 * Language    : C
 * Description : Definition of copy functions for the IPP connector
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef COPY_IPP_H
#define COPY_IPP_H

//
//  W A R N I N G
//  -------------
//
// This file is not part of the IU API.  It exists purely as an
// implementation detail.  This header file may change from version to
// version without notice, or even be removed.
//

#include <iucore/coredefs.h>
#include <iucore/memorydefs.h>
#include "memorydefs_ipp.h"

namespace iu {

// 2D; copy host -> host
template<typename PixelType, unsigned int NumChannels, class Allocator, IuPixelType _pixel_type>
void copy(const iu::ImageIpp<PixelType, NumChannels, Allocator, _pixel_type> *src,
          iu::ImageIpp<PixelType, NumChannels, Allocator, _pixel_type> *dst)
{
  Allocator::copy(src->data(), src->pitch(), dst->data(), dst->pitch(), dst->size());
}

// 2D; copy host -> device
template<typename PixelTypeSrc, typename PixelTypeDst, unsigned int NumChannels,
         class AllocatorSrc, class AllocatorDst, IuPixelType _pixel_type>
void copy(const iu::ImageIpp<PixelTypeSrc, NumChannels, AllocatorSrc, _pixel_type> *src,
          const iu::ImageGpu<PixelTypeDst, AllocatorDst, _pixel_type> *dst)
{
  cudaError_t status;
  if (sizeof(PixelTypeSrc)*NumChannels != sizeof(PixelTypeDst))
    throw IuException("Source and destination pixel type do not macht.", __FILE__, __FUNCTION__, __LINE__);
  unsigned int roi_width = dst->roi().width;
  unsigned int roi_height = dst->roi().height;
  status = cudaMemcpy2D((void*)dst->data(dst->roi().x, dst->roi().y), dst->pitch(),
                        (void*)src->data(src->roi().x, src->roi().y), src->pitch(),
                        roi_width * NumChannels * sizeof(PixelTypeSrc), roi_height,
                        cudaMemcpyHostToDevice);
  if (status != cudaSuccess) throw IuException("cudaMemcpy2D returned error code", __FILE__, __FUNCTION__, __LINE__);
}

// 2D; copy device -> host
template<typename PixelTypeSrc, typename PixelTypeDst, unsigned int NumChannels,
         class AllocatorSrc, class AllocatorDst, IuPixelType _pixel_type>
void copy(const iu::ImageGpu<PixelTypeSrc, AllocatorSrc, _pixel_type> *src,
          const iu::ImageIpp<PixelTypeDst, NumChannels, AllocatorDst, _pixel_type> *dst)
{
  cudaError_t status;
  if (sizeof(PixelTypeSrc) != sizeof(PixelTypeDst)*NumChannels)
    throw IuException("Source and destination pixel type do not macht.", __FILE__, __FUNCTION__, __LINE__);

  unsigned int roi_width = dst->roi().width;
  unsigned int roi_height = dst->roi().height;
  status = cudaMemcpy2D((void*)dst->data(dst->roi().x, dst->roi().y), dst->pitch(),
                        (void*)src->data(src->roi().x, src->roi().y), src->pitch(),
                        roi_width * sizeof(PixelTypeSrc), roi_height,
                        cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) throw IuException("cudaMemcpy2D returned error code", __FILE__, __FUNCTION__, __LINE__);
}

} // namespace iuprivate

#endif // COPY_IPP_H
