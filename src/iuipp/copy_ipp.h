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
template<typename PixelType, unsigned int NumChannels, class Allocator>
void copy(const iu::ImageIpp<PixelType, NumChannels, Allocator> *src,
          iu::ImageIpp<PixelType, NumChannels, Allocator> *dst)
{
  Allocator::copy(src->data(), src->pitch(), dst->data(), dst->pitch(), dst->size());
}

// 2D; copy host -> device
template<typename PixelTypeSrc, typename PixelTypeDst, unsigned int NumChannels,
         class AllocatorSrc, class AllocatorDst>
void copy(const iu::ImageIpp<PixelTypeSrc, NumChannels, AllocatorSrc> *src,
          const iu::ImageGpu<PixelTypeDst, AllocatorDst> *dst)
{
  cudaError_t status;
  IU_ASSERT(sizeof(PixelTypeSrc)*NumChannels == sizeof(PixelTypeDst));
  unsigned int roi_width = dst->roi().width;
  unsigned int roi_height = dst->roi().height;
  status = cudaMemcpy2D((void*)dst->data(dst->roi().x, dst->roi().y), dst->pitch(),
                        (void*)src->data(src->roi().x, src->roi().y), src->pitch(),
                        roi_width * NumChannels * sizeof(PixelTypeSrc), roi_height,
                        cudaMemcpyHostToDevice);
  IU_ASSERT(status == cudaSuccess);
}

// 2D; copy device -> host
template<typename PixelTypeSrc, typename PixelTypeDst, unsigned int NumChannels,
         class AllocatorSrc, class AllocatorDst>
void copy(const iu::ImageGpu<PixelTypeSrc, AllocatorSrc> *src,
          const iu::ImageIpp<PixelTypeDst, NumChannels, AllocatorDst> *dst)
{
  cudaError_t status;
  if(sizeof(PixelTypeSrc) != (sizeof(PixelTypeDst)*NumChannels))
  {
    printf("PixelType sizes do not match. No copy possible.\n");
    return;
  }

  printf("pitch (src/dst) = %d / %d\n", src->pitch(), dst->pitch());
  unsigned int roi_width = dst->roi().width;
  unsigned int roi_height = dst->roi().height;
  status = cudaMemcpy2D((void*)dst->data(dst->roi().x, dst->roi().y), dst->pitch(),
                        (void*)src->data(src->roi().x, src->roi().y), src->pitch(),
                        roi_width * sizeof(PixelTypeSrc), roi_height,
                        cudaMemcpyDeviceToHost);
  IU_ASSERT(status == cudaSuccess);
}

} // namespace iuprivate

#endif // COPY_IPP_H
