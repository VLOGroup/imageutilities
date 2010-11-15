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

// TODO copy: problems with channels and vector types!c
//// 2D; copy host -> device
//template<typename PixelType, unsigned int NumChannels, class AllocatorIpp, class AllocatorNpp>
//void copy(const iu::ImageIpp<PixelType, NumChannels, AllocatorIpp> *src,
//          iu::ImageGpu<PixelType, NumChannels, AllocatorNpp> *dst)
//{
//  cudaError_t status;
//  unsigned int roi_width = dst->roi().width;
//  unsigned int roi_height = dst->roi().height;
//  status = cudaMemcpy2D(dst->data(dst->roi().x, dst->roi().y), dst->pitch(),
//                        src->data(src->roi().x, src->roi().y), src->pitch(),
//                        roi_width * NumChannels * sizeof(PixelType), roi_height,
//                        cudaMemcpyHostToDevice);
//  IU_ASSERT(status == cudaSuccess);
//}

//// 2D; copy device -> host
//template<typename PixelType, unsigned int NumChannels, class AllocatorNpp, class AllocatorIpp>
//void copy(const iu::ImageGpu<PixelType, NumChannels, AllocatorNpp> *src,
//          iu::ImageIpp<PixelType, NumChannels, AllocatorIpp> *dst)
//{
//  cudaError_t status;
//  unsigned int roi_width = dst->roi().width;
//  unsigned int roi_height = dst->roi().height;
//  status = cudaMemcpy2D(dst->data(dst->roi().x, dst->roi().y), dst->pitch(),
//                        src->data(src->roi().x, src->roi().y), src->pitch(),
//                        roi_width * NumChannels * sizeof(PixelType), roi_height,
//                        cudaMemcpyDeviceToHost);
//  IU_ASSERT(status == cudaSuccess);
//}

} // namespace iuprivate

#endif // COPY_IPP_H
