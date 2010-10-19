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
 * Project     : VMLibraries
 * Module      : ImageUtilities; Matlab connector
 * Class       : none
 * Language    : C
 * Description : Interface for Matlab and iu memory layouts
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */


#ifndef IUPRIVATE_IUMATLABCONNECTOR_H
#define IUPRIVATE_IUMATLABCONNECTOR_H

//
//  W A R N I N G
//  -------------
//
// This file is not part of the IU API.  It exists purely as an
// implementation detail.  This header file may change from version to
// version without notice, or even be removed.
//

#include <iostream>
#include <iudefs.h>
#include <iucutil.h>
#include <iucore/memorydefs.h>

namespace iuprivate {

//-----------------------------------------------------------------------------
// [host] conversion from matlab to ImageCpu memory layout
template<typename PixelType, class Allocator>
IuStatus convertMatlabToCpu(double* matlab_src_buffer, unsigned int width, unsigned int height,
                            iu::ImageCpu<PixelType, Allocator> *dst)

{
  if(width > dst->width() || height > dst->height())
  {
    std::cerr << "Error in convertMatlabToCpu: memory dimensions mismatch!" << std::endl;
    return IU_MEM_COPY_ERROR;
  }

  // iterate over the smaller block of input and output
  for (unsigned int y = iu::max(0, dst->roi().y); y<iu::min(height, dst->roi().height); ++y)
  {
    for (unsigned int x = iu::max(0, dst->roi().x); x<iu::min(width, dst->roi().width); ++x)
    {
      *dst->data(x,y) = static_cast<float>(matlab_src_buffer[y + x*height]);
    }
  }

  return IU_NO_ERROR;
}

//-----------------------------------------------------------------------------
// [device] conversion from matlab to ImageGpu memory layout
template<typename PixelType, class Allocator>
IuStatus convertMatlabToGpu(double* matlab_src_buffer, unsigned int width, unsigned int height,
                            iu::ImageGpu<PixelType, Allocator> *dst)
{
  iu::ImageCpu_32f_C1 tmp_cpu(dst->size());
  tmp_cpu.roi() = dst->roi();

  IuStatus status = convertMatlabToCpu(matlab_src_buffer, width, height, &tmp_cpu);
  if(status != IU_SUCCESS)
    return status;

  iu::copy(&tmp_cpu, dst);
  return IU_NO_ERROR;
}

//-----------------------------------------------------------------------------
// [host] conversion from ImageCpu to matlab memory layout
template<typename PixelType, class Allocator>
IuStatus convertCpuToMatlab(iu::ImageCpu<PixelType, Allocator> *src,
                            double* matlab_dst_buffer, unsigned int width, unsigned int height)
{
  if(width > src->width() || height > src->height())
  {
    std::cerr << "Error in convertMatlabToCpu: memory dimensions mismatch!" << std::endl;
    return IU_MEM_COPY_ERROR;
  }

  // iterate over the smaller block of input and output
  for (unsigned int y = iu::max(0, src->roi().y); y<iu::min(height, src->roi().height); ++y)
  {
    for (unsigned int x = iu::max(0, src->roi().x); x<iu::min(width, src->roi().width); ++x)
    {
      matlab_dst_buffer[y + x*height] = static_cast<double>(*src->data(x,y));
    }
  }

  return IU_NO_ERROR;
}

//-----------------------------------------------------------------------------
// [device] conversion from matlab to ImageGpu memory layout
template<typename PixelType, class Allocator>
IuStatus convertGpuToMatlab(iu::ImageGpu<PixelType, Allocator> *src,
                            double* matlab_dst_buffer, unsigned int width, unsigned int height)
{
  iu::ImageCpu_32f_C1 tmp_cpu(src->size());
  tmp_cpu.roi() = src->roi();
  iu::copy(src, &tmp_cpu);

  IuStatus status = iuprivate::convertCpuToMatlab(&tmp_cpu, matlab_dst_buffer, width, height);
  if(status != IU_SUCCESS)
    return status;

  return IU_NO_ERROR;
}

} // namespace iuprivate


#endif // IUPRIVATE_IUMATLABCONNECTOR_H
