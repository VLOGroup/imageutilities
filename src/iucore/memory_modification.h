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
 * Description : Definition of memory modifications
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */


#ifndef IUCORE_MEMORY_MODIFICATION_H
#define IUCORE_MEMORY_MODIFICATION_H

//
//  W A R N I N G
//  -------------
//
// This file is not part of the IU API.  It exists purely as an
// implementation detail.  This header file may change from version to
// version without notice, or even be removed.
//

#include <iostream>
#include "coredefs.h"
#include "memorydefs.h"
#include "memory_modification.cuh"

namespace iuprivate {


// 1D set value; host; 8-bit
void setValue(const Npp8u& value, iu::LinearHostMemory_8u* srcdst);
void setValue(const Npp32f& value, iu::LinearHostMemory_32f* srcdst);
void setValue(const Npp8u& value, iu::LinearDeviceMemory_8u* srcdst);
void setValue(const Npp32f& value, iu::LinearDeviceMemory_32f* srcdst);

// 2D set pixel value; host;
template<typename PixelType, unsigned int NumChannels, class Allocator>
void setValue(const PixelType &value, iu::ImageCpu<PixelType, NumChannels, Allocator> *srcdst, const IuRect& roi)
{
  for(unsigned int y=0; y<srcdst->height(); ++y)
  {
    for(unsigned int x=0; x<srcdst->width(); ++x)
    {
      for(unsigned int channel=0; channel<NumChannels; channel++)
      {
        srcdst->data(x,y)[channel] = value;
      }
    }
  }
}

// 2D set pixel value; device;
template<typename PixelType, unsigned int NumChannels, class Allocator>
void setValue(const PixelType &value,
              iu::ImageNpp<PixelType, NumChannels, Allocator> *srcdst, const IuRect& roi)
{
  NppStatus status;
  status = cuSetValue(value, srcdst, roi);
  IU_ASSERT(status == NPP_SUCCESS);
}

// 2D clamping. clamps every pixel; device;
void clamp(const Npp32f& min, const Npp32f& max,
           iu::ImageNpp_32f_C1 *srcdst, const IuRect &roi);

// 2D conversion; device; 32-bit 3-channel -> 32-bit 4-channel
void convert(const iu::ImageNpp_32f_C3* src, const IuRect& src_roi, iu::ImageNpp_32f_C4* dst, const IuRect& dst_roi);

// 2D conversion; device; 32-bit 4-channel -> 32-bit 3-channel
void convert(const iu::ImageNpp_32f_C4* src, const IuRect& src_roi, iu::ImageNpp_32f_C3* dst, const IuRect& dst_roi);

// [host] 2D bit depth conversion; 32f_C1 -> 8u_C1;
void convert_32f8u_C1(const iu::ImageCpu_32f_C1* src, iu::ImageCpu_8u_C1 *dst,
                      float mul_constant=255.0f, float add_constant=0.0f);

// [host] 2D bit depth conversion; 16u_C1 -> 32f_C1;
void convert_16u32f_C1(const iu::ImageCpu_16u_C1* src, iu::ImageCpu_32f_C1 *dst,
                       float mul_constant=1.0f/65535.0f, float add_constant=0.0f);

} // namespace iuprivate

#endif // IUCORE_MEMORY_MODIFICATION_H
