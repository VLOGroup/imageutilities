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
 * Description : Definition of set value functions
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */


#ifndef IUCORE_SETVALUE_H
#define IUCORE_SETVALUE_H

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

namespace iuprivate {

/* ***************************************************************************
 *  Declaration of CUDA WRAPPERS
 * ***************************************************************************/
extern IuStatus cuSetValue(const unsigned char& value, iu::LinearDeviceMemory_8u_C1* dst);
extern IuStatus cuSetValue(const int& value, iu::LinearDeviceMemory_32s_C1* dst);
extern IuStatus cuSetValue(const float& value, iu::LinearDeviceMemory_32f_C1* dst);
extern IuStatus cuSetValue(const unsigned char& value, iu::ImageGpu_8u_C1 *dst, const IuRect &roi);
extern IuStatus cuSetValue(const uchar2& value, iu::ImageGpu_8u_C2 *dst, const IuRect &roi);
extern IuStatus cuSetValue(const uchar3& value, iu::ImageGpu_8u_C3 *dst, const IuRect &roi);
extern IuStatus cuSetValue(const uchar4& value, iu::ImageGpu_8u_C4 *dst, const IuRect &roi);
extern IuStatus cuSetValue(const int& value, iu::ImageGpu_32s_C1 *dst, const IuRect &roi);
extern IuStatus cuSetValue(const float& value, iu::ImageGpu_32f_C1 *dst, const IuRect &roi);
extern IuStatus cuSetValue(const float2& value, iu::ImageGpu_32f_C2 *dst, const IuRect &roi);
extern IuStatus cuSetValue(const float3& value, iu::ImageGpu_32f_C3 *dst, const IuRect &roi);
extern IuStatus cuSetValue(const float4& value, iu::ImageGpu_32f_C4 *dst, const IuRect &roi);
extern IuStatus cuSetValue(const unsigned char& value, iu::VolumeGpu_8u_C1 *dst, const IuCube &roi);
extern IuStatus cuSetValue(const uchar2& value, iu::VolumeGpu_8u_C2 *dst, const IuCube &roi);
extern IuStatus cuSetValue(const uchar4& value, iu::VolumeGpu_8u_C4 *dst, const IuCube &roi);
extern IuStatus cuSetValue(const float& value, iu::VolumeGpu_32f_C1 *dst, const IuCube &roi);
extern IuStatus cuSetValue(const float2& value, iu::VolumeGpu_32f_C2 *dst, const IuCube &roi);
extern IuStatus cuSetValue(const float4& value, iu::VolumeGpu_32f_C4 *dst, const IuCube &roi);
/* ***************************************************************************/


// 1D set value; host; 8-bit
void setValue(const unsigned char& value, iu::LinearHostMemory_8u_C1* srcdst);
void setValue(const uchar2& value, iu::LinearHostMemory_8u_C2* srcdst);
void setValue(const uchar3& value, iu::LinearHostMemory_8u_C3* srcdst);
void setValue(const uchar4& value, iu::LinearHostMemory_8u_C4* srcdst);

// 1D set value; host; 32-bit int
void setValue(const int& value, iu::LinearHostMemory_32s_C1* srcdst);
void setValue(const int2& value, iu::LinearHostMemory_32s_C2* srcdst);
void setValue(const int3& value, iu::LinearHostMemory_32s_C3* srcdst);
void setValue(const int4& value, iu::LinearHostMemory_32s_C4* srcdst);

// 1D set value; host; 32-bit float
void setValue(const float& value, iu::LinearHostMemory_32f_C1* srcdst);
void setValue(const float2& value, iu::LinearHostMemory_32f_C2* srcdst);
void setValue(const float3& value, iu::LinearHostMemory_32f_C3* srcdst);
void setValue(const float4& value, iu::LinearHostMemory_32f_C4* srcdst);

// 1D set value; device; 8-bit
void setValue(const unsigned char& value, iu::LinearDeviceMemory_8u_C1* srcdst);
void setValue(const uchar2& value, iu::LinearDeviceMemory_8u_C2* srcdst);
void setValue(const uchar3& value, iu::LinearDeviceMemory_8u_C3* srcdst);
void setValue(const uchar4& value, iu::LinearDeviceMemory_8u_C4* srcdst);

// 1D set value; device; 32-bit int
void setValue(const int& value, iu::LinearDeviceMemory_32s_C1* srcdst);
void setValue(const int2& value, iu::LinearDeviceMemory_32s_C2* srcdst);
void setValue(const int3& value, iu::LinearDeviceMemory_32s_C3* srcdst);
void setValue(const int4& value, iu::LinearDeviceMemory_32s_C4* srcdst);

// 1D set value; device; 32-bit float
void setValue(const float& value, iu::LinearDeviceMemory_32f_C1* srcdst);
void setValue(const float2& value, iu::LinearDeviceMemory_32f_C2* srcdst);
void setValue(const float3& value, iu::LinearDeviceMemory_32f_C3* srcdst);
void setValue(const float4& value, iu::LinearDeviceMemory_32f_C4* srcdst);

// 2D set pixel value; host;
template<typename PixelType, class Allocator>
inline void setValue(const PixelType &value, iu::ImageCpu<PixelType, Allocator> *srcdst,
                     const IuRect& roi)
{
  for(unsigned int y=roi.y; y<roi.height; ++y)
  {
    for(unsigned int x=roi.x; x<roi.width; ++x)
    {
      *srcdst->data(x,y) = value;
    }
  }
}

// 3D set pixel value; host;
template<typename PixelType, class Allocator>
inline void setValue(const PixelType &value, iu::VolumeCpu<PixelType, Allocator> *srcdst,
                     const IuCube& roi)
{
  for(unsigned int z=roi.z; z<roi.depth; ++z)
  {
    for(unsigned int y=roi.y; y<roi.height; ++y)
    {
      for(unsigned int x=roi.x; x<roi.width; ++x)
      {
        *srcdst->data(x,y,z) = value;
      }
    }
  }
}

// 2D set pixel value; device;
template<typename PixelType, class Allocator, IuPixelType _pixel_type>
void setValue(const PixelType &value, iu::ImageGpu<PixelType, Allocator, _pixel_type> *srcdst, const IuRect& roi)
{
  IuStatus status;
  status = cuSetValue(value, srcdst, roi);
  IU_ASSERT(status == IU_SUCCESS);
}

// 3D set pixel value; device;
template<typename PixelType, class Allocator>
void setValue(const PixelType &value, iu::VolumeGpu<PixelType, Allocator> *srcdst, const IuCube& roi)
{
  IuStatus status;
  status = cuSetValue(value, srcdst, roi);
  IU_ASSERT(status == IU_SUCCESS);
}


} // namespace iuprivate

#endif // IUCORE_SETVALUE_H
