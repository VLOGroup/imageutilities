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
 * Description : Implementation of memory modifications
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#include <cstring>
#include "memory_modification.h"


namespace iuprivate {

///////////////////////////////////////////////////////////////////////////////

// [1D; host] set values; 8-bit
void setValue(const Npp8u& value, iu::LinearHostMemory_8u* srcdst)
{
  memset((void*)srcdst->data(), value, srcdst->bytes());
}

// [1D; host] set values; 32-bit
void setValue(const float& value, iu::LinearHostMemory_32f* srcdst)
{
  // we are using for loops because memset is only safe on integer type arrays

  Npp32f* buffer = srcdst->data();
  for(unsigned int i=0; i<srcdst->length(); ++i)
  {
    buffer[i] = value;
  }
}

// [1D; device] set values; 8-bit
void setValue(const Npp8u& value, iu::LinearDeviceMemory_8u* srcdst)
{
  // cudaMemset is slow so we are firing up a kernel
  NppStatus status = cuSetValue(value, srcdst);
  IU_ASSERT(status == NPP_SUCCESS);
}

// [1D; host] set values; 32-bit
void setValue(const Npp32f& value, iu::LinearDeviceMemory_32f* srcdst)
{
  // cudaMemset is slow so we are firing up a kernel
  NppStatus status = cuSetValue(value, srcdst);
  IU_ASSERT(status == NPP_SUCCESS);
}

///////////////////////////////////////////////////////////////////////////////


// [device] clamping of pixels to [min,max]; 32-bit
void clamp(const Npp32f& min, const Npp32f& max,
           iu::ImageNpp_32f_C1 *srcdst, const IuRect &roi)
{
  NppStatus status = cuClamp(min, max, srcdst, roi);
  IU_ASSERT(status == NPP_SUCCESS);
}

///////////////////////////////////////////////////////////////////////////////

// [device] conversion 32f_C3 -> 32f_C4
void convert(const iu::ImageNpp_32f_C3* src, const IuRect& src_roi, iu::ImageNpp_32f_C4* dst, const IuRect& dst_roi)
{
  NppStatus status;
  status = cuConvert(src, src_roi, dst, dst_roi);
  IU_ASSERT(status == NPP_NO_ERROR);
}

// [device] conversion 32f_C4 -> 32f_C3
void convert(const iu::ImageNpp_32f_C4* src, const IuRect& src_roi, iu::ImageNpp_32f_C3* dst, const IuRect& dst_roi)
{
  NppStatus status;
  status = cuConvert(src, src_roi, dst, dst_roi);
  IU_ASSERT(status == NPP_NO_ERROR);
}

///////////////////////////////////////////////////////////////////////////////

// [host] 2D bit depth conversion; 32f_C1 -> 8u_C1;
void convert_32f8u_C1(const iu::ImageCpu_32f_C1* src, iu::ImageCpu_8u_C1 *dst,
                      float mul_constant, float add_constant)
{
  for (unsigned int x=0; x<dst->width(); x++)
  {
    for (unsigned int y=0; y<dst->height(); y++)
    {
      Npp32f val = *src->data(x,y);
      *dst->data(x,y) = mul_constant*val + add_constant;
    }
  }
}

// [host] 2D bit depth conversion; 16u_C1 -> 32f_C1;
void convert_16u32f_C1(const iu::ImageCpu_16u_C1* src, iu::ImageCpu_32f_C1 *dst,
                       float mul_constant, float add_constant)
{
  for (unsigned int x=0; x<dst->width(); x++)
  {
    for (unsigned int y=0; y<dst->height(); y++)
    {
      Npp16u val = *src->data(x,y); //((*src->data(x,y) & 0x00ffU) << 8) | ((*src->data(x,y) & 0xff00U) >> 8);
      *dst->data(x,y) = mul_constant*(Npp32f)val + add_constant;
    }
  }
}


} // namespace iuprivate
