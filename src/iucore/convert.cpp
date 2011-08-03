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
 * Description : Implementation of convert operations on host and device memory
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#include <cstring>
#include "convert.h"


namespace iuprivate {

/* ***************************************************************************
 *  Declaration of CUDA WRAPPERS
 * ***************************************************************************/
extern IuStatus cuConvert(const iu::ImageGpu_32f_C3* src, const IuRect& src_roi,
                          iu::ImageGpu_32f_C4* dst, const IuRect& dst_roi);
extern IuStatus cuConvert(const iu::ImageGpu_32f_C4* src, const IuRect& src_roi,
                          iu::ImageGpu_32f_C3* dst, const IuRect& dst_roi);
extern IuStatus cuConvert_8u_32f(const iu::ImageGpu_8u_C1* src, const IuRect& src_roi,
                                 iu::ImageGpu_32f_C1* dst, const IuRect& dst_roi,
                                 float mul_constant,  float add_constant);
extern IuStatus cuConvert_32f_8u(const iu::ImageGpu_32f_C1* src, const IuRect& src_roi,
                                 iu::ImageGpu_8u_C1* dst, const IuRect& dst_roi,
                                 float mul_constant, unsigned char add_constant);
extern IuStatus cuConvert_32f_8u(const iu::ImageGpu_32f_C4* src, const IuRect& src_roi,
                                 iu::ImageGpu_8u_C4* dst, const IuRect& dst_roi,
                                 float mul_constant, unsigned char add_constant);
extern IuStatus cuConvert_rgb_to_hsv(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst, bool normalize);
extern IuStatus cuConvert_hsv_to_rgb(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst, bool denormalize);
/* ***************************************************************************/



/* ***************************************************************************
 *  FUNCTION IMPLEMENTATIONS
 * ***************************************************************************/

//-----------------------------------------------------------------------------
// [host] 2D bit depth conversion; 32f_C1 -> 8u_C1;
void convert_32f8u_C1(const iu::ImageCpu_32f_C1* src, iu::ImageCpu_8u_C1 *dst,
                      float mul_constant, float add_constant)
{
  for (unsigned int x=0; x<dst->width(); x++)
  {
    for (unsigned int y=0; y<dst->height(); y++)
    {
      float val = *src->data(x,y);
      *dst->data(x,y) = mul_constant*val + add_constant;
    }
  }
}

//-----------------------------------------------------------------------------
// [host] 2D bit depth conversion; 16u_C1 -> 32f_C1;
void convert_16u32f_C1(const iu::ImageCpu_16u_C1* src, iu::ImageCpu_32f_C1 *dst,
                       float mul_constant, float add_constant)
{
  for (unsigned int x=0; x<dst->width(); x++)
  {
    for (unsigned int y=0; y<dst->height(); y++)
    {
      unsigned short val = *src->data(x,y); //((*src->data(x,y) & 0x00ffU) << 8) | ((*src->data(x,y) & 0xff00U) >> 8);
      *dst->data(x,y) = mul_constant*(float)val + add_constant;
    }
  }
}

//-----------------------------------------------------------------------------
// [device] conversion 32f_C1 -> 8u_C1
void convert_32f8u_C1(const iu::ImageGpu_32f_C1* src, const IuRect& src_roi, iu::ImageGpu_8u_C1 *dst, const IuRect& dst_roi,
                      float mul_constant, unsigned char add_constant)
{
  IuStatus status;
  status = cuConvert_32f_8u(src, src_roi, dst, dst_roi, mul_constant, add_constant);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}

//-----------------------------------------------------------------------------
// [device] conversion 32f_C4 -> 8u_C4
void convert_32f8u_C4(const iu::ImageGpu_32f_C4* src, const IuRect& src_roi, iu::ImageGpu_8u_C4 *dst, const IuRect& dst_roi,
                      float mul_constant, unsigned char add_constant)
{
  IuStatus status;
  status = cuConvert_32f_8u(src, src_roi, dst, dst_roi, mul_constant, add_constant);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}



//-----------------------------------------------------------------------------
// [device] conversion 8u_C1 -> 32f_C1
void convert_8u32f_C1(const iu::ImageGpu_8u_C1* src, const IuRect& src_roi, iu::ImageGpu_32f_C1 *dst, const IuRect& dst_roi,
                      float mul_constant, float add_constant)
{
  IuStatus status;
  status = cuConvert_8u_32f(src, src_roi, dst, dst_roi, mul_constant, add_constant);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}


//-----------------------------------------------------------------------------
// [device] conversion 32f_C3 -> 32f_C4
void convert(const iu::ImageGpu_32f_C3* src, const IuRect& src_roi, iu::ImageGpu_32f_C4* dst, const IuRect& dst_roi)
{
  IuStatus status;
  status = cuConvert(src, src_roi, dst, dst_roi);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}

//-----------------------------------------------------------------------------
// [device] conversion 32f_C4 -> 32f_C3
void convert(const iu::ImageGpu_32f_C4* src, const IuRect& src_roi, iu::ImageGpu_32f_C3* dst, const IuRect& dst_roi)
{
  IuStatus status;
  status = cuConvert(src, src_roi, dst, dst_roi);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}

//-----------------------------------------------------------------------------
// [device] conversion RGB -> HSV
void convertRgbHsv(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst, bool normalize)
{
  IuStatus status;
  status = cuConvert_rgb_to_hsv(src, dst, normalize);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}

//-----------------------------------------------------------------------------
// [device] conversion HSV -> RGB
void convertHsvRgb(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst, bool denormalize)
{
  IuStatus status;
  status = cuConvert_hsv_to_rgb(src, dst, denormalize);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}

} // namespace iuprivate
