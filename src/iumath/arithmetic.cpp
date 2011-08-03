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
 * Module      : Math
 * Class       : none
 * Language    : C++
 * Description : Implementation of arithmetic functions
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#include "arithmetic.cuh"
#include "arithmetic.h"

namespace iuprivate {

///////////////////////////////////////////////////////////////////////////////

// [device] weighted add; Not-in-place; 32-bit;
void addWeighted(const iu::ImageGpu_32f_C1* src1, const float& weight1,
                 const iu::ImageGpu_32f_C1* src2, const float& weight2,
                 iu::ImageGpu_32f_C1* dst, const IuRect& roi)
{
  IuStatus status;
  status = cuAddWeighted(src1, weight1, src2, weight2, dst, roi);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}

///////////////////////////////////////////////////////////////////////////////

// [gpu] multiplication with factor; Not-in-place; 8-bit; 1-channel
void mulC(const iu::ImageGpu_8u_C1* src, const unsigned char& factor, iu::ImageGpu_8u_C1* dst, const IuRect& roi)
{
  IuStatus status;
  status = cuMulC(src, factor, dst, roi);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}

// [gpu] multiplication with factor; Not-in-place; 8-bit; 4-channel
void mulC(const iu::ImageGpu_8u_C4* src, const uchar4& factor, iu::ImageGpu_8u_C4* dst, const IuRect& roi)
{
  IuStatus status;
  status = cuMulC(src, factor, dst, roi);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}

///////////////////////////////////////////////////////////////////////////////

// [gpu] multiplication with factor; Not-in-place; 32-bit; 1-channel
void mulC(const iu::ImageGpu_32f_C1* src, const float& factor, iu::ImageGpu_32f_C1* dst, const IuRect& roi)
{
  IuStatus status;
  status = cuMulC(src, factor, dst, roi);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}

// [gpu] multiplication with factor; Not-in-place; 32-bit; 2-channel
void mulC(const iu::ImageGpu_32f_C2* src, const float2& factor, iu::ImageGpu_32f_C2* dst, const IuRect& roi)
{
  IuStatus status;
  status = cuMulC(src, factor, dst, roi);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}

// [gpu] multiplication with factor; Not-in-place; 32-bit; 4-channel
void mulC(const iu::ImageGpu_32f_C4* src, const float4& factor, iu::ImageGpu_32f_C4* dst, const IuRect& roi)
{
  IuStatus status;
  status = cuMulC(src, factor, dst, roi);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}


///////////////////////////////////////////////////////////////////////////////

// [gpu] multiplication with factor; Not-in-place; 32-bit; 1-channel
void mulC(const iu::VolumeGpu_32f_C1* src, const float& factor, iu::VolumeGpu_32f_C1* dst)
{
  IuStatus status;
  status = cuMulC(src, factor, dst);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}

///////////////////////////////////////////////////////////////////////////////

// [gpu] add val; Not-in-place; 8-bit; 1-channel
void addC(const iu::ImageGpu_8u_C1* src, const unsigned char& val, iu::ImageGpu_8u_C1* dst, const IuRect& roi)
{
  IuStatus status;
  status = cuAddC(src, val, dst, roi);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}

// [gpu] add val; Not-in-place; 8-bit; 4-channel
void addC(const iu::ImageGpu_8u_C4* src, const uchar4& val, iu::ImageGpu_8u_C4* dst, const IuRect& roi)
{
  IuStatus status;
  status = cuAddC(src, val, dst, roi);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}

///////////////////////////////////////////////////////////////////////////////

// [gpu] add val; Not-in-place; 32-bit; 1-channel
void addC(const iu::ImageGpu_32f_C1* src, const float& val, iu::ImageGpu_32f_C1* dst, const IuRect& roi)
{
  IuStatus status;
  status = cuAddC(src, val, dst, roi);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}

// [gpu] add val; Not-in-place; 32-bit; 2-channel
void addC(const iu::ImageGpu_32f_C2* src, const float2& val, iu::ImageGpu_32f_C2* dst, const IuRect& roi)
{
  IuStatus status;
  status = cuAddC(src, val, dst, roi);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}

// [gpu] add val; Not-in-place; 32-bit; 4-channel
void addC(const iu::ImageGpu_32f_C4* src, const float4& val, iu::ImageGpu_32f_C4* dst, const IuRect& roi)
{
  IuStatus status;
  status = cuAddC(src, val, dst, roi);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}

} // namespace iu
