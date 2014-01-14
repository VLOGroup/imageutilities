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
#include <iucore/copy.h>

namespace iuprivate {

///////////////////////////////////////////////////////////////////////////////

// [device] weighted add; Not-in-place; 32-bit;
void addWeighted(const iu::ImageGpu_32f_C1* src1, const float& weight1,
                 const iu::ImageGpu_32f_C1* src2, const float& weight2,
                 iu::ImageGpu_32f_C1* dst, const IuRect& roi)
{
  cuAddWeighted(src1, weight1, src2, weight2, dst, roi);
}
// [host] weighted add; Not-in-place; 32-bit;
void addWeighted(const iu::ImageCpu_32f_C1* src1, const float& weight1,
                 const iu::ImageCpu_32f_C1* src2, const float& weight2,
                 iu::ImageCpu_32f_C1* dst, const IuRect& roi)
{
  iu::ImageGpu_32f_C1 d_src1(src1->size());
  iu::ImageGpu_32f_C1 d_src2(src1->size());
  iu::ImageGpu_32f_C1 d_dst(dst->size());

  iuprivate::copy(src1, &d_src1);
  iuprivate::copy(src2, &d_src2);

  cuAddWeighted(&d_src1, weight1, &d_src2, weight2, &d_dst, roi);

  iuprivate::copy(&d_dst, dst);
}

///////////////////////////////////////////////////////////////////////////////

// [gpu] multiplication with factor; Not-in-place; 8-bit; 1-channel
void mulC(const iu::ImageGpu_8u_C1* src, const unsigned char& factor, iu::ImageGpu_8u_C1* dst, const IuRect& roi)
{
  
  cuMulC(src, factor, dst, roi);

}

// [gpu] multiplication with factor; Not-in-place; 8-bit; 4-channel
void mulC(const iu::ImageGpu_8u_C4* src, const uchar4& factor, iu::ImageGpu_8u_C4* dst, const IuRect& roi)
{
  
  cuMulC(src, factor, dst, roi);

}

///////////////////////////////////////////////////////////////////////////////

// [gpu] multiplication with factor; Not-in-place; 32-bit; 1-channel
void mulC(const iu::ImageGpu_32f_C1* src, const float& factor, iu::ImageGpu_32f_C1* dst, const IuRect& roi)
{
  
  cuMulC(src, factor, dst, roi);

}

// [gpu] multiplication with factor; Not-in-place; 32-bit; 2-channel
void mulC(const iu::ImageGpu_32f_C2* src, const float2& factor, iu::ImageGpu_32f_C2* dst, const IuRect& roi)
{
  
  cuMulC(src, factor, dst, roi);

}

// [gpu] multiplication with factor; Not-in-place; 32-bit; 4-channel
void mulC(const iu::ImageGpu_32f_C4* src, const float4& factor, iu::ImageGpu_32f_C4* dst, const IuRect& roi)
{
  
  cuMulC(src, factor, dst, roi);

}


///////////////////////////////////////////////////////////////////////////////

// [gpu] multiplication with factor; Not-in-place; 32-bit; 1-channel
void mulC(const iu::VolumeGpu_32f_C1* src, const float& factor, iu::VolumeGpu_32f_C1* dst)
{
  
  cuMulC(src, factor, dst);

}

///////////////////////////////////////////////////////////////////////////////

// [gpu] add val; Not-in-place; 8-bit; 1-channel
void addC(const iu::ImageGpu_8u_C1* src, const unsigned char& val, iu::ImageGpu_8u_C1* dst, const IuRect& roi)
{
  
  cuAddC(src, val, dst, roi);

}

// [gpu] add val; Not-in-place; 8-bit; 4-channel
void addC(const iu::ImageGpu_8u_C4* src, const uchar4& val, iu::ImageGpu_8u_C4* dst, const IuRect& roi)
{
  
  cuAddC(src, val, dst, roi);

}

///////////////////////////////////////////////////////////////////////////////

// [gpu] add val; Not-in-place; 32-bit; 1-channel
void addC(const iu::ImageGpu_32f_C1* src, const float& val, iu::ImageGpu_32f_C1* dst, const IuRect& roi)
{
  
  cuAddC(src, val, dst, roi);

}

// [gpu] add val; Not-in-place; 32-bit; 2-channel
void addC(const iu::ImageGpu_32f_C2* src, const float2& val, iu::ImageGpu_32f_C2* dst, const IuRect& roi)
{
  
  cuAddC(src, val, dst, roi);

}

// [gpu] add val; Not-in-place; 32-bit; 4-channel
void addC(const iu::ImageGpu_32f_C4* src, const float4& val, iu::ImageGpu_32f_C4* dst, const IuRect& roi)
{
  
  cuAddC(src, val, dst, roi);

}

} // namespace iu
