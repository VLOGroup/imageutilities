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
void addWeighted(const iu::ImageNpp_32f_C1* src1, const Npp32f& weight1,
                 const iu::ImageNpp_32f_C1* src2, const Npp32f& weight2,
                 iu::ImageNpp_32f_C1* dst, const IuRect& roi)
{
  NppStatus status;
  status = cuAddWeighted(src1, weight1, src2, weight2, dst, roi);
  IU_ASSERT(status == NPP_NO_ERROR);
}

///////////////////////////////////////////////////////////////////////////////

// [device] multiplication with factor; Not-in-place; 8-bit; 1-channel
void mulC(const iu::ImageNpp_8u_C1* src, const Npp8u& factor, iu::ImageNpp_8u_C1* dst, const IuRect& roi)
{
  NppStatus status;
  status = cuMulC(src, factor, dst, roi);
  IU_ASSERT(status == NPP_NO_ERROR);
}

//// [device] multiplication with factor; Not-in-place; 8-bit; 3-channel
//void mulC(const iu::ImageNpp_8u_C3* src, const Npp8u factor[3], iu::ImageNpp_8u_C3* dst, const IuRect& roi)
//{
//  NppStatus status;
//  status = cuMulC(src, factor, dst, roi);
//  IU_ASSERT(status == NPP_NO_ERROR);
//}

// [device] multiplication with factor; Not-in-place; 8-bit; 4-channel
void mulC(const iu::ImageNpp_8u_C4* src, const Npp8u factor[4], iu::ImageNpp_8u_C4* dst, const IuRect& roi)
{
  NppStatus status;
  status = cuMulC(src, factor, dst, roi);
  IU_ASSERT(status == NPP_NO_ERROR);
}

///////////////////////////////////////////////////////////////////////////////

// [device] multiplication with factor; Not-in-place; 32-bit; 1-channel
void mulC(const iu::ImageNpp_32f_C1* src, const Npp32f& factor, iu::ImageNpp_32f_C1* dst, const IuRect& roi)
{
  NppStatus status;
  status = cuMulC(src, factor, dst, roi);
  IU_ASSERT(status == NPP_NO_ERROR);
}

//// [device] multiplication with factor; Not-in-place; 32-bit; 3-channel
//void mulC(const iu::ImageNpp_32f_C3* src, const Npp32f factor[3], iu::ImageNpp_32f_C3* dst, const IuRect& roi)
//{
//  NppStatus status;
//  status = cuMulC(src, factor, dst, roi);
//  IU_ASSERT(status == NPP_NO_ERROR);
//}

// [device] multiplication with factor; Not-in-place; 32-bit; 4-channel
void mulC(const iu::ImageNpp_32f_C4* src, const Npp32f factor[3], iu::ImageNpp_32f_C4* dst, const IuRect& roi)
{
  NppStatus status;
  status = cuMulC(src, factor, dst, roi);
  IU_ASSERT(status == NPP_NO_ERROR);
}

} // namespace iu
