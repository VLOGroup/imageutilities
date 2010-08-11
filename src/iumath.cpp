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
 * Module      : Math Module
 * Class       : Wrapper
 * Language    : C
 * Description : Implementation of public interfaces to math module
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */


#include "iumath.h"
#include "math/arithmetic.h"
#include "math/statistics.h"

namespace iu {

/* ***************************************************************************
     ARITHMETICS
 * ***************************************************************************/

// [device] weighted add; Not-in-place; 32-bit;
void addWeighted(const iu::ImageNpp_32f_C1* src1, const Npp32f& weight1,
                 const iu::ImageNpp_32f_C1* src2, const Npp32f& weight2,
                 iu::ImageNpp_32f_C1* dst, const IuRect& roi)
{iuprivate::addWeighted(src1, weight1, src2, weight2, dst, roi);}

// [device] multiplication with factor; Not-in-place; 8-bit;
void mulC(const ImageNpp_8u_C1* src, const Npp8u& factor, ImageNpp_8u_C1* dst, const IuRect& roi)
{iuprivate::mulC(src, factor, dst, roi);}
void mulC(const ImageNpp_8u_C4* src, const Npp8u factor[4], ImageNpp_8u_C4* dst, const IuRect& roi)
{iuprivate::mulC(src, factor, dst, roi);}

// [device] multiplication with factor; Not-in-place; 32-bit;
void mulC(const ImageNpp_32f_C1* src, const Npp32f& factor, ImageNpp_32f_C1* dst, const IuRect& roi)
{iuprivate::mulC(src, factor, dst, roi);}
void mulC(const ImageNpp_32f_C4* src, const Npp32f factor[4], ImageNpp_32f_C4* dst, const IuRect& roi)
{iuprivate::mulC(src, factor, dst, roi);}



/* ***************************************************************************
     STATISTICS
 * ***************************************************************************/

// find min/max; device; 8-bit
void minMax(const iu::ImageNpp_8u_C1* src, const IuRect& roi, Npp8u& min, Npp8u& max)
{iuprivate::minMax(src, roi, min, max);}
void minMax(const iu::ImageNpp_8u_C4* src, const IuRect& roi, Npp8u min[4], Npp8u max[4])
{iuprivate::minMax(src, roi, min, max);}

// find min/max; device; 32-bit
void minMax(const iu::ImageNpp_32f_C1* src, const IuRect& roi, Npp32f& min, Npp32f& max)
{iuprivate::minMax(src, roi, min, max);}
void minMax(const iu::ImageNpp_32f_C2* src, const IuRect& roi, Npp32f min[2], Npp32f max[2])
{iuprivate::minMax(src, roi, min, max);}
void minMax(const iu::ImageNpp_32f_C4* src, const IuRect& roi, Npp32f min[4], Npp32f max[4])
{iuprivate::minMax(src, roi, min, max);}

// compute sum; device; 8-bit
void summation(const iu::ImageNpp_8u_C1* src, const IuRect& roi, Npp64s& sum)
{iuprivate::summation(src, roi, sum);}
//void summation(iu::ImageNpp_8u_C4* src, const IuRect& roi, Npp64s sum[4]);

// compute sum; device; 32-bit
void summation(const iu::ImageNpp_32f_C1* src, const IuRect& roi, Npp64f& sum)
{iuprivate::summation(src, roi, sum);}
//void summation(iu::ImageNpp_32f_C4* src, const IuRect& roi, Npp64f sum[4]);


// |src1-src2|
void normDiffL1(const iu::ImageNpp_32f_C1* src1, const iu::ImageNpp_32f_C1* src2, const IuRect& roi, Npp64f& norm)
{iuprivate::normDiffL1(src1, src2, roi, norm);}
// |src-value|
void normDiffL1(const iu::ImageNpp_32f_C1* src, const Npp32f& value, const IuRect& roi, Npp64f& norm)
{iuprivate::normDiffL1(src, value, roi, norm);}
// ||src1-src2||
void normDiffL2(const iu::ImageNpp_32f_C1* src1, const iu::ImageNpp_32f_C1* src2, const IuRect& roi, Npp64f& norm)
{iuprivate::normDiffL1(src1, src2, roi, norm);}
// ||src-value||
void normDiffL2(const iu::ImageNpp_32f_C1* src, const Npp32f& value, const IuRect& roi, Npp64f& norm)
{iuprivate::normDiffL1(src, value, roi, norm);}

/* ***************************************************************************
     ERROR MEASUREMENTS
 * ***************************************************************************/

// [device] compute mse; 32-bit
void mse(const iu::ImageNpp_32f_C1* src, const iu::ImageNpp_32f_C1* reference, const IuRect& roi, Npp64f& mse)
{iuprivate::mse(src, reference, roi, mse);}

// [device] compute ssim; 32-bit
void ssim(const iu::ImageNpp_32f_C1* src, const iu::ImageNpp_32f_C1* reference, const IuRect& roi, Npp64f& ssim)
{iuprivate::ssim(src, reference, roi, ssim);}

} // namespace iu
