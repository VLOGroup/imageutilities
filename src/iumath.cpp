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
#include "iumath/arithmetic.h"
#include "iumath/statistics.h"

namespace iu {

/* ***************************************************************************
     ARITHMETICS
 * ***************************************************************************/

// [device] weighted add; Not-in-place; 32-bit;
void addWeighted(const iu::ImageGpu_32f_C1* src1, const float& weight1,
                 const iu::ImageGpu_32f_C1* src2, const float& weight2,
                 iu::ImageGpu_32f_C1* dst, const IuRect& roi)
{iuprivate::addWeighted(src1, weight1, src2, weight2, dst, roi);}

// [gpu] multiplication with factor; Not-in-place; 8-bit;
void mulC(const iu::ImageGpu_8u_C1* src, const unsigned char& factor, iu::ImageGpu_8u_C1* dst, const IuRect& roi)
{iuprivate::mulC(src, factor, dst, roi);}
void mulC(const iu::ImageGpu_8u_C4* src, const uchar4& factor, iu::ImageGpu_8u_C4* dst, const IuRect& roi)
{iuprivate::mulC(src, factor, dst, roi);}

// [gpu] multiplication with factor; Not-in-place; 32-bit;
void mulC(const iu::ImageGpu_32f_C1* src, const float& factor, iu::ImageGpu_32f_C1* dst, const IuRect& roi)
{iuprivate::mulC(src, factor, dst, roi);}
void mulC(const iu::ImageGpu_32f_C2* src, const float2& factor, iu::ImageGpu_32f_C2* dst, const IuRect& roi)
{iuprivate::mulC(src, factor, dst, roi);}
void mulC(const iu::ImageGpu_32f_C4* src, const float4& factor, iu::ImageGpu_32f_C4* dst, const IuRect& roi)
{iuprivate::mulC(src, factor, dst, roi);}

// [gpu] volume multiplication with factor; Not-in-place; 32-bit;
void mulC(const iu::VolumeGpu_32f_C1* src, const float& factor, iu::VolumeGpu_32f_C1* dst)
{iuprivate::mulC(src, factor, dst);}


// [gpu] addval; Not-in-place; 8-bit;
void addC(const iu::ImageGpu_8u_C1* src, const unsigned char& val, iu::ImageGpu_8u_C1* dst, const IuRect& roi)
{iuprivate::addC(src, val, dst, roi);}
void addC(const iu::ImageGpu_8u_C4* src, const uchar4& val, iu::ImageGpu_8u_C4* dst, const IuRect& roi)
{iuprivate::addC(src, val, dst, roi);}

// [gpu] addval; Not-in-place; 32-bit;
void addC(const iu::ImageGpu_32f_C1* src, const float& val, iu::ImageGpu_32f_C1* dst, const IuRect& roi)
{iuprivate::addC(src, val, dst, roi);}
void addC(const iu::ImageGpu_32f_C2* src, const float2& val, iu::ImageGpu_32f_C2* dst, const IuRect& roi)
{iuprivate::addC(src, val, dst, roi);}
void addC(const iu::ImageGpu_32f_C4* src, const float4& val, iu::ImageGpu_32f_C4* dst, const IuRect& roi)
{iuprivate::addC(src, val, dst, roi);}


/* ***************************************************************************
     STATISTICS
 * ***************************************************************************/

// find min/max; device; 8-bit
void minMax(const ImageGpu_8u_C1* src, const IuRect& roi, unsigned char& min, unsigned char& max)
{iuprivate::minMax(src, roi, min, max);}
void minMax(const ImageGpu_8u_C4* src, const IuRect& roi, uchar4& min, uchar4& max)
{iuprivate::minMax(src, roi, min, max);}

// find min/max; device; 32-bit
void minMax(const iu::ImageGpu_32f_C1* src, const IuRect& roi, float& min, float& max)
{iuprivate::minMax(src, roi, min, max);}
void minMax(const ImageGpu_32f_C2* src, const IuRect& roi, float2& min, float2& max)
{iuprivate::minMax(src, roi, min, max);}
void minMax(const ImageGpu_32f_C4* src, const IuRect& roi, float4& min, float4& max)
{iuprivate::minMax(src, roi, min, max);}

// find min/max; volume; device; 32-bit
void minMax(VolumeGpu_32f_C1* src, float& min, float& max)
{iuprivate::minMax(src, min, max);}

// find min value and its coordinates; 32-bit
void min(const iu::ImageGpu_32f_C1* src, const IuRect&roi, float& min, int& x, int& y)
{iuprivate::min(src, roi, min, x, y);}

// find max value and its coordinates; 32-bit
void max(const iu::ImageGpu_32f_C1* src, const IuRect&roi, float& max, int& x, int& y)
{iuprivate::max(src, roi, max, x, y);}

// compute sum; device; 8-bit
void summation(const iu::ImageGpu_8u_C1* src, const IuRect& roi, long& sum)
{iuprivate::summation(src, roi, sum);}
//void summation(iu::ImageGpu_8u_C4* src, const IuRect& roi, long sum[4]);

// compute sum; device; 32-bit
void summation(const iu::ImageGpu_32f_C1* src, const IuRect& roi, double& sum)
{iuprivate::summation(src, roi, sum);}
//void summation(iu::ImageGpu_32f_C4* src, const IuRect& roi, double sum[4]);

// compute sum; device; 3D; 32-bit
void summation(iu::VolumeGpu_32f_C1* src, const IuCube& roi, double& sum)
{iuprivate::summation(src, roi, sum);}


// |src1-src2|
void normDiffL1(const iu::ImageGpu_32f_C1* src1, const iu::ImageGpu_32f_C1* src2, const IuRect& roi, double& norm)
{iuprivate::normDiffL1(src1, src2, roi, norm);}
// |src-value|
void normDiffL1(const iu::ImageGpu_32f_C1* src, const float& value, const IuRect& roi, double& norm)
{iuprivate::normDiffL1(src, value, roi, norm);}
// ||src1-src2||
void normDiffL2(const iu::ImageGpu_32f_C1* src1, const iu::ImageGpu_32f_C1* src2, const IuRect& roi, double& norm)
{iuprivate::normDiffL1(src1, src2, roi, norm);}
// ||src-value||
void normDiffL2(const iu::ImageGpu_32f_C1* src, const float& value, const IuRect& roi, double& norm)
{iuprivate::normDiffL1(src, value, roi, norm);}

/* ***************************************************************************
     ERROR MEASUREMENTS
 * ***************************************************************************/

// [device] compute mse; 32-bit
void mse(const iu::ImageGpu_32f_C1* src, const iu::ImageGpu_32f_C1* reference, const IuRect& roi, double& mse)
{iuprivate::mse(src, reference, roi, mse);}

// [device] compute ssim; 32-bit
void ssim(const iu::ImageGpu_32f_C1* src, const iu::ImageGpu_32f_C1* reference, const IuRect& roi, double& ssim)
{iuprivate::ssim(src, reference, roi, ssim);}

/* ***************************************************************************
     HISTOGRAMS
 * ***************************************************************************/

void colorHistogram(const iu::ImageGpu_8u_C4* binned_image, const iu::ImageGpu_8u_C1* mask,
                              iu::VolumeGpu_32f_C1* hist, unsigned char mask_val)
{iuprivate::colorHistogram(binned_image, mask, hist, mask_val);}


} // namespace iu
