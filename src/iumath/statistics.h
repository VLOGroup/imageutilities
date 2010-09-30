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
 * Description : Definition of statistics functions
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUMATH_STATISTICS_H
#define IUMATH_STATISTICS_H

#include <iucore/coredefs.h>
#include <iucore/memorydefs.h>

namespace iuprivate {


/** Finds the minimum and maximum value of an image in a certain ROI.
 * \param src Source image [device]
 * \param src_roi Region of interest in the source image.
 * \param[out] min Minium value found in the source image.
 * \param[out] max Maximum value found in the source image.
 *
 * \note supported ipp: 8u_C1, 8u_C3, 8u_C4, 32f_C1, 32f_C3, 32f_C4,
 * \note supported npp: 8u_C1, 8u_C4, 32f_C1, 32f_C4,
 */
//// find min/max; host; 8-bit
//void minMax(const iu::ImageCpu_8u_C1* src, const IuRect& roi, Npp8u& min, Npp8u& max);
//void minMax(const iu::ImageCpu_8u_C3* src, const IuRect& roi, Npp8u min[3], Npp8u max[3]);
//void minMax(const iu::ImageCpu_8u_C4* src, const IuRect& roi, Npp8u min[4], Npp8u max[4]);

//// find min/max; host; 32-bit
//void minMax(const iu::ImageCpu_32f_C1* src, const IuRect& roi, float& min, float& max);
//void minMax(const iu::ImageCpu_32f_C3* src, const IuRect& roi, float min[3], float max[3]);
//void minMax(const iu::ImageCpu_32f_C4* src, const IuRect& roi, float min[4], float max[4]);

// find min/max; device; 8-bit
void minMax(const iu::ImageGpu_8u_C1 *src, const IuRect &roi, unsigned char& min, unsigned char& max);
void minMax(const iu::ImageGpu_8u_C4 *src, const IuRect &roi, uchar4& min, uchar4& max);

// find min/max; device; 32-bit
void minMax(const iu::ImageGpu_32f_C1 *src, const IuRect &roi, float& min, float& max);
void minMax(const iu::ImageGpu_32f_C2 *src, const IuRect &roi, float2& min, float2& max);
void minMax(const iu::ImageGpu_32f_C4 *src, const IuRect &roi, float4& min, float4& max);


/** Computes the sum of pixels in a certain ROI of an image.
 * \param src Source image [device]
 * \param src_roi Region of interest in the source image.
 * \param[out] sum Contains computed sum.
 *
 * \note supported ipp: 8u_C1, 8u_C3, 8u_C4, 32f_C1, 32f_C3, 32f_C4,
 * \note supported npp: 8u_C1, 8u_C4, 32f_C1, 32f_C4,
 */
// compute sum; host; 8-bit
//void summation(const iu::ImageCpu_8u_C1* src, const IuRect& roi, Npp64s& sum);
//void summation(const iu::ImageCpu_8u_C3* src, const IuRect& roi, Npp64s sum[3]);
//void summation(const iu::ImageCpu_8u_C4* src, const IuRect& roi, Npp64s sum[4]);

//// compute sum; host; 32-bit
//void summation(const iu::ImageCpu_32f_C1* src, const IuRect& roi, Npp64f& sum);
//void summation(const iu::ImageCpu_32f_C3* src, const IuRect& roi, Npp64f sum[3]);
//void summation(const iu::ImageCpu_32f_C4* src, const IuRect& roi, Npp64f sum[4]);

// compute sum; device; 8-bit
void summation(const iu::ImageGpu_8u_C1* src, const IuRect& roi, Npp64s& sum);
//void summation(iu::ImageGpu_8u_C4* src, const IuRect& roi, Npp8u sum[4]);

// compute sum; device; 32-bit
void summation(const iu::ImageGpu_32f_C1* src, const IuRect& roi, Npp64f& sum);
//void summation(iu::ImageGpu_32f_C4* src, const IuRect& roi, Npp32f sum[4]);


/** Computes the L1 norm of differences between pixel values of two images. |src1-src2|
 * \param src1 Pointer to the first source image.
 * \param src2 Pointer to the second source image.
 * \param roi Region of interest in the source image.
 * \param norm Contains computed L1 norm.
 */
void normDiffL1(const iu::ImageGpu_32f_C1* src1, const iu::ImageGpu_32f_C1* src2, const IuRect& roi, Npp64f& norm);

/** Computes the L1 norm of differences between pixel values of an image and a constant value. |src-value|
 * \param src1 Pointer to the first source image.
 * \param value Subtrahend applied to every pixel on \a src image before calculating the L1 norm.
 * \param roi Region of interest in the source image.
 * \param norm Contains computed L1 norm.
 */
void normDiffL1(const iu::ImageGpu_32f_C1* src, const Npp32f& value, const IuRect& roi, Npp64f& norm);

/** Computes the L2 norm of differences between pixel values of two images. ||src1-src2||
 * \param src1 Pointer to the first source image.
 * \param src2 Pointer to the second source image.
 * \param roi Region of interest in the source image.
 * \param norm Contains computed L2 norm.
 */
void normDiffL2(const iu::ImageGpu_32f_C1* src1, const iu::ImageGpu_32f_C1* src2, const IuRect& roi, Npp64f& norm);

/** Computes the L2 norm of differences between pixel values of an image and a constant value. ||src-value||
 * \param src Pointer to the first source image.
 * \param value Subtrahend applied to every pixel on \a src image before calculating the L2 norm.
 * \param roi Region of interest in the source image.
 * \param norm Contains computed L2 norm.
 */
void normDiffL2(const iu::ImageGpu_32f_C1* src, const Npp32f& value, const IuRect& roi, Npp64f& norm);

// internal computation of the mean-squared error
void mse(const iu::ImageGpu_32f_C1* src, const iu::ImageGpu_32f_C1* reference, const IuRect& roi, Npp64f& mse);

// internal computation of the structural similarity index
void ssim(const iu::ImageGpu_32f_C1* src, const iu::ImageGpu_32f_C1* reference, const IuRect& roi, Npp64f& ssim);

} // namespace iuprivate

#endif // IUMATH_STATISTICS_H
