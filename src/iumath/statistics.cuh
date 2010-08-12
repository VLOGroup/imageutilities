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
 * Description : Definition of Cuda wrappers for statistics functions
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUMATH_STATISTICS_CUH
#define IUMATH_STATISTICS_CUH

#include <iucore/coredefs.h>
#include <iucore/memorydefs.h>

namespace iuprivate {

/* ***************************************************************************
   MIN/MAX
*/

/** Cuda wrappers for finding the minimum and maximum value of an image in a certain ROI.
 * \param src Source image [device]
 * \param src_roi Region of interest in the source image.
 * \param[out] min Minium value found in the source image.
 * \param[out] max Maximum value found in the source image.
 *
 * \note supported npp: 8u_C1, 8u_C4, 32f_C1, 32f_C4
 * \note 3-channel stuff not supported due to texture usage!
 */
NppStatus cuMinMax(const iu::ImageNpp_8u_C1 *src, const IuRect &roi, Npp8u& min, Npp8u& max);
NppStatus cuMinMax(const iu::ImageNpp_8u_C4 *src, const IuRect &roi, Npp8u min[4], Npp8u max[4]);
NppStatus cuMinMax(const iu::ImageNpp_32f_C1 *src, const IuRect &roi, Npp32f& min, Npp32f& max);
NppStatus cuMinMax(const iu::ImageNpp_32f_C2 *src, const IuRect &roi, Npp32f min[2], Npp32f max[2]);
NppStatus cuMinMax(const iu::ImageNpp_32f_C4 *src, const IuRect &roi, Npp32f min[4], Npp32f max[4]);

/* ***************************************************************************
   SUMMATION
*/

/** Cuda wrappers for computing the sum of an image in a certain ROI.
 * \param src Source image [device]
 * \param src_roi Region of interest in the source image.
 * \param[out] sum Contains computed sum.
 *
 * \note supported npp: 8u_C1, 8u_C4, 32f_C1, 32f_C4
 * \note 3-channel stuff not supported due to texture usage!
 */
NppStatus cuSummation(const iu::ImageNpp_8u_C1 *src, const IuRect &roi, Npp64s& sum);
//NppStatus cuSummation(const iu::ImageNpp_8u_C4 *src, const IuRect &roi, Npp64s sum[4]);
NppStatus cuSummation(const iu::ImageNpp_32f_C1 *src, const IuRect &roi, Npp64f& sum);
//NppStatus cuSummation(const iu::ImageNpp_32f_C4 *src, const IuRect &roi, Npp64f sum[4]);

/* ***************************************************************************
   NORMS (L1/L2)
*/

/** Cuda wrapper computing the L1 norm of differences between pixel values two images.
 * \param src1 Pointer to the first source image.
 * \param src2 Pointer to the second source image.
 * \param roi Region of interest in the source image.
 * \param norm Contains computed L1 norm.
 */
NppStatus cuNormDiffL1(const iu::ImageNpp_32f_C1* src1, const iu::ImageNpp_32f_C1* src2, const IuRect& roi, Npp64f& norm);

/** Cuda wrapper coputing the L1 norm of differences between pixel values of an image and a constant value. |src-value|
 * \param src1 Pointer to the first source image.
 * \param value Subtrahend applied to every pixel on \a src image before calculating the L1 norm.
 * \param roi Region of interest in the source image.
 * \param norm Contains computed L1 norm.
 */
NppStatus cuNormDiffL1(const iu::ImageNpp_32f_C1* src, const Npp32f& value, const IuRect& roi, Npp64f& norm);

/** Cuda wrapper computing the L2 norm of differences between pixel values two images.
 * \param src1 Pointer to the first source image.
 * \param src2 Pointer to the second source image.
 * \param roi Region of interest in the source image.
 * \param norm Contains computed L2 norm.
 */
NppStatus cuNormDiffL2(const iu::ImageNpp_32f_C1* src1, const iu::ImageNpp_32f_C1* src2, const IuRect& roi, Npp64f& norm);

/** Cuda wrapper coputing the L2 norm of differences between pixel values of an image and a constant value. |src-value|
 * \param src1 Pointer to the first source image.
 * \param value Subtrahend applied to every pixel on \a src image before calculating the L2 norm.
 * \param roi Region of interest in the source image.
 * \param norm Contains computed L2 norm.
 */
NppStatus cuNormDiffL2(const iu::ImageNpp_32f_C1* src, const Npp32f& value, const IuRect& roi, Npp64f& norm);

/* ***************************************************************************
   ERROR MEASUREMENTS
*/

/** Cuda wrapper to compute the mean-squared error between the src and the reference image.
 *
 *
 */
NppStatus cuMse(const iu::ImageNpp_32f_C1* src, const iu::ImageNpp_32f_C1* reference, const IuRect& roi, Npp64f& mse);

/** Cuda wrapper to compute structural similarity index between the src and the reference image.
 *
 *
 */
NppStatus cuSsim(const iu::ImageNpp_32f_C1* src, const iu::ImageNpp_32f_C1* reference, const IuRect& roi, Npp64f& ssim);



}
#endif // IUMATH_STATISTICS_CUH
