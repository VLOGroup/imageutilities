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
 * \note supported gpu: 8u_C1, 8u_C4, 32f_C1, 32f_C4
 * \note 3-channel stuff not supported due to texture usage!
 */
IuStatus cuMinMax(const iu::ImageGpu_8u_C1 *src, const IuRect &roi, unsigned char& min, unsigned char& max);
IuStatus cuMinMax(const iu::ImageGpu_8u_C4 *src, const IuRect &roi, uchar4& min, uchar4& max);
IuStatus cuMinMax(const iu::ImageGpu_32f_C1 *src, const IuRect &roi, float& min, float& max);
IuStatus cuMinMax(const iu::ImageGpu_32f_C2 *src, const IuRect &roi, float2& min, float2& max);
IuStatus cuMinMax(const iu::ImageGpu_32f_C4 *src, const IuRect &roi, float4& min, float4& max);

/** Cuda wrappers for finding the minimum and maximum value of a volume
 * \param src Source volume [device]
 * \param[out] min Minium value found in the source volume.
 * \param[out] max Maximum value found in the source volume.
 *
 * \note supported gpu: 32f_C1
 */
IuStatus cuMinMax(iu::VolumeGpu_32f_C1 *src, float& min_C1, float& max_C1);

/** Cuda wrappers for finding the minimum value of an image in a certain ROI and the minimums coordinates.
 * \param src Source image [device]
 * \param src_roi Region of interest in the source image.
 * \param[out] min minimum value found in the source image.
 * \param[out] x x-coordinate of minimum value
 * \param[out] y y-coordinate of minimum value
 *
 * \note supported gpu: 32f_C1
 */
IuStatus cuMin(const iu::ImageGpu_32f_C1 *src, const IuRect &roi, float& min, int& min_x, int& min_y);

/** Cuda wrappers for finding the maximum value of an image in a certain ROI and the maximums coordinates.
 * \param src Source image [device]
 * \param src_roi Region of interest in the source image.
 * \param[out] max Maximum value found in the source image.
 * \param[out] x x-coordinate of maximum value
 * \param[out] y y-coordinate of maximum value
 *
 * \note supported gpu: 32f_C1
 */
IuStatus cuMax(const iu::ImageGpu_32f_C1 *src, const IuRect &roi, float& max, int& max_x, int& max_y);



/* ***************************************************************************
   SUMMATION
*/

/** Cuda wrappers for computing the sum of an image in a certain ROI.
 * \param src Source image [device]
 * \param src_roi Region of interest in the source image.
 * \param[out] sum Contains computed sum.
 *
 * \note supported gpu: 8u_C1, 8u_C4, 32f_C1, 32f_C4
 * \note 3-channel stuff not supported due to texture usage!
 */
IuStatus cuSummation(const iu::ImageGpu_8u_C1 *src, const IuRect &roi, long& sum);
//IuStatus cuSummation(const iu::ImageGpu_8u_C4 *src, const IuRect &roi, long sum[4]);
IuStatus cuSummation(const iu::ImageGpu_32f_C1 *src, const IuRect &roi, double& sum);
//IuStatus cuSummation(const iu::ImageGpu_32f_C4 *src, const IuRect &roi, double sum[4]);

/* ***************************************************************************
   NORMS (L1/L2)
*/

/** Cuda wrapper computing the L1 norm of differences between pixel values two images.
 * \param src1 Pointer to the first source image.
 * \param src2 Pointer to the second source image.
 * \param roi Region of interest in the source image.
 * \param norm Contains computed L1 norm.
 */
IuStatus cuNormDiffL1(const iu::ImageGpu_32f_C1* src1, const iu::ImageGpu_32f_C1* src2, const IuRect& roi, double& norm);

/** Cuda wrapper coputing the L1 norm of differences between pixel values of an image and a constant value. |src-value|
 * \param src1 Pointer to the first source image.
 * \param value Subtrahend applied to every pixel on \a src image before calculating the L1 norm.
 * \param roi Region of interest in the source image.
 * \param norm Contains computed L1 norm.
 */
IuStatus cuNormDiffL1(const iu::ImageGpu_32f_C1* src, const float& value, const IuRect& roi, double& norm);

/** Cuda wrapper computing the L2 norm of differences between pixel values two images.
 * \param src1 Pointer to the first source image.
 * \param src2 Pointer to the second source image.
 * \param roi Region of interest in the source image.
 * \param norm Contains computed L2 norm.
 */
IuStatus cuNormDiffL2(const iu::ImageGpu_32f_C1* src1, const iu::ImageGpu_32f_C1* src2, const IuRect& roi, double& norm);

/** Cuda wrapper coputing the L2 norm of differences between pixel values of an image and a constant value. |src-value|
 * \param src1 Pointer to the first source image.
 * \param value Subtrahend applied to every pixel on \a src image before calculating the L2 norm.
 * \param roi Region of interest in the source image.
 * \param norm Contains computed L2 norm.
 */
IuStatus cuNormDiffL2(const iu::ImageGpu_32f_C1* src, const float& value, const IuRect& roi, double& norm);

/* ***************************************************************************
   ERROR MEASUREMENTS
*/

/** Cuda wrapper to compute the mean-squared error between the src and the reference image.
 *
 *
 */
IuStatus cuMse(const iu::ImageGpu_32f_C1* src, const iu::ImageGpu_32f_C1* reference, const IuRect& roi, double& mse);

/** Cuda wrapper to compute structural similarity index between the src and the reference image.
 *
 *
 */
IuStatus cuSsim(const iu::ImageGpu_32f_C1* src, const iu::ImageGpu_32f_C1* reference, const IuRect& roi, double& ssim);


/* ***************************************************************************
   HISTOGRAMS
*/

IuStatus cuColorHistogram(const iu::ImageGpu_8u_C4* binned_image, const iu::ImageGpu_8u_C1* mask,
                          iu::VolumeGpu_32f_C1* hist, unsigned char mask_val);

}
#endif // IUMATH_STATISTICS_CUH
