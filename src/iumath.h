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
 * Description : Public interfaces to math module
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IU_MATH_MODULE_H
#define IU_MATH_MODULE_H

#include "iudefs.h"

namespace iu {

/** \defgroup Math
 *  \brief The math module.
 *  TODO more detailed docu
 *  @{
 */

/* ***************************************************************************
     ARITHMETIC
 * ***************************************************************************/

//////////////////////////////////////////////////////////////////////////////
/** @defgroup Arithmetics
 *  @ingroup Math
 *  TODO more detailed docu
 *  @{
 */

/** Adding an image with an additional weighting factor to another.
 * \param src1 Source image 1.
 * \param src2 Source image 2.
 * \param weight Weighting of image 2 before its added to image 1.
 * \param dst Result image dst=weight1*src1 + weight1*src2.
 * \param roi Region of interest in the source and destination image
 *
 * \note supported gpu: 32f_C1
 */
// [device] weighted add; Not-in-place; 32-bit;
IU_DLLAPI void addWeighted(const iu::ImageGpu_32f_C1* src1, const float& weight1,
                           const iu::ImageGpu_32f_C1* src2, const float& weight2,
                           iu::ImageGpu_32f_C1* dst, const IuRect& roi);


/** Multiplication of every pixel with a constant factor. (can be called in-place)
 * \param src Source image.
 * \param factor Multiplication factor applied to each pixel.
 * \param dst Destination image.
 * \param roi Region of interest in the source and destination image
 *
 * \note supported gpu: 8u_C1, 8u_C4, 32f_C1, 32f_C4,
 */
// [gpu] multiplication with factor; Not-in-place; 8-bit;
IU_DLLAPI void mulC(const iu::ImageGpu_8u_C1* src, const unsigned char& factor, iu::ImageGpu_8u_C1* dst, const IuRect& roi);
IU_DLLAPI void mulC(const iu::ImageGpu_8u_C4* src, const uchar4& factor, iu::ImageGpu_8u_C4* dst, const IuRect& roi);

// [gpu] multiplication with factor; Not-in-place; 32-bit;
IU_DLLAPI void mulC(const iu::ImageGpu_32f_C1* src, const float& factor, iu::ImageGpu_32f_C1* dst, const IuRect& roi);
IU_DLLAPI void mulC(const iu::ImageGpu_32f_C4* src, const float4& factor, iu::ImageGpu_32f_C4* dst, const IuRect& roi);

/** In-place multiplication of every pixel with a constant factor.
 * \param factor Multiplication factor applied to each pixel.
 * \param srcdst Source and destination
 * \param roi Region of interest in the source/destination image
 *
 * \note supported gpu: 8u_C1, 8u_C4, 32f_C1, 32f_C4,
 */
// multiplication with factor; host; 8-bit
// :TODO: \todo implement inplace interfaces.

/** @} */ // end of Arithmetics

/* ***************************************************************************
     STATISTICS
 * ***************************************************************************/

//////////////////////////////////////////////////////////////////////////////
/** @defgroup Statistics
 *  @ingroup Math
 *  TODO more detailed docu
 *  @{
 */

/** Finds the minimum and maximum value of an image in a certain ROI.
 * \param src Source image [device]
 * \param src_roi Region of interest in the source image.
 * \param[out] min Minium value found in the source image.
 * \param[out] max Maximum value found in the source image.
 *
 * \note supported gpu: 8u_C1, 8u_C4, 32f_C1, 32f_C4,
 */
// find min/max; device; 8-bit
IU_DLLAPI void minMax(const ImageGpu_8u_C1* src, const IuRect& roi, unsigned char& min, unsigned char& max);
IU_DLLAPI void minMax(const ImageGpu_8u_C4* src, const IuRect& roi, uchar4& min, uchar4& max);
// find min/max; device; 32-bit
IU_DLLAPI void minMax(const ImageGpu_32f_C1* src, const IuRect& roi, float& min, float& max);
IU_DLLAPI void minMax(const ImageGpu_32f_C2* src, const IuRect& roi, float2& min, float2& max);
IU_DLLAPI void minMax(const ImageGpu_32f_C4* src, const IuRect& roi, float4& min, float4& max);


/** Computes the sum of pixels in a certain ROI of an image.
 * \param src Source image [device]
 * \param src_roi Region of interest in the source image.
 * \param[out] sum Contains computed sum.
 *
 * \note supported gpu: 8u_C1, 8u_C4, 32f_C1, 32f_C4,
 */
// compute sum; device; 8-bit
IU_DLLAPI void summation(const ImageGpu_8u_C1* src, const IuRect& roi, long& sum);
//IU_DLLAPI void summation(const ImageGpu_8u_C4* src, const IuRect& roi, long sum[4]);

// compute sum; device; 32-bit
IU_DLLAPI void summation(const ImageGpu_32f_C1* src, const IuRect& roi, double& sum);
//IU_DLLAPI void summation(const ImageGpu_32f_C4* src, const IuRect& roi, double sum[4]);


/** Computes the L1 norm of differences between pixel values of two images. |src1-src2|
 * \param src1 Pointer to the first source image.
 * \param src2 Pointer to the second source image.
 * \param roi Region of interest in the source image.
 * \param norm Contains computed L1 norm.
 */
IU_DLLAPI void normDiffL1(const ImageGpu_32f_C1* src1, const ImageGpu_32f_C1* src2, const IuRect& roi, double& norm);

/** Computes the L1 norm of differences between pixel values of an image and a constant value. |src-value|
 * \param src1 Pointer to the first source image.
 * \param value Subtrahend applied to every pixel on \a src image before calculating the L1 norm.
 * \param roi Region of interest in the source image.
 * \param norm Contains computed L1 norm.
 */
IU_DLLAPI void normDiffL1(const ImageGpu_32f_C1* src, const float& value, const IuRect& roi, double& norm);

/** Computes the L2 norm of differences between pixel values of two images. ||src1-src2||
 * \param src1 Pointer to the first source image.
 * \param src2 Pointer to the second source image.
 * \param roi Region of interest in the source image.
 * \param norm Contains computed L2 norm.
 */
IU_DLLAPI void normDiffL2(const ImageGpu_32f_C1* src1, const ImageGpu_32f_C1* src2, const IuRect& roi, double& norm);

/** Computes the L2 norm of differences between pixel values of an image and a constant value. ||src-value||
 * \param src Pointer to the first source image.
 * \param value Subtrahend applied to every pixel on \a src image before calculating the L2 norm.
 * \param roi Region of interest in the source image.
 * \param norm Contains computed L2 norm.
 */
IU_DLLAPI void normDiffL2(const ImageGpu_32f_C1* src, const float& value, const IuRect& roi, double& norm);

/** @} */ // end of Statistics

/* ***************************************************************************
     ERROR MEASUREMENTS
 * ***************************************************************************/

//////////////////////////////////////////////////////////////////////////////
/** @defgroup Error Measurements
 *  @ingroup Math
 *  TODO more detailed docu
 *  @{
 */

/** Computes the mean-squared error between the src and the reference image.
 * \param src Pointer to the source image.
 * \param reference Pointer to the refernce image.
 * \param roi Region of interest in the source and reference image.
 * \param mse Contains the computed mean-squared error.
 */
IU_DLLAPI void mse(const iu::ImageGpu_32f_C1* src, const iu::ImageGpu_32f_C1* reference, const IuRect& roi, double& mse);

/** Computes the structural similarity index between the src and the reference image.
 * \param src Pointer to the source image.
 * \param reference Pointer to the refernce image.
 * \param roi Region of interest in the source and reference image.
 * \param mse Contains the computed  structural similarity index.
 */
IU_DLLAPI void ssim(const iu::ImageGpu_32f_C1* src, const iu::ImageGpu_32f_C1* reference, const IuRect& roi, double& ssim);

/** @} */ // end of Error Measurements


/** @} */ // end of Math

} // namespace iu

#endif // IU_MATH_MODULE_H
