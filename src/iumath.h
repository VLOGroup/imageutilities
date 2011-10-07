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
IUCORE_DLLAPI void addWeighted(const iu::ImageGpu_32f_C1* src1, const float& weight1,
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
IUCORE_DLLAPI void mulC(const iu::ImageGpu_8u_C1* src, const unsigned char& factor, iu::ImageGpu_8u_C1* dst, const IuRect& roi);
IUCORE_DLLAPI void mulC(const iu::ImageGpu_8u_C4* src, const uchar4& factor, iu::ImageGpu_8u_C4* dst, const IuRect& roi);

// [gpu] multiplication with factor; Not-in-place; 32-bit;
IUCORE_DLLAPI void mulC(const iu::ImageGpu_32f_C1* src, const float& factor, iu::ImageGpu_32f_C1* dst, const IuRect& roi);
IUCORE_DLLAPI void mulC(const iu::ImageGpu_32f_C2* src, const float2& factor, iu::ImageGpu_32f_C2* dst, const IuRect& roi);
IUCORE_DLLAPI void mulC(const iu::ImageGpu_32f_C4* src, const float4& factor, iu::ImageGpu_32f_C4* dst, const IuRect& roi);

/** Multiplication of every pixel in a volume with a constant factor. (can be called in-place)
 * \param src Source volume.
 * \param factor Multiplication factor applied to each pixel.
 * \param dst Destination volume.
 *
 * \note supported gpu: 32f_C1
 */
// [gpu] multiplication with factor; Not-in-place; 32-bit;
IUCORE_DLLAPI void mulC(const iu::VolumeGpu_32f_C1* src, const float& factor, iu::VolumeGpu_32f_C1* dst);

/** In-place multiplication of every pixel with a constant factor.
 * \param factor Multiplication factor applied to each pixel.
 * \param srcdst Source and destination
 * \param roi Region of interest in the source/destination image
 *
 * \note supported gpu: 8u_C1, 8u_C4, 32f_C1, 32f_C4,
 */
// multiplication with factor; host; 8-bit
// :TODO: \todo implement inplace interfaces.

/** Addition to every pixel of a constant value. (can be called in-place)
 * \param src Source image.
 * \param val Value to be added.
 * \param dst Destination image.
 * \param roi Region of interest in the source and destination image
 *
 * \note supported gpu: 8u_C1, 8u_C4, 32f_C1, 32f_C4,
 */
// [gpu] add val; Not-in-place; 8-bit;
IUCORE_DLLAPI void addC(const iu::ImageGpu_8u_C1* src, const unsigned char& val, iu::ImageGpu_8u_C1* dst, const IuRect& roi);
IUCORE_DLLAPI void addC(const iu::ImageGpu_8u_C4* src, const uchar4& val, iu::ImageGpu_8u_C4* dst, const IuRect& roi);

// [gpu] add val; Not-in-place; 32-bit;
IUCORE_DLLAPI void addC(const iu::ImageGpu_32f_C1* src, const float& val, iu::ImageGpu_32f_C1* dst, const IuRect& roi);
IUCORE_DLLAPI void addC(const iu::ImageGpu_32f_C2* src, const float2& val, iu::ImageGpu_32f_C2* dst, const IuRect& roi);
IUCORE_DLLAPI void addC(const iu::ImageGpu_32f_C4* src, const float4& val, iu::ImageGpu_32f_C4* dst, const IuRect& roi);


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
IUCORE_DLLAPI void minMax(const ImageGpu_8u_C1* src, const IuRect& roi, unsigned char& min, unsigned char& max);
IUCORE_DLLAPI void minMax(const ImageGpu_8u_C4* src, const IuRect& roi, uchar4& min, uchar4& max);
// find min/max; device; 32-bit
IUCORE_DLLAPI void minMax(const ImageGpu_32f_C1* src, const IuRect& roi, float& min, float& max);
IUCORE_DLLAPI void minMax(const ImageGpu_32f_C2* src, const IuRect& roi, float2& min, float2& max);
IUCORE_DLLAPI void minMax(const ImageGpu_32f_C4* src, const IuRect& roi, float4& min, float4& max);


/** Finds the minimum and maximum value of a volume.
 * \param src Source image [device]
 * \param[out] min Minium value found in the source volume.
 * \param[out] max Maximum value found in the source volume.
 *
 * \note supported gpu: 32f_C1
 */
// find min/max; volume; device; 32-bit
IUCORE_DLLAPI void minMax(VolumeGpu_32f_C1* src, float& min, float& max);


/** Finds the minimum value of an image in a certain ROI and the minimums coordinates.
 * \param src Source image [device]
 * \param src_roi Region of interest in the source image.
 * \param[out] min minimum value found in the source image.
 * \param[out] x x-coordinate of minimum value
 * \param[out] y y-coordinate of minimum value
 *
 * \note supported gpu: 32f_C1
 */
// find min+coords; device; 32-bit
IUCORE_DLLAPI void min(const iu::ImageGpu_32f_C1* src, const IuRect&roi, float& min, int& x, int& y);

/** Finds the maximum value of an image in a certain ROI and the maximums coordinates.
 * \param src Source image [device]
 * \param src_roi Region of interest in the source image.
 * \param[out] max Maximum value found in the source image.
 * \param[out] x x-coordinate of maximum value
 * \param[out] y y-coordinate of maximum value
 *
 * \note supported gpu: 32f_C1
 */
// find max+coords; device; 32-bit
IUCORE_DLLAPI void max(const iu::ImageGpu_32f_C1* src, const IuRect&roi, float& max, int& x, int& y);


/** Computes the sum of pixels in a certain ROI of an image.
 * \param src Source image [device]
 * \param src_roi Region of interest in the source image.
 * \param[out] sum Contains computed sum.
 *
 * \note supported gpu: 8u_C1, 8u_C4, 32f_C1, 32f_C4,
 */
// compute sum; device; 8-bit
IUCORE_DLLAPI void summation(const ImageGpu_8u_C1* src, const IuRect& roi, long& sum);
//IUCORE_DLLAPI void summation(const ImageGpu_8u_C4* src, const IuRect& roi, long sum[4]);

// compute sum; device; 32-bit
IUCORE_DLLAPI void summation(const ImageGpu_32f_C1* src, const IuRect& roi, double& sum);
IUCORE_DLLAPI void summation(VolumeGpu_32f_C1* src, const IuCube& roi, double& sum);
//IUCORE_DLLAPI void summation(const ImageGpu_32f_C4* src, const IuRect& roi, double sum[4]);


/** Computes the L1 norm of differences between pixel values of two images. |src1-src2|
 * \param src1 Pointer to the first source image.
 * \param src2 Pointer to the second source image.
 * \param roi Region of interest in the source image.
 * \param norm Contains computed L1 norm.
 */
IUCORE_DLLAPI void normDiffL1(const ImageGpu_32f_C1* src1, const ImageGpu_32f_C1* src2, const IuRect& roi, double& norm);

/** Computes the L1 norm of differences between pixel values of an image and a constant value. |src-value|
 * \param src1 Pointer to the first source image.
 * \param value Subtrahend applied to every pixel on \a src image before calculating the L1 norm.
 * \param roi Region of interest in the source image.
 * \param norm Contains computed L1 norm.
 */
IUCORE_DLLAPI void normDiffL1(const ImageGpu_32f_C1* src, const float& value, const IuRect& roi, double& norm);

/** Computes the L2 norm of differences between pixel values of two images. ||src1-src2||
 * \param src1 Pointer to the first source image.
 * \param src2 Pointer to the second source image.
 * \param roi Region of interest in the source image.
 * \param norm Contains computed L2 norm.
 */
IUCORE_DLLAPI void normDiffL2(const ImageGpu_32f_C1* src1, const ImageGpu_32f_C1* src2, const IuRect& roi, double& norm);

/** Computes the L2 norm of differences between pixel values of an image and a constant value. ||src-value||
 * \param src Pointer to the first source image.
 * \param value Subtrahend applied to every pixel on \a src image before calculating the L2 norm.
 * \param roi Region of interest in the source image.
 * \param norm Contains computed L2 norm.
 */
IUCORE_DLLAPI void normDiffL2(const ImageGpu_32f_C1* src, const float& value, const IuRect& roi, double& norm);

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
IUCORE_DLLAPI void mse(const iu::ImageGpu_32f_C1* src, const iu::ImageGpu_32f_C1* reference, const IuRect& roi, double& mse);

/** Computes the structural similarity index between the src and the reference image.
 * \param src Pointer to the source image.
 * \param reference Pointer to the refernce image.
 * \param roi Region of interest in the source and reference image.
 * \param mse Contains the computed  structural similarity index.
 */
IUCORE_DLLAPI void ssim(const iu::ImageGpu_32f_C1* src, const iu::ImageGpu_32f_C1* reference, const IuRect& roi, double& ssim);

/** @} */ // end of Error Measurements


/* ***************************************************************************
     HISTOGRAMS
 * ***************************************************************************/

//////////////////////////////////////////////////////////////////////////////
/** @defgroup Histograms
 *  @ingroup Math
 *  TODO more detailed docu
 *  @{
 */

/** Computes the color histogram of an already binned image
 * \param binned_image Already binned image (make sure this fits to actual hist
 * \param mask         A mask image (only pixels where the mask value equals mask_val will be taken into account)
 * \param hist         The output histogram
 * \param mask_val     The mask value
 */
IUCORE_DLLAPI void colorHistogram(const iu::ImageGpu_8u_C4* binned_image, const iu::ImageGpu_8u_C1* mask,
                                  iu::VolumeGpu_32f_C1* hist, unsigned char mask_val);

/** @} */ // end of Histograms



/** @} */ // end of Math

} // namespace iu

#endif // IU_MATH_MODULE_H
