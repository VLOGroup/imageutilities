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
 * Module      : Filter Module
 * Class       : Wrapper
 * Language    : C
 * Description : Public interfaces to filter module
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IU_FILTER_MODULE_H
#define IU_FILTER_MODULE_H

#include "iudefs.h"

namespace iu {

/** \defgroup Filter
 *  \brief The filter module.
 *  TODO more detailed docu
 *  @{
 */

/* ***************************************************************************
     Filters
 * ***************************************************************************/

//////////////////////////////////////////////////////////////////////////////
/** @defgroup Denoising
 *  @ingroup Filter
 *  TODO more detailed docu
 *  @{
 */

/** 2D Median Filter
 * \brief Filters a device image using a 3x3 median filter
 * \param src Source image [device].
 * \param dst Destination image [device]
 * \param roi Region of interest in the dsetination image.
 */
IU_DLLAPI void filterMedian3x3(const ImageGpu_32f_C1* src, ImageGpu_32f_C1* dst, const IuRect& roi);

/** Gaussian Convolution
 * \brief Filters a device image using a Gaussian filter
 * \param src Source image [device].
 * \param dst Destination image [device]
 * \param roi Region of interest in the dsetination image.
 * \param sigma Controls the amount of smoothing
 * \param kernel_size Sets the size of the used Gaussian kernel. If =0 the size is calculated.
 */
IU_DLLAPI void filterGauss(const ImageGpu_32f_C1* src, ImageGpu_32f_C1* dst, const IuRect& roi,
                           float sigma, int kernel_size=0);

/** ROF Filter
 * \brief Applies a ROF filters to a device image
 * \param src Source image [device].
 * \param dst Destination image [device]
 * \param roi Region of interest in the dsetination image.
 * \param lambda Controls the amount of smoothing (weights data to regularization term).
 * \param iterations Sets the amount of iterations for the primal-dual optimization.
 */IU_DLLAPI void filterRof(const ImageGpu_32f_C1* src, ImageGpu_32f_C1* dst, const IuRect& roi,
                            float lambda, int iterations=50);

/** @} */ // end of Denoising

//////////////////////////////////////////////////////////////////////////////
/** @defgroup Structure-Texture Decomposition
 *  @ingroup Filter
 *  TODO more detailed docu
 *  @{
 */

/** Structure-texture decomposition using a gaussian filter to separate structure and texture
 * TODO parameter documentation
 */
IU_DLLAPI void decomposeStructureTextureGauss(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst,
                                              const IuRect& roi, float weight=0.8f, float sigma=10.0f, int kernel_size=0);

/** Structure-texture decomposition using a ROF filter to separate structure and texture
 * TODO parameter documentation
 */
IU_DLLAPI void decomposeStructureTextureRof(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst,
                                            const IuRect& roi, float weight=0.8f, float lambda=1.0f, int iterations=100);

/** @} */ // end of Structure-Texture Decomposition

/** @} */ // end of Filter Module

} // namespace iu

#endif // IU_FILTER_MODULE_H
