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
/* ***************************************************************************
     Denoising
 * ***************************************************************************/
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
IUCORE_DLLAPI void filterMedian3x3(const ImageGpu_32f_C1* src, ImageGpu_32f_C1* dst, const IuRect& roi);

/** Gaussian Convolution
 * \brief Filters a device image using a Gaussian filter
 * \param src Source image [device].
 * \param dst Destination image [device]
 * \param roi Region of interest in the dsetination image.
 * \param sigma Controls the amount of smoothing
 * \param kernel_size Sets the size of the used Gaussian kernel. If =0 the size is calculated.
 */
IUCORE_DLLAPI void filterGauss(const ImageGpu_32f_C1* src, ImageGpu_32f_C1* dst, const IuRect& roi,
                               float sigma, int kernel_size=0);
IUCORE_DLLAPI void filterGauss(const VolumeGpu_32f_C1* src, VolumeGpu_32f_C1* dst,
                               float sigma, int kernel_size=0);
IUCORE_DLLAPI void filterGauss(const ImageGpu_32f_C4* src, ImageGpu_32f_C4* dst, const IuRect& roi,
                               float sigma, int kernel_size=0);

/** Bilateral Filtering
 * \brief Filters a device image using a Bilateral filter
 * \param src Source image [device].
 * \param dst Destination image [device]
 * \param roi Region of interest in the dsetination image.
 * \param prior Image to compute the weights to guide the bilat filter (if =0 the src image is used).
 * \param iters Number of filtering iterations.
 * \param sigma_spatial Spatial (gaussian) influence.
 * \param sigma_range Bilat influence.
 * \param radius Filter radius.
 */
IUCORE_DLLAPI void filterBilateral(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst, const IuRect& roi,
                                   const iu::ImageGpu_32f_C1* prior=NULL, const int iters=1,
                                   const float sigma_spatial=4.0f, const float sigma_range=0.1f,
                                   const int radius=5);
IUCORE_DLLAPI void filterBilateral(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst, const IuRect& roi,
                                   const iu::ImageGpu_32f_C4* prior, const int iters=1,
                                   const float sigma_spatial=4.0f, const float sigma_range=0.1f,
                                   const int radius=5);
IUCORE_DLLAPI void filterBilateral(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst, const IuRect& roi,
                                   const iu::ImageGpu_32f_C4* prior=NULL, const int iters=1,
                                   const float sigma_spatial=4.0f, const float sigma_range=0.1f,
                                   const int radius=5);

/** @} */ // end of Denoising


/* ***************************************************************************
     edge calculation
 * ***************************************************************************/


IUCORE_DLLAPI void filterEdge(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C2* dst, const IuRect& roi);

IUCORE_DLLAPI void filterEdge(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst, const IuRect& roi,
                              float alpha, float beta, float minval);

IUCORE_DLLAPI void filterEdge(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C2* dst, const IuRect& roi,
                              float alpha, float beta, float minval);

IUCORE_DLLAPI void filterEdge(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C4* dst, const IuRect& roi,
                              float alpha, float beta, float minval);

IUCORE_DLLAPI void filterEdge(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C1* dst, const IuRect& roi,
                              float alpha, float beta, float minval);

IUCORE_DLLAPI void filterEdge(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C2* dst, const IuRect& roi,
                              float alpha, float beta, float minval);

IUCORE_DLLAPI void filterEdge(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst, const IuRect& roi,
                              float alpha, float beta, float minval);



//////////////////////////////////////////////////////////////////////////////
/* ***************************************************************************
     other filters
 * ***************************************************************************/

IUCORE_DLLAPI void cubicBSplinePrefilter(iu::ImageGpu_32f_C1* srcdst);


/** @} */ // end of Filter Module

} // namespace iu


#endif // IU_FILTER_MODULE_H
