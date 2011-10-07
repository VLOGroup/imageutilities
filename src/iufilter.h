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
