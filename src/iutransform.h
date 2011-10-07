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

#ifndef IU_TRANSFORM_MODULE_H
#define IU_TRANSFORM_MODULE_H

#include "iudefs.h"

namespace iu {

/** \defgroup Geometric Transformation
 *  \brief Geometric image transformations
 *  TODO more detailed docu
 *  @{
 */

/* ***************************************************************************
     Geometric Transformations
 * ***************************************************************************/

//////////////////////////////////////////////////////////////////////////////
/* ***************************************************************************
     Image resize
 * ***************************************************************************/
/** @defgroup Image Transformations
 *  @ingroup Geometric Transformation
 *  TODO more detailed docu
 *  @{
 */

/** Image reduction.
 * \brief Scaling the image \a src down to the size of \a dst.
 * \param[in] src Source image [device]
 * \param[out] dst Destination image [device]
 * \param[in] interpolation The type of interpolation used for scaling down the image.
 * \param[in] gauss_prefilter Toggles gauss prefiltering. The sigma and kernel size is chosen dependent on the scale factor.
 * \param[in] bicubic_bspline_prefilter Only reasonable for cubic (spline) interpolation.
 *
 * \note The bcubic_bspline_prefilter yields sharper results when switched on. Note that this only works nicely with a scale_factor=0.5f.
 */
IUCORE_DLLAPI void reduce(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst,
                      IuInterpolationType interpolation = IU_INTERPOLATE_LINEAR,
                      bool gauss_prefilter = true, bool bicubic_bspline_prefilter = false);

/** Image prolongation.
 * \brief Scaling the image \a src up to the size of \a dst.
 * \param[in] src Source image [device]
 * \param[out] dst Destination image [device]
 * \param[in] interpolation The type of interpolation used for scaling up the image.
 * :TODO: \param[in] bicubic_bspline_prefilter Only reasonable for cubic (spline) interpolation.
 *
 * \note The bcubic_bspline_prefilter yields sharper results when switched on. Note that this only works nicely with a scale_factor=0.5f.
 */
IUCORE_DLLAPI void prolongate(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst,
                          IuInterpolationType interpolation = IU_INTERPOLATE_NEAREST);
IUCORE_DLLAPI void prolongate(const iu::ImageGpu_32f_C2* src, iu::ImageGpu_32f_C2* dst,
                          IuInterpolationType interpolation = IU_INTERPOLATE_NEAREST);
IUCORE_DLLAPI void prolongate(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst,
                          IuInterpolationType interpolation = IU_INTERPOLATE_NEAREST);

/** Image remapping (warping).
 * \brief Remapping the image \a src with the given disparity fields dx, dy.
 * \param[in] src Source image [device]
 * \param[in] dx_map Disparities (dense) in x direction [device]
 * \param[in] dy_map Disparities (dense) in y direction [device]
 * \param[out] dst Destination image [device]
 * \param[in] interpolation The type of interpolation used for scaling up the image.
 * :TODO: \param[in] bicubic_bspline_prefilter Only reasonable for cubic (spline) interpolation.
 *
 * \note The bcubic_bspline_prefilter yields sharper results when switched on. Note that this only works nicely with a scale_factor=0.5f.
 */
IUCORE_DLLAPI void remap(iu::ImageGpu_8u_C1* src,
                     iu::ImageGpu_32f_C1* dx_map, iu::ImageGpu_32f_C1* dy_map,
                     iu::ImageGpu_8u_C1* dst,
                     IuInterpolationType interpolation = IU_INTERPOLATE_LINEAR);
IUCORE_DLLAPI void remap(iu::ImageGpu_32f_C1* src,
                     iu::ImageGpu_32f_C1* dx_map, iu::ImageGpu_32f_C1* dy_map,
                     iu::ImageGpu_32f_C1* dst,
                     IuInterpolationType interpolation = IU_INTERPOLATE_LINEAR);
//IUCORE_DLLAPI IuStatus remap(iu::ImageGpu_32f_C2* src,
//                     iu::ImageGpu_32f_C1* dx_map, iu::ImageGpu_32f_C1* dy_map,
//                     iu::ImageGpu_32f_C2* dst,
//                     IuInterpolationType interpolation = IU_INTERPOLATE_LINEAR);
//IUCORE_DLLAPI IuStatus remap(iu::ImageGpu_32f_C4* src,
//                     iu::ImageGpu_32f_C1* dx_map, iu::ImageGpu_32f_C1* dy_map,
//                     iu::ImageGpu_32f_C4* dst,
//                     IuInterpolationType interpolation = IU_INTERPOLATE_LINEAR);


/** @} */ // end of Image Transformations


} // namespace iu


#endif // IU_TRANSFORM_MODULE_H
