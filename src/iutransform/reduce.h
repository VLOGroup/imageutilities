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
 * Module      : Geometric Transformations
 * Class       : none
 * Language    : C++
 * Description : Definition of reduction transformations
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUPRIVATE_REDUCE_H
#define IUPRIVATE_REDUCE_H

#include <iucore/coredefs.h>
#include <iucore/memorydefs.h>

namespace iuprivate {

// device; 32-bit; 1-channel
IuStatus reduce(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst,
            IuInterpolationType interpolation = IU_INTERPOLATE_LINEAR,
            bool gauss_prefilter = true, bool bicubic_bspline_prefilter = false);

} // namespace iuprivate

#endif // IUPRIVATE_REDUCE_H
