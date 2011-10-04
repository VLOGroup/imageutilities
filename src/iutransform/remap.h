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
 * Module      : Geometric Transform
 * Class       : none
 * Language    : C++
 * Description : Declaration of remap transformations (with dense disparities)
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUPRIVATE_REMAP_H
#define IUPRIVATE_REMAP_H

#include <iucore/coredefs.h>
#include <iucore/memorydefs.h>

namespace iuprivate {

void remap(iu::ImageGpu_8u_C1* src,
               iu::ImageGpu_32f_C1* dx_map, iu::ImageGpu_32f_C1* dy_map,
               iu::ImageGpu_8u_C1* dst, IuInterpolationType interpolation);
void remap(iu::ImageGpu_32f_C1* src,
               iu::ImageGpu_32f_C1* dx_map, iu::ImageGpu_32f_C1* dy_map,
               iu::ImageGpu_32f_C1* dst, IuInterpolationType interpolation);
//IuStatus remap(iu::ImageGpu_32f_C2* src,
//               iu::ImageGpu_32f_C1* dx_map, iu::ImageGpu_32f_C1* dy_map,
//               iu::ImageGpu_32f_C2* dst, IuInterpolationType interpolation);
//IuStatus remap(iu::ImageGpu_32f_C4* src,
//               iu::ImageGpu_32f_C1* dx_map, iu::ImageGpu_32f_C1* dy_map,
//               iu::ImageGpu_32f_C4* dst, IuInterpolationType interpolation);

}  // namespace iuprivate

#endif // IUPRIVATE_REMAP_H
