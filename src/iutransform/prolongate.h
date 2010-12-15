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
 * Description : Declaration of prolongate transformations
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUPRIVATE_PROLONGATE_H
#define IUPRIVATE_PROLONGATE_H

#include <iucore/coredefs.h>
#include <iucore/memorydefs.h>

namespace iuprivate {

IuStatus prolongate(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst,
                    IuInterpolationType interpolation = IU_INTERPOLATE_NEAREST);
IuStatus prolongate(const iu::ImageGpu_32f_C2* src, iu::ImageGpu_32f_C2* dst,
                    IuInterpolationType interpolation = IU_INTERPOLATE_NEAREST);
IuStatus prolongate(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst,
                    IuInterpolationType interpolation = IU_INTERPOLATE_NEAREST);



}  // namespace iuprivate

#endif // IUPRIVATE_PROLONGATE_H
