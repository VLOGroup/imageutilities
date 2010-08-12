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
 * Description : Definition of geometric transformations
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUPRIVATE_RESAMPLE_H
#define IUPRIVATE_RESAMPLE_H

#include "core/coredefs.h"
#include "core/memorydefs.h"

namespace iuprivate {

// device; 32-bit; 1-channel
void reduce(const iu::ImageNpp_32f_C1* src, iu::ImageNpp_32f_C1* dst, bool prefilter);

} // namespace iuprivate

#endif // IUPRIVATE_RESAMPLE_H
