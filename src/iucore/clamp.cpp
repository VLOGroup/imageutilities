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
 * Module      : Core
 * Class       : none
 * Language    : C
 * Description : Implementation of memory modifications
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#include "clamp.h"


namespace iuprivate {

extern void cuClamp(const float& min, const float& max,
                        iu::ImageGpu_32f_C1 *srcdst, const IuRect &roi);


// [device] clamping of pixels to [min,max]; 32-bit
void clamp(const float& min, const float& max,
           iu::ImageGpu_32f_C1 *srcdst, const IuRect &roi)
{
  cuClamp(min, max, srcdst, roi);
}


} // namespace iuprivate
