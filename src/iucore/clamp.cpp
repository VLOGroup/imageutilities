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

extern IuStatus cuClamp(const float& min, const float& max,
                        iu::ImageGpu_32f_C1 *srcdst, const IuRect &roi);


// [device] clamping of pixels to [min,max]; 32-bit
void clamp(const float& min, const float& max,
           iu::ImageGpu_32f_C1 *srcdst, const IuRect &roi)
{
  IuStatus status = cuClamp(min, max, srcdst, roi);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}


} // namespace iuprivate
