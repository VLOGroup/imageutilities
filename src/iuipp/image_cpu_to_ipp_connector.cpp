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
 * Module      : IPP-to-CPU Connector
 * Class       : none
 * Language    : C
 * Description : Implementation of some memory conversions between cpu and ipp images
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

//
//  W A R N I N G
//  -------------
//
// This file is not part of the IU API.  It exists purely as an
// implementation detail.  This header file may change from version to
// version without notice, or even be removed.
//

#include "memorydefs_ipp.h"
#include "image_cpu_to_ipp_connector.h"

namespace iuprivate {

iu::ImageIpp_8u_C1* convertToIpp_8u_C1(iu::ImageCpu_8u_C1* src)
{
  iu::ImageIpp_8u_C1* img = new iu::ImageIpp_8u_C1((Ipp8u*)src->data(), src->width(), src->height(),
                                                     src->pitch(), true);
  return img;
}

iu::ImageIpp_8u_C3* convertToIpp_8u_C3(iu::ImageCpu_8u_C3* src)
{
  iu::ImageIpp_8u_C3* img = new iu::ImageIpp_8u_C3((Ipp8u*)src->data(), src->width(), src->height(),
                                                   src->pitch(), true);
  return img;
}

iu::ImageIpp_8u_C4* convertToIpp_8u_C4(iu::ImageCpu_8u_C4* src)
{
  iu::ImageIpp_8u_C4* img = new iu::ImageIpp_8u_C4((Ipp8u*)src->data(), src->width(), src->height(),
                                                   src->pitch(), true);
  return img;
}

iu::ImageIpp_32f_C1* convertToIpp_32f_C1(iu::ImageCpu_32f_C1* src)
{
  iu::ImageIpp_32f_C1* img = new iu::ImageIpp_32f_C1((Ipp32f*)src->data(), src->width(), src->height(),
                                                     src->pitch(), true);
  return img;
}

iu::ImageIpp_32f_C3* convertToIpp_32f_C3(iu::ImageCpu_32f_C3* src)
{
  iu::ImageIpp_32f_C3* img = new iu::ImageIpp_32f_C3((Ipp32f*)src->data(), src->width(), src->height(),
                                                     src->pitch(), true);
  return img;
}

iu::ImageIpp_32f_C4* convertToIpp_32f_C4(iu::ImageCpu_32f_C4* src)
{
  iu::ImageIpp_32f_C4* img = new iu::ImageIpp_32f_C4((Ipp32f*)src->data(), src->width(), src->height(),
                                                     src->pitch(), true);
  return img;
}

} // namespace iuprivate
