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
 * Description : Definition of some memory conversions so that an ImageIPP can be used directly instead of an ImageCPU
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

#include <iudefs.h>
#include "memorydefs_ipp.h"

namespace iuprivate {

iu::ImageCpu_8u_C1* convertToCpu_8u_C1(iu::ImageIpp_8u_C1* src)
{
  iu::ImageCpu_8u_C1* img = new iu::ImageCpu_8u_C1(src->data(), src->width(), src->height(),
                                                     src->pitch(), true);
  return img;
}

iu::ImageCpu_8u_C3* convertToCpu_8u_C3(iu::ImageIpp_8u_C3* src)
{
  iu::ImageCpu_8u_C3* img = new iu::ImageCpu_8u_C3((uchar3*)src->data(), src->width(), src->height(),
                                                   src->pitch(), true);
  return img;
}

iu::ImageCpu_8u_C4* convertToCpu_8u_C4(iu::ImageIpp_8u_C4* src)
{
  iu::ImageCpu_8u_C4* img = new iu::ImageCpu_8u_C4((uchar4*)src->data(), src->width(), src->height(),
                                                   src->pitch(), true);
  return img;
}



iu::ImageCpu_32f_C1* convertToCpu_32f_C1(iu::ImageIpp_32f_C1* src)
{
  iu::ImageCpu_32f_C1* img = new iu::ImageCpu_32f_C1(src->data(), src->width(), src->height(),
                                                     src->pitch(), true);
  return img;
}

iu::ImageCpu_32f_C3* convertToCpu_32f_C3(iu::ImageIpp_32f_C3* src)
{
  iu::ImageCpu_32f_C3* img = new iu::ImageCpu_32f_C3((float3*)src->data(), src->width(), src->height(),
                                                     src->pitch(), true);
  return img;
}

iu::ImageCpu_32f_C4* convertToCpu_32f_C4(iu::ImageIpp_32f_C4* src)
{
  iu::ImageCpu_32f_C4* img = new iu::ImageCpu_32f_C4((float4*)src->data(), src->width(), src->height(),
                                                     src->pitch(), true);
  return img;
}

} // namespace iuprivate