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

#ifndef IMAGE_IPP_TO_CPU_CONNECTOR_H
#define IMAGE_IPP_TO_CPU_CONNECTOR_H

//
//  W A R N I N G
//  -------------
//
// This file is not part of the IU API.  It exists purely as an
// implementation detail.  This header file may change from version to
// version without notice, or even be removed.
//

#include <iudefs.h>
//#include <iucore.h>
#include "memorydefs_ipp.h"

namespace iu {

/** Converts the ImageIPP structure to an ImageCPU type. The data keeps owned with the src image.
 * @param[in] src Source image which still owns the data buffer after the conversion.
 * @returns ImageCpu type of the corresponding memory. No data owned from this instance.
 * @attention The returned image structured only holds a pointer to the original data structure.
 */
inline iu::ImageCpu_32f_C1* convertToCpu_32f_C1(iu::ImageIpp_32f_C1* src)
{
  iu::ImageCpu_32f_C1* img = new iu::ImageCpu_32f_C1(src->data(), src->width(), src->height(),
                                                     src->pitch(), true);
  return img;
}

/** Converts the ImageIPP structure to an ImageCPU type. The data keeps owned with the src image.
 * @param[in] src Source image which still owns the data buffer after the conversion.
 * @returns ImageCpu type of the corresponding memory. No data owned from this instance.
 * @attention The returned image structured only holds a pointer to the original data structure.
 */
inline iu::ImageCpu_32f_C3* convertToCpu_32f_C3(iu::ImageIpp_32f_C3* src)
{
  iu::ImageCpu_32f_C3* img = new iu::ImageCpu_32f_C3((float3*)src->data(), src->width(), src->height(),
                                                     src->pitch(), true);
  return img;
}

/** Converts the ImageIPP structure to an ImageCPU type. The data keeps owned with the src image.
 * @param[in] src Source image which still owns the data buffer after the conversion.
 * @returns ImageCpu type of the corresponding memory. No data owned from this instance.
 * @attention The returned image structured only holds a pointer to the original data structure.
 */
inline iu::ImageCpu_32f_C4* convertToCpu_32f_C4(iu::ImageIpp_32f_C4* src)
{
  iu::ImageCpu_32f_C4* img = new iu::ImageCpu_32f_C4((float4*)src->data(), src->width(), src->height(),
                                                     src->pitch(), true);
  return img;
}

/** Converts the ImageIPP structure to an ImageCPU type. The data keeps owned with the src image.
 * @param[in] src Source image which still owns the data buffer after the conversion.
 * @returns ImageCpu type of the corresponding memory. No data owned from this instance.
 * @attention The returned image structured only holds a pointer to the original data structure.
 */
inline iu::ImageCpu_8u_C1* convertToCpu_8u_C1(iu::ImageIpp_8u_C1* src)
{
  iu::ImageCpu_8u_C1* img = new iu::ImageCpu_8u_C1(src->data(), src->width(), src->height(),
                                                     src->pitch(), true);
  return img;
}

/** Converts the ImageIPP structure to an ImageCPU type. The data keeps owned with the src image.
 * @param[in] src Source image which still owns the data buffer after the conversion.
 * @returns ImageCpu type of the corresponding memory. No data owned from this instance.
 * @attention The returned image structured only holds a pointer to the original data structure.
 */
inline iu::ImageCpu_8u_C3* convertToCpu_8u_C3(iu::ImageIpp_8u_C3* src)
{
  iu::ImageCpu_8u_C3* img = new iu::ImageCpu_8u_C3((uchar3*)src->data(), src->width(), src->height(),
                                                   src->pitch(), true);
  return img;
}

/** Converts the ImageIPP structure to an ImageCPU type. The data keeps owned with the src image.
 * @param[in] src Source image which still owns the data buffer after the conversion.
 * @returns ImageCpu type of the corresponding memory. No data owned from this instance.
 * @attention The returned image structured only holds a pointer to the original data structure.
 */
inline iu::ImageCpu_8u_C4* convertToCpu_8u_C4(iu::ImageIpp_8u_C4* src)
{
  iu::ImageCpu_8u_C4* img = new iu::ImageCpu_8u_C4((uchar4*)src->data(), src->width(), src->height(),
                                                   src->pitch(), true);
  return img;
}

///////////////////////////////////////////////////////////////////////////////


/** Converts the ImageCPU structure to an ImageIPP type. The data keeps owned with the src image.
 * @param[in] src Source image which still owns the data buffer after the conversion.
 * @returns ImageCpu type of the corresponding memory. No data owned from this instance.
 * @attention The returned image structured only holds a pointer to the original data structure.
 */
inline iu::ImageIpp_32f_C1* convertToIpp_32f_C1(iu::ImageCpu_32f_C1* src)
{
  iu::ImageIpp_32f_C1* img = new iu::ImageIpp_32f_C1((Ipp32f*)src->data(), src->width(), src->height(),
                                                     src->pitch(), true);
  return img;
}

/** Converts the ImageCPU structure to an ImageIPP type. The data keeps owned with the src image.
 * @param[in] src Source image which still owns the data buffer after the conversion.
 * @returns ImageCpu type of the corresponding memory. No data owned from this instance.
 * @attention The returned image structured only holds a pointer to the original data structure.
 */
inline iu::ImageIpp_32f_C3* convertToIpp_32f_C3(iu::ImageCpu_32f_C3* src)
{
  iu::ImageIpp_32f_C3* img = new iu::ImageIpp_32f_C3((Ipp32f*)src->data(), src->width(), src->height(),
                                                     src->pitch(), true);
  return img;
}

/** Converts the ImageCPU structure to an ImageIPP type. The data keeps owned with the src image.
 * @param[in] src Source image which still owns the data buffer after the conversion.
 * @returns ImageCpu type of the corresponding memory. No data owned from this instance.
 * @attention The returned image structured only holds a pointer to the original data structure.
 */
inline iu::ImageIpp_32f_C4* convertToIpp_32f_C4(iu::ImageCpu_32f_C4* src)
{
  iu::ImageIpp_32f_C4* img = new iu::ImageIpp_32f_C4((Ipp32f*)src->data(), src->width(), src->height(),
                                                     src->pitch(), true);
  return img;
}

/** Converts the ImageCPU structure to an ImageIPP type. The data keeps owned with the src image.
 * @param[in] src Source image which still owns the data buffer after the conversion.
 * @returns ImageCpu type of the corresponding memory. No data owned from this instance.
 * @attention The returned image structured only holds a pointer to the original data structure.
 */
inline iu::ImageIpp_8u_C1* convertToIpp_8u_C1(iu::ImageCpu_8u_C1* src)
{
  iu::ImageIpp_8u_C1* img = new iu::ImageIpp_8u_C1((Ipp8u*)src->data(), src->width(), src->height(),
                                                     src->pitch(), true);
  return img;
}

/** Converts the ImageCPU structure to an ImageIPP type. The data keeps owned with the src image.
 * @param[in] src Source image which still owns the data buffer after the conversion.
 * @returns ImageCpu type of the corresponding memory. No data owned from this instance.
 * @attention The returned image structured only holds a pointer to the original data structure.
 */
inline iu::ImageIpp_8u_C3* convertToIpp_8u_C3(iu::ImageCpu_8u_C3* src)
{
  iu::ImageIpp_8u_C3* img = new iu::ImageIpp_8u_C3((Ipp8u*)src->data(), src->width(), src->height(),
                                                   src->pitch(), true);
  return img;
}

/** Converts the ImageCPU structure to an ImageIPP type. The data keeps owned with the src image.
 * @param[in] src Source image which still owns the data buffer after the conversion.
 * @returns ImageCpu type of the corresponding memory. No data owned from this instance.
 * @attention The returned image structured only holds a pointer to the original data structure.
 */
inline iu::ImageIpp_8u_C4* convertToIpp_8u_C4(iu::ImageCpu_8u_C4* src)
{
  iu::ImageIpp_8u_C4* img = new iu::ImageIpp_8u_C4((Ipp8u*)src->data(), src->width(), src->height(),
                                                   src->pitch(), true);
  return img;
}


} // namespace iuprivate

#endif // IMAGE_IPP_TO_CPU_CONNECTOR_H
