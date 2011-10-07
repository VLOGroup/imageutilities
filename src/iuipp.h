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
 * Module      : Core Module for IPP-Connector
 * Class       : none
 * Language    : C
 * Description : Public interfaces to the IPP module
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger\icg.tugraz.at
 *
 */
#ifndef IU_IPP_H
#define IU_IPP_H

#include "iuipp/copy_ipp.h"

namespace iu {

/** Converts the ImageIpp structure to an ImageCpu type. The data keeps owned with the src image.
 * @param[in] src Source image which still owns the data buffer after the conversion.
 * @returns ImageIpp type of the corresponding memory. No data owned from this instance.
 * @attention The returned image structured only holds a pointer to the original data structure.
 */
IUIPP_DLLAPI iu::ImageCpu_8u_C1* convertToCpu_8u_C1(iu::ImageIpp_8u_C1* src);
IUIPP_DLLAPI iu::ImageCpu_8u_C3* convertToCpu_8u_C3(iu::ImageIpp_8u_C3* src);
IUIPP_DLLAPI iu::ImageCpu_8u_C4* convertToCpu_8u_C4(iu::ImageIpp_8u_C4* src);
IUIPP_DLLAPI iu::ImageCpu_32f_C1* convertToCpu_32f_C1(iu::ImageIpp_32f_C1* src);
IUIPP_DLLAPI iu::ImageCpu_32f_C3* convertToCpu_32f_C3(iu::ImageIpp_32f_C3* src);
IUIPP_DLLAPI iu::ImageCpu_32f_C4* convertToCpu_32f_C4(iu::ImageIpp_32f_C4* src);

/** Converts the ImageCPU structure to an ImageIpp type. The data keeps owned with the src image.
 * @param[in] src Source image which still owns the data buffer after the conversion.
 * @returns ImageCpu type of the corresponding memory. No data owned from this instance.
 * @attention The returned image structured only holds a pointer to the original data structure.
 */
IUIPP_DLLAPI iu::ImageIpp_8u_C1* convertToIpp_8u_C1(iu::ImageCpu_8u_C1* src);
IUIPP_DLLAPI iu::ImageIpp_8u_C3* convertToIpp_8u_C3(iu::ImageCpu_8u_C3* src);
IUIPP_DLLAPI iu::ImageIpp_8u_C4* convertToIpp_8u_C4(iu::ImageCpu_8u_C4* src);
IUIPP_DLLAPI iu::ImageIpp_32f_C1* convertToIpp_32f_C1(iu::ImageCpu_32f_C1* src);
IUIPP_DLLAPI iu::ImageIpp_32f_C3* convertToIpp_32f_C3(iu::ImageCpu_32f_C3* src);
IUIPP_DLLAPI iu::ImageIpp_32f_C4* convertToIpp_32f_C4(iu::ImageCpu_32f_C4* src);

} // namespace iu

#endif // IU_IPP_H
