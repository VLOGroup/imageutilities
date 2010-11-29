/*
 * Copyright (c) ICG. All rights reserved.
 *
 * Institute for Computer Graphics and Vision
 * Graz University of Technology / Austria
 *
 *
 * This software is distributed WITHOUT ANY WARRANTY {iuprivate::convertToCpp_8u_C1(src);} without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 * PURPOSE.  See the above copyright notices for more information.
 *
 *
 * Project     : ImageUtilities
 * Module      : Core Module for IPP-Connector
 * Class       : none
 * Language    : C
 * Description : Implementation of public interfaces to the IPP module
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger\icg.tugraz.at
 *
 */

#include "iuipp/image_ipp_to_cpu_connector.h"
#include "iuipp/image_cpu_to_ipp_connector.h"
#include "iuipp.h"

namespace iu {

iu::ImageCpu_8u_C1* convertToCpu_8u_C1(iu::ImageIpp_8u_C1* src) {return iuprivate::convertToCpu_8u_C1(src);}
iu::ImageCpu_8u_C3* convertToCpu_8u_C3(iu::ImageIpp_8u_C3* src) {return iuprivate::convertToCpu_8u_C3(src);}
iu::ImageCpu_8u_C4* convertToCpu_8u_C4(iu::ImageIpp_8u_C4* src) {return iuprivate::convertToCpu_8u_C4(src);}
iu::ImageCpu_32f_C1* convertToCpu_32f_C1(iu::ImageIpp_32f_C1* src) {return iuprivate::convertToCpu_32f_C1(src);}
iu::ImageCpu_32f_C3* convertToCpu_32f_C3(iu::ImageIpp_32f_C3* src) {return iuprivate::convertToCpu_32f_C3(src);}
iu::ImageCpu_32f_C4* convertToCpu_32f_C4(iu::ImageIpp_32f_C4* src) {return iuprivate::convertToCpu_32f_C4(src);}

iu::ImageIpp_8u_C1* convertToIpp_8u_C1(iu::ImageCpu_8u_C1* src) {return iuprivate::convertToIpp_8u_C1(src);}
iu::ImageIpp_8u_C3* convertToIpp_8u_C3(iu::ImageCpu_8u_C3* src) {return iuprivate::convertToIpp_8u_C3(src);}
iu::ImageIpp_8u_C4* convertToIpp_8u_C4(iu::ImageCpu_8u_C4* src) {return iuprivate::convertToIpp_8u_C4(src);}
iu::ImageIpp_32f_C1* convertToIpp_32f_C1(iu::ImageCpu_32f_C1* src) {return iuprivate::convertToIpp_32f_C1(src);}
iu::ImageIpp_32f_C3* convertToIpp_32f_C3(iu::ImageCpu_32f_C3* src) {return iuprivate::convertToIpp_32f_C3(src);}
iu::ImageIpp_32f_C4* convertToIpp_32f_C4(iu::ImageCpu_32f_C4* src) {return iuprivate::convertToIpp_32f_C4(src);}

} // namespace iu

