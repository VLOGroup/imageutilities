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
 * Module      : IO Module
 * Class       : Wrapper
 * Language    : C
 * Description : Implementation of public interfaces to IO module
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#include "iuio.h"
#include "iuio/imageiopgm.h"

namespace iu {

/* ***************************************************************************
     read 16-bit 2d image
 * ***************************************************************************/

iu::ImageCpu_16u_C1* imread_16u_C1(const std::string& filename)
{ return iuprivate::imread_16u_C1(filename); }

iu::ImageCpu_32f_C1* imread_12u32f_C1(const std::string& filename)
{ return iuprivate::imread_12u32f_C1(filename); }

iu::ImageNpp_32f_C1* imread_cu12u32f_C1(const std::string& filename)
{ return iuprivate::imread_cu12u32f_C1(filename); }


} // namespace iu
