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
 * Module      : IO;
 * Class       : none
 * Language    : C++
 * Description : Definition of image I/O functions for pgm images
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUPRIVATE_IMAGEIOPGM_H
#define IUPRIVATE_IMAGEIOPGM_H

#include <iucore/coredefs.h>
#include <iucore/memorydefs.h>
#include <string>

namespace iuprivate {

/* Read images from disc. */
iu::ImageCpu_16u_C1* imread_16u_C1(const std::string& filename);
iu::ImageCpu_32f_C1* imread_16u32f_C1(const std::string& filename, int max_val=65536);
iu::ImageGpu_32f_C1* imread_cu16u32f_C1(const std::string& filename, int max_val=65536);

//iu::ImageGpu_32f_C1* imread_16u_C4(const std::string& filename);

} // namespace iuprivate

#endif // IUPRIVATE_IMAGEIOPGM_H
