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
 * Module      : IO
 * Class       : none
 * Language    : C++
 * Description : Definition of image I/O functions
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */


#ifndef IUIO_IMAGEIO_H
#define IUIO_IMAGEIO_H

#include <iucore/coredefs.h>
#include <iucore/memorydefs.h>
#include <string>

namespace iuprivate {

iu::ImageCpu_32f_C1* imread_32f_C1(const std::string& filename);
iu::ImageNpp_32f_C1* imread_cu32f_C1(const std::string& filename);

bool imsave(iu::ImageCpu_32f_C1* image, const std::string& filename);
bool imsave(iu::ImageNpp_32f_C1* image, const std::string& filename);

void imshow(iu::ImageCpu_32f_C1* image, const std::string& winname);
void imshow(iu::ImageCpu_32f_C3* image, const std::string& winname);
void imshow(iu::ImageCpu_32f_C4* image, const std::string& winname);
void imshow(iu::ImageNpp_32f_C1* image, const std::string& winname);
void imshow(iu::ImageNpp_32f_C4* image, const std::string& winname);

} // namespace iuprivate


#endif // IUIO_IMAGEIO_H
