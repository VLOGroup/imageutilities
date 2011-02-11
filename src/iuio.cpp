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
#include "iuio/imageio.h"
#include "iuio/imageiopgm.h"

namespace iu {

/* ***************************************************************************
     read 2d image
 * ***************************************************************************/

iu::ImageCpu_8u_C1* imread_8u_C1(const std::string& filename)
{ return iuprivate::imread_8u_C1(filename); }

iu::ImageCpu_8u_C3* imread_8u_C3(const std::string& filename)
{ return iuprivate::imread_8u_C3(filename); }

iu::ImageCpu_8u_C4* imread_8u_C4(const std::string& filename)
{ return iuprivate::imread_8u_C4(filename); }

iu::ImageCpu_32f_C1* imread_32f_C1(const std::string& filename)
{ return iuprivate::imread_32f_C1(filename); }

iu::ImageCpu_32f_C3* imread_32f_C3(const std::string& filename)
{ return iuprivate::imread_32f_C3(filename); }

iu::ImageCpu_32f_C4* imread_32f_C4(const std::string& filename)
{ return iuprivate::imread_32f_C4(filename); }

iu::ImageGpu_8u_C1* imread_cu8u_C1(const std::string& filename)
{ return iuprivate::imread_cu8u_C1(filename); }

iu::ImageGpu_8u_C4* imread_cu8u_C4(const std::string& filename)
{ return iuprivate::imread_cu8u_C4(filename); }

iu::ImageGpu_32f_C1* imread_cu32f_C1(const std::string& filename)
{ return iuprivate::imread_cu32f_C1(filename); }

iu::ImageGpu_32f_C4* imread_cu32f_C4(const std::string& filename)
{ return iuprivate::imread_cu32f_C4(filename); }

/* ***************************************************************************
     write 2d image
 * ***************************************************************************/

bool imsave(iu::ImageCpu_8u_C1* image, const std::string& filename, const bool& normalize)
{ return iuprivate::imsave(image, filename, normalize); }

bool imsave(iu::ImageCpu_8u_C3* image, const std::string& filename, const bool& normalize)
{ return iuprivate::imsave(image, filename, normalize); }

bool imsave(iu::ImageCpu_8u_C4* image, const std::string& filename, const bool& normalize)
{ return iuprivate::imsave(image, filename, normalize); }

bool imsave(iu::ImageCpu_32f_C1* image, const std::string& filename, const bool& normalize)
{ return iuprivate::imsave(image, filename, normalize); }

bool imsave(iu::ImageCpu_32f_C3* image, const std::string& filename, const bool& normalize)
{ return iuprivate::imsave(image, filename, normalize); }

bool imsave(iu::ImageCpu_32f_C4* image, const std::string& filename, const bool& normalize)
{ return iuprivate::imsave(image, filename, normalize); }

bool imsave(iu::ImageGpu_8u_C1* image, const std::string& filename, const bool& normalize)
{ return iuprivate::imsave(image, filename, normalize); }

bool imsave(iu::ImageGpu_8u_C4* image, const std::string& filename, const bool& normalize)
{ return iuprivate::imsave(image, filename, normalize); }

bool imsave(iu::ImageGpu_32f_C1* image, const std::string& filename, const bool& normalize)
{ return iuprivate::imsave(image, filename, normalize); }

bool imsave(iu::ImageGpu_32f_C4* image, const std::string& filename, const bool& normalize)
{ return iuprivate::imsave(image, filename, normalize); }

/* ***************************************************************************
     show 2d image
 * ***************************************************************************/

void imshow(iu::ImageCpu_8u_C1* image, const std::string& winname, const bool& normalize)
{ iuprivate::imshow(image, winname, normalize); }

void imshow(iu::ImageCpu_8u_C3* image, const std::string& winname, const bool& normalize)
{ iuprivate::imshow(image, winname, normalize); }

void imshow(iu::ImageCpu_8u_C4* image, const std::string& winname, const bool& normalize)
{ iuprivate::imshow(image, winname, normalize); }

void imshow(iu::ImageCpu_32f_C1* image, const std::string& winname, const bool& normalize)
{ iuprivate::imshow(image, winname, normalize); }

void imshow(iu::ImageCpu_32f_C3* image, const std::string& winname, const bool& normalize)
{ iuprivate::imshow(image, winname, normalize); }

void imshow(iu::ImageCpu_32f_C4* image, const std::string& winname, const bool& normalize)
{ iuprivate::imshow(image, winname, normalize); }

void imshow(iu::ImageGpu_8u_C1* image, const std::string& winname, const bool& normalize)
{ iuprivate::imshow(image, winname, normalize); }

void imshow(iu::ImageGpu_8u_C4* image, const std::string& winname, const bool& normalize)
{ iuprivate::imshow(image, winname, normalize); }

void imshow(iu::ImageGpu_32f_C1* image, const std::string& winname, const bool& normalize)
{ iuprivate::imshow(image, winname, normalize); }

void imshow(iu::ImageGpu_32f_C4* image, const std::string& winname, const bool& normalize)
{ iuprivate::imshow(image, winname, normalize); }


} // namespace iu
