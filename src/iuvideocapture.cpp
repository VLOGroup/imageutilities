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
 * Class       : VideoCapture
 * Language    : C++
 * Description : Definition of the public interface of retrieving images through the VideoCapture.
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#include <iuio/videocapture.h>
#include "iuvideocapture.h"

namespace iu {

//-----------------------------------------------------------------------------
VideoCapture::VideoCapture()
{
  video_capture_ = new iuprivate::VideoCapture();
}

//-----------------------------------------------------------------------------
VideoCapture::VideoCapture(std::string& filename)
{
  video_capture_ = new iuprivate::VideoCapture(filename);
}


//-----------------------------------------------------------------------------
VideoCapture::VideoCapture(int device)
{
  video_capture_ = new iuprivate::VideoCapture(device);
}

//-----------------------------------------------------------------------------
VideoCapture::~VideoCapture()
{
  delete(video_capture_);
}

//-----------------------------------------------------------------------------
bool VideoCapture::getImage(iu::ImageCpu_32f_C1* image)
{
  return video_capture_->getImage(image);
}

//-----------------------------------------------------------------------------
bool VideoCapture::getImage(iu::ImageGpu_32f_C1* image)
{
  return video_capture_->getImage(image);
}

//-----------------------------------------------------------------------------
bool VideoCapture::isNewImageAvailable()
{
  return video_capture_->isNewImageAvailable();
}

//-----------------------------------------------------------------------------
IuSize VideoCapture::size()
{
  return video_capture_->size();
}

} // namespace iu
