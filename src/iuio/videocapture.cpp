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
 * Description : Definition of the private interface of retrieving images throuth the VideoCaptureThread.
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#include "videocapture.h"
#include "iucore.h"

namespace iuprivate {

//-----------------------------------------------------------------------------
VideoCapture::VideoCapture::VideoCapture()
{
  cap_ = new VideoCaptureThread();
  cap_->registerExternalImage(&frame_, &new_image_available_);
  cap_->start();
}

//-----------------------------------------------------------------------------
VideoCapture::VideoCapture(std::string& filename)
{
  cap_ = new VideoCaptureThread(filename);
  cap_->registerExternalImage(&frame_, &new_image_available_);
  cap_->start();
}

//-----------------------------------------------------------------------------
VideoCapture::VideoCapture(int device)
{
  cap_ = new VideoCaptureThread(device);
  cap_->registerExternalImage(&frame_, &new_image_available_);
  cap_->start();
}

//-----------------------------------------------------------------------------
VideoCapture::~VideoCapture()
{
  cap_->quit();
  delete(cap_);
}

//-----------------------------------------------------------------------------
bool VideoCapture::getImage(iu::ImageCpu_32f_C1* image)
{
  if(!new_image_available_)
    return false;

  cap_->getLock().lockForRead();
  cv::Mat frame_8u_C1;
  printf("1\n");
  cv::cvtColor(frame_, frame_8u_C1, CV_BGR2GRAY);
  printf("2\n");
  cv::Mat image_mat(image->height(), image->width(), CV_32FC1, image->data(), image->pitch());
  printf("3\n");
  frame_8u_C1.convertTo(image_mat, image_mat.type(), 1.0f/255.0f, 0);
  printf("4\n");
  cap_->getLock().unlock();

  new_image_available_ = false;
  return true;
}

//-----------------------------------------------------------------------------
bool VideoCapture::getImage(iu::ImageNpp_32f_C1* image)
{
  if(!new_image_available_)
    return false;

  cap_->getLock().lockForRead();
  iu::ImageCpu_32f_C1 cpu_image(frame_.cols, frame_.rows);
  iu::copy(&cpu_image, image);
  cap_->getLock().unlock();

  new_image_available_ = false;
  return true;
}

//-----------------------------------------------------------------------------
IuSize VideoCapture::size()
{
  cap_->getLock().lockForRead();
  IuSize sz(frame_.cols, frame_.rows);
  cap_->getLock().unlock();
  return sz;
}

} // namespace iuprivate
