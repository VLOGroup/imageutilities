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

#include <QMutex>
#include <QMutexLocker>
#include <iudefs.h>
#include <iucore/copy.h>
#include "videocapture.h"

namespace iuprivate {

//-----------------------------------------------------------------------------
VideoCapture::VideoCapture() :
	new_image_available_(false)
{
  cap_ = new VideoCaptureThread();
  cap_->registerExternalImage(&frame_, &new_image_available_, size_);
  cap_->start();
}

//-----------------------------------------------------------------------------
VideoCapture::VideoCapture(std::string& filename) :
    new_image_available_(false)
{
  cap_ = new VideoCaptureThread(filename);
  cap_->registerExternalImage(&frame_, &new_image_available_, size_);
  cap_->start();
}

//-----------------------------------------------------------------------------
VideoCapture::VideoCapture(int device) :
    new_image_available_(false)
{
  cap_ = new VideoCaptureThread(device);
  cap_->registerExternalImage(&frame_, &new_image_available_, size_);
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
  QMutexLocker locker(cap_->getMutex());

  printf("VideCapture::getImage: 0\n");
  if(!new_image_available_)
    return false;
  printf("VideCapture::getImage: 1\n");

  printf("! VideCapture::getImage: 2\n");
  cv::Mat frame_8u_C1;
  printf("! VideCapture::getImage: 3\n");
  cv::cvtColor(frame_, frame_8u_C1, CV_BGR2GRAY);
  printf("! VideCapture::getImage: 4\n");
  cv::Mat image_mat(image->height(), image->width(), CV_32FC1, image->data(), image->pitch());
  printf("! VideCapture::getImage: 5\n");
  frame_8u_C1.convertTo(image_mat, image_mat.type(), 1.0f/255.0f, 0);
  printf("! VideCapture::getImage: 6\n");

  new_image_available_ = false;
  return true;
}

//-----------------------------------------------------------------------------
bool VideoCapture::getImage(iu::ImageGpu_32f_C1* image)
{
  QMutexLocker locker(cap_->getMutex());

  printf("VideCapture::getImage: 1\n");
  if(!new_image_available_)
    return false;

  printf("VideCapture::getImage: 2\n");
  iu::ImageCpu_32f_C1 cpu_image(frame_.cols, frame_.rows);
  printf("VideCapture::getImage: 3\n");
  iuprivate::copy(&cpu_image, image);
  printf("VideCapture::getImage: 4\n");

  new_image_available_ = false;
  printf("VideCapture::getImage: 5\n");
  return true;
}

//-----------------------------------------------------------------------------
IuSize VideoCapture::size()
{
  return size_;
}

} // namespace iuprivate
