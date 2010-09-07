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
 * Class       : VideoCaptureThread
 * Language    : C++
 * Description : Implementation of a thread to capture videos from either files or cameras with OpenCVs VideoCapture.
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#include "videocapturethread.h"

namespace iuprivate {

//-----------------------------------------------------------------------------
VideoCaptureThread::VideoCaptureThread() :
    stop_thread_(false),
    sleep_time_usecs_(30),
    cv_cap_(0),
    ext_frame_(0),
    ext_new_image_available_(0)
{
}

//-----------------------------------------------------------------------------
VideoCaptureThread::VideoCaptureThread(std::string &filename) :
    stop_thread_(false),
    sleep_time_usecs_(30),
    cv_cap_(0),
    ext_frame_(0),
    ext_new_image_available_(0)
{
  cv_cap_ = new cv::VideoCapture(filename);
  // grab the first frame
  if (cv_cap_->isOpened())
    (*cv_cap_) >> frame_;
}

//-----------------------------------------------------------------------------
VideoCaptureThread::VideoCaptureThread(int device) :
    stop_thread_(false),
    sleep_time_usecs_(30),
    cv_cap_(0),
    ext_frame_(0),
    ext_new_image_available_(0)
{
  cv_cap_ = new cv::VideoCapture(device);
  // grab the first frame
  if (cv_cap_->isOpened())
    (*cv_cap_) >> frame_;
}

//-----------------------------------------------------------------------------
VideoCaptureThread::~VideoCaptureThread()
{
  printf("delete VideoCaptureThread");
  stop_thread_ = true;
  wait();
  cv_cap_->release();
  delete(cv_cap_);
}

//-----------------------------------------------------------------------------
void VideoCaptureThread::run()
{
  forever
  {
    if(stop_thread_)
      return;

    // first check if capture device is (still) ok
    if (!cv_cap_->isOpened())
    {
      printf("VideoCaptureThread: Capture device not ready\n");
      stop_thread_ = true;
      return;
    }

    printf("thread: get next frame\n");
    (*cv_cap_) >> frame_;

    // copy to 'external' data
    printf("thread: cp frame to external data\n");
    if(ext_frame_ != 0)
    {
      QMutexLocker lock(&mutex_);
      printf("! thread: cp operation\n");
      frame_.copyTo(*ext_frame_);
      printf("! thread: cp set flag");
      *ext_new_image_available_ = true;
      printf("! thread: cp done\n");
    }

    printf("thread: sleep\n");
    this->usleep(sleep_time_usecs_);
  }
}

//-----------------------------------------------------------------------------
void VideoCaptureThread::registerExternalImage(cv::Mat* image, bool* new_image_available,
                                               IuSize& cap_size)
{
  QMutexLocker lock(&mutex_);
  ext_frame_ = image;
  ext_new_image_available_ = new_image_available;
  cap_size = IuSize(frame_.cols, frame_.rows);
}


} // namespace iuprivate

