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
    thread_runs_(false),
    sleep_time_usecs_(30),
    ext_frame_(0),
    ext_new_image_available_(0)
{
}

//-----------------------------------------------------------------------------
VideoCaptureThread::VideoCaptureThread(std::string &filename) :
    thread_runs_(false),
    sleep_time_usecs_(30),
    cv_cap_(filename),
    ext_frame_(0),
    ext_new_image_available_(0)
{
}

//-----------------------------------------------------------------------------
VideoCaptureThread::VideoCaptureThread(int device) :
    thread_runs_(false),
    sleep_time_usecs_(30),
    cv_cap_(device),
    ext_frame_(0),
    ext_new_image_available_(0)
{
}

//-----------------------------------------------------------------------------
VideoCaptureThread::~VideoCaptureThread()
{
}

//-----------------------------------------------------------------------------
void VideoCaptureThread::run()
{
  thread_runs_ = true;

  while (thread_runs_)
  {
    // first check if capture device is (still) ok
    if (!cv_cap_.isOpened())
    {
      printf("VideoCaptureThread: Capture device not ready\n");
      thread_runs_ = false;
      break;
    }

    if(cv_cap_.grab())
    {
      cv_cap_ >> frame_;

      // copy to 'external' data
      if(ext_frame_ != 0)
      {
        lock_.lockForWrite();
        frame_.copyTo(*ext_frame_);
        *ext_new_image_available_ = true;
        lock_.unlock();
      }
    }
    else
      thread_runs_ = false;

    this->usleep(sleep_time_usecs_);
  }
}

//-----------------------------------------------------------------------------
void VideoCaptureThread::registerExternalImage(cv::Mat* image, bool* new_image_available)
{
  ext_frame_ = image;
  ext_new_image_available_ = new_image_available;
}


} // namespace iuprivate
