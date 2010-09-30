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


#ifndef IUPRIVATE_VIDEOCAPTURE_H
#define IUPRIVATE_VIDEOCAPTURE_H

#include <cv.h>
#include <iudefs.h>
#include "videocapturethread.h"

namespace iuprivate {

class VideoCapture
{
public:
  /** Default constructor. */
  VideoCapture();

  /** Constructor that opens a video file. */
  VideoCapture(std::string& filename);

  /** Constructor that opens a camera. */
  VideoCapture(int device);

  /** Default destructor. */
  ~VideoCapture();

  /** Converts and gets image data. */
  bool getImage(iu::ImageCpu_32f_C1* image);

  /** Converts and gets image data. */
  bool getImage(iu::ImageGpu_32f_C1* image);

  /** Query state for available images. */
  inline bool isNewImageAvailable() {return new_image_available_;}

  /** Returns the image size of the stream. */
  IuSize size();

private:
  VideoCaptureThread* cap_;
  IuSize size_;
  cv::Mat frame_;
  bool new_image_available_;
};

} // namespace iuprivate

#endif // IUPRIVATE_VIDEOCAPTURE_H
