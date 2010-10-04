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
 * Description : Definition of VideoCapture that uses OpenCV to read either from a video file or a camera.
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUPRIVATE_VIDEOCAPTURE_H
#define IUPRIVATE_VIDEOCAPTURE_H

#include <cv.h>
#include <highgui.h>
#include <iudefs.h>

//
//  W A R N I N G
//  -------------
//
// This file is not part of the IU API.  It exists purely as an
// implementation detail.  This header file may change from version to
// version without notice, or even be removed.
//


namespace iuprivate {

class VideoCapture : public cv::VideoCapture
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

  /** Retrieves cpu image (8-bit; 1-channel). */
  virtual bool retrieve(iu::ImageCpu_8u_C1* image);

  /** Retrieves cpu image (32-bit; 1-channel). */
  virtual bool retrieve(iu::ImageCpu_32_C1* image);

  /** Retrieves gpu image (32-bit; 1-channel). */
  virtual bool retrieve(iu::ImageGpu_32_C1* image);

  IuSize getSize();

protected:
  cv::Mat frame_; /**< Current frame. Used to read internally. */

};

} // namespace iuprivate

#endif // IUPRIVATE_VIDEOCAPTURE_H
