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


#ifndef IUVIDEOCAPTURE_H
#define IUVIDEOCAPTURE_H

namespace iuprivate {
  class VideoCapture;
}

namespace iu {

#include <iudefs.h>

class IU_DLLAPI VideoCapture
{
public:
  /** Default constructor. */
  VideoCapture();

  /** Constructor that opens a video file. */
  VideoCapture(std::string& filename);

  /** Constructor that opens a camera. */
  VideoCapture(int device);

  /** Default Destructor. */
  ~VideoCapture();

  /** Converts and gets image data. */
  bool getImage(iu::ImageCpu_32f_C1* image);

  /** Converts and gets image data. */
  bool getImage(iu::ImageGpu_32f_C1* image);

  /** Query state for available images. */
  bool isNewImageAvailable();

  /** Returns the image size of the stream. */
  IuSize size();

private:
  iuprivate::VideoCapture* video_capture_;

};

} // namespace iu

#endif // IUVIDEOCAPTURE_H
