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

#ifndef IU_VIDEOCAPTURE_H
#define IU_VIDEOCAPTURE_H

#include <iudefs.h>

// forward declarations
namespace iuprivate {
class VideoCapture;
}

namespace iu {

class IUIO_DLLAPI VideoCapture
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

  // grab the next frame
  virtual bool grab();

  /** Retrieves cpu image (8-bit; 1-channel). */
  virtual void retrieve(iu::ImageCpu_8u_C1* image);

  /** Retrieves cpu image (32-bit; 1-channel). */
  virtual void retrieve(iu::ImageCpu_32f_C1* image);

  /** Retrieves gpu image (32-bit; 1-channel). */
  virtual void retrieve(iu::ImageGpu_32f_C1* image);

  /** Returns the size of the available images. */
  IuSize size();

  /** Returns the framerate of the videostream */
  int getFPS();

  /** Set the framerate of the videostream */
  int setFPS(int fps);

  /** Returns the frame width. */
  int getFrameWidth();

  /** Sets the frame width. */
  int setFrameWidth(int width);

  /** Returns the frame height. */
  int getFrameHeight();

  /** Sets the frame height. */
  int setFrameHeight(int height);

  /** Returns the total number of frames in the videostream. Useful for video files,
    * don't know what this returns on a live camerastream... */
  int totalFrameCount();

  /** Returns the frame index of the next frame (0-based) */
  int frameIdx();

  /** Returns the specified VideoCapture property (see OpenCV manual) */
  double get(int propId);

  /** Sets a property in the VideoCapture (see OpenCV manual) */
  bool set(int propId, double value);


private:
  iuprivate::VideoCapture* vidcap_;

};

} // namespace iu

#endif // IU_VIDEOCAPTURE_H
