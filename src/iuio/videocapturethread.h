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
 * Description : Definition of a thread to capture videos from either files or cameras with OpenCVs VideoCapture.
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUPRIVATE_VIDEOCAPTURETHREAD_H
#define IUPRIVATE_VIDEOCAPTURETHREAD_H

#include <QThread>
#include <QMutex>
#include <cv.h>
#include <highgui.h>
#include <iudefs.h>

namespace iuprivate {

class VideoCaptureThread : public QThread
{
public:
  /** Default constructor. */
  VideoCaptureThread();

  /** Constructor that opens a video file. */
  VideoCaptureThread(std::string& filename);

  /** Constructor that opens a camera. */
  VideoCaptureThread(int device);

  /** Default destructor. */
  ~VideoCaptureThread();

  /** The starting point of the thread. Here all the magic happens. */
  void run();

  inline QMutex* getMutex() {return &mutex_;}

  /** Registers an external image where the data is copied to when available. */
  void registerExternalImage(cv::Mat*, bool* new_image_available,
                             IuSize& cap_size);

private:
  QMutex mutex_; /**< Lock to simplify read and write access of images. */
  bool stop_thread_; /**< Flag of the threads run state. */
  unsigned long sleep_time_usecs_; /**< The thread sleeps for \a sleep_time_usecs_ microseconds after grabbing an image. */

  cv::VideoCapture* cv_cap_; /**< OpenCVs video capture. This is used to read all the data. */
  cv::Mat frame_; /**< Current frame. Used to read internally. */

  cv::Mat* ext_frame_; /**< External frame that is used for external sync. */
  bool* ext_new_image_available_; /**< External flag if there is a new image available. Gets set when the external image is updated. */
};

} // namespace iuprivate

#endif // IUPRIVATE_VIDEOCAPTURETHREAD_H
