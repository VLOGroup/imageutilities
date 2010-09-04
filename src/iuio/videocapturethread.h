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
#include <QReadWriteLock>
#include <cv.h>
#include <highgui.h>
#include <iudefs.h>

namespace iuprivate {

class VideoCaptureThread : public QThread
{
public:
  /** Constructor. */
  VideoCaptureThread();

  /** The starting point of the thread. Here all the magic happens. */
  void run();

private:
  QReadWriteLock lock_; /**< Lock to simplify read and write access of images. */
  bool thread_runs_; /**< Flag of the threads run state. */

  cv::VideoCapture cv_cap_; /**< OpenCVs video capture. This is used to read all the data. */
  cv::Mat frame_; /**< Current frame. Used to read internally. */

  iu::ImageCpu_32f_C1* cur_frame_; /**< Current frame used to distribute outside the thrad. */
};

} // namespace iuprivate

#endif // IUPRIVATE_VIDEOCAPTURETHREAD_H
