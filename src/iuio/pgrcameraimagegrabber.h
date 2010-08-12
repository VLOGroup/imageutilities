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
 * Project     : vmgpu
 * Module      : Tools
 * Language    : C++
 * Description : Definition/Implementation of a thread responsible for grabbing images from the camera
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */


#ifndef PGRCAMERAIMAGEGRABBER_H
#define PGRCAMERAIMAGEGRABBER_H

#include <QtCore>
#include "FlyCapture2.h"
#include "pgrcameradata.h"

class PGRCameraImageGrabber : public QThread
{
public:
  /** PGRCameraImageGrabber constructor. */
  PGRCameraImageGrabber(PGRCameraData* data) :
    QThread(),
    data_(data),
    thread_running_(false)
    {

    }

  /** run function -- this is what the thread actually does. */
  void run()
    {
      thread_running_ = true;

      while(thread_running_)
      {
        data_->image_mutex_.lock();
        FlyCapture2::Error error = data_->cam_->RetrieveBuffer(data_->raw_image_);
        data_->no_image_wait_cond_.wakeAll();
        data_->image_mutex_.unlock();
        this->usleep(5);
      }
    }

public slots:
  void quit()
    {
      thread_running_ = false;
      QThread::quit();
    }

private:
  PGRCameraData* data_;
  bool thread_running_;
};

#endif // PGRCAMERAIMAGEGRABBER_H

