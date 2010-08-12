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
 * Description : Definition of a Camera using the Fyrefly SDK from Point Grey
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 * comment: Many code parts are taken from the examples of the FlyCapture SDK examples
 */


#ifndef PGRCAMERADATA_H
#define PGRCAMERADATA_H

#include <QMutex>
#include <QWaitCondition>
#include "FlyCapture2.h"

class PGRCameraData
{
public:
  // invalid data struct
  PGRCameraData()
    {
      bus_mgr_ = new FlyCapture2::BusManager();
      cam_ = new FlyCapture2::Camera();
      raw_image_ = new FlyCapture2::Image();
      processed_image_ = new FlyCapture2::Image();
    }


  ~PGRCameraData()
    {
      delete(bus_mgr_);
      delete(cam_);
      delete(raw_image_);
      delete(processed_image_);
    }

  FlyCapture2::BusManager* bus_mgr_;
  FlyCapture2::Camera* cam_;
  FlyCapture2::Image* raw_image_;
  FlyCapture2::Image* processed_image_;

  QMutex image_mutex_;
  QWaitCondition no_image_wait_cond_; // Wait condition until an image is available for the consumer.
};

#endif // PGRCAMERA_H
