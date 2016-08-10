#pragma once

#include "FlyCapture2.h"

namespace iuprivate {

class PGRCameraData
{
public:
  // invalid data struct
  PGRCameraData()
    {
      bus_mgr_ = new FlyCapture2::BusManager();
      cam_ = new FlyCapture2::Camera();
      raw_image_ = new FlyCapture2::Image();
    }


  ~PGRCameraData()
    {
      delete(bus_mgr_);
      delete(cam_);
      delete(raw_image_);
    }

  FlyCapture2::BusManager* bus_mgr_;
  FlyCapture2::Camera* cam_;
  FlyCapture2::Image* raw_image_;
};

} // namespace iu


