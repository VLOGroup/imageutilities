#ifndef FW1394SOURCE_H
#define FW1394SOURCE_H

#include "videosource.h"


class Camera;

/** A videosource to read from firewire cameras */
class FW1394Source : public VideoSource
{
  Q_OBJECT
public:
  /** initialize camera camId */
  FW1394Source();
  virtual ~FW1394Source();

  /** get image from camera */
  cv::Mat getImage();

  /** get image width */
  unsigned int getWidth() { return width_; }

  /** get image height */
  unsigned int getHeight() { return height_; }

  /** get current frame number */
  unsigned int getCurrentFrameNr() { return frameNr_; }

private:
  Camera *cam_1394_;
  float fps_;
  cv::Mat camImageRGB_;
};

#endif // FW1394SOURCE_H
