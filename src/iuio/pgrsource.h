#ifndef PGRSOURCE_H
#define PGRSOURCE_H

#include "videosource.h"

class PGRCameraData;
namespace FlyCapture2 {
  class Error;
}


/** A videosource to read from Pointgrey Firewire cameras */
class PGRSource : public VideoSource
{
  Q_OBJECT
public:
  /** initialize camera camId */
  PGRSource(unsigned int camId=0);
  virtual ~PGRSource();

  /** get image from camera */
  cv::Mat getImage();

  /** get image width */
  unsigned int getWidth() { return width_; }

  /** get image height */
  unsigned int getHeight() { return height_; }

  /** get current frame number */
  unsigned int getCurrentFrameNr() { return frameNr_; }


private:
  bool init(unsigned int camId);
  bool connectToCamera(unsigned int camId);
  bool startCapture();
  void grab();
  void printError(FlyCapture2::Error* error);

  PGRCameraData* data_;
};

#endif // PGRSOURCE_H
