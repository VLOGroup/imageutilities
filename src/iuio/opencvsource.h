#ifndef OPENCVSOURCE_H
#define OPENCVSOURCE_H

#include "videosource.h"
#include <opencv2/highgui/highgui.hpp>



/** A videosource derived from opencv videocapture */
class OpenCVSource : public VideoSource
{
  Q_OBJECT
public:
  /** initialize camera camId */
  OpenCVSource(unsigned int camId=0);
  virtual ~OpenCVSource();

  /** get image from camera */
  cv::Mat getImage();

  /** get image width */
  unsigned int getWidth() { return width_; }

  /** get image height */
  unsigned int getHeight() { return height_; }

  /** get current frame number */
  unsigned int getCurrentFrameNr() { return frameNr_; }

private:
  cv::VideoCapture *videocapture_;

  float fps_;
  cv::Mat imageBGR_;
  cv::Mat imageRGB_;

};

#endif // OPENCVSOURCE_H
