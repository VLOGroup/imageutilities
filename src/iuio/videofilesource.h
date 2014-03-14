#ifndef VIDEOFILESOURCE_H
#define VIDEOFILESOURCE_H

#include <string>
#include "videosource.h"
#include <opencv2/highgui/highgui.hpp>

/** A videosource to read images from a video file */
class VideofileSource : public VideoSource
{
  Q_OBJECT
public:
  /** open file filename */
  VideofileSource(std::string filename);
  virtual ~VideofileSource();

  /** get image from camera */
  cv::Mat getImage();


  /** get image width */
  unsigned int getWidth() { return width_; }

  /** get image height */
  unsigned int getHeight() { return height_; }

  /** get image height */
  unsigned int getCurrentFrameNr() { return frameNr_; }

  /** get frames per second */
  float getFPS() { return fps_; }


private:
  cv::VideoCapture *videocapture_;

  unsigned int totalFrames_;
  float fps_;
  cv::Mat imageBGR_;
  cv::Mat imageGray_;
  cv::Mat imageRGB_;
};

#endif // VIDEOFILESOURCE_H
