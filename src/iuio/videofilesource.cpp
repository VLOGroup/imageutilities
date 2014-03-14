#include "videofilesource.h"

#include <cstdio>


VideofileSource::VideofileSource(std::string filename)
{
  videocapture_ = new cv::VideoCapture();

  if (!videocapture_->open(filename))
  {
    width_ = 0;
    height_ = 0;
    std::cerr << "VideofileSource: Error opening file " << filename << std::endl;
  }

  width_ = videocapture_->get(CV_CAP_PROP_FRAME_WIDTH);
  height_ = videocapture_->get(CV_CAP_PROP_FRAME_HEIGHT);
  totalFrames_ = videocapture_->get(CV_CAP_PROP_FRAME_COUNT);
  fps_ = videocapture_->get(CV_CAP_PROP_FPS);
  frameNr_ = 0;
}

VideofileSource::~VideofileSource()
{
  videocapture_->release();
  delete videocapture_;
}

cv::Mat VideofileSource::getImage()
{
  // read always returns a 8uC3 bgr image
  if (!videocapture_->read(imageBGR_))
  {
    std::cerr << "VideofileSource::getImage(): Error reading image, reached end of file?" << std::endl;
    return cv::Mat::zeros(width_, height_, CV_8UC1);
  }
//  printf("mat type %d 8uC1 %d 8uC3 %d\n", imageBGR_.type(), CV_8UC1, CV_8UC3);

  frameNr_++;

  cv::cvtColor(imageBGR_, imageGray_, CV_BGR2GRAY);
//  cv::cvtColor(imageBGR_, imageRGB_, CV_BGR2RGB);
  return imageGray_;
}


