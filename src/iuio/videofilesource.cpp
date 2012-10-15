#include "videofilesource.h"



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
  if (!videocapture_->read(image_))
  {
    std::cerr << "VideofileSource::getImage(): Error reading image, reached end of file?" << std::endl;
    return cv::Mat::zeros(width_, height_, CV_8UC3);
  }

  frameNr_++;

  return image_;
}


