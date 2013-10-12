#include "opencvsource.h"

OpenCVSource::OpenCVSource(unsigned int camId)
{
    videocapture_ = new cv::VideoCapture(0);

    if (!videocapture_->isOpened())
    {
      width_ = 0;
      height_ = 0;
      std::cerr << "OpenCVSource: Error opening camera " << camId << std::endl;
    }

    videocapture_->release();

    videocapture_->set(CV_CAP_PROP_FRAME_WIDTH, 1024);
    videocapture_->set(CV_CAP_PROP_FRAME_HEIGHT, 768);
    videocapture_->set(CV_CAP_PROP_FPS, 30);

    videocapture_->open(0);

    width_ = videocapture_->get(CV_CAP_PROP_FRAME_WIDTH);
    height_ = videocapture_->get(CV_CAP_PROP_FRAME_HEIGHT);
    fps_ = videocapture_->get(CV_CAP_PROP_FPS);
    frameNr_ = 0;
}

OpenCVSource::~OpenCVSource()
{
    videocapture_->release();
    delete videocapture_;
}


cv::Mat OpenCVSource::getImage()
{
    if (!videocapture_->read(imageBGR_))
    {
      std::cerr << "OpenCVSource::getImage(): Error reading from camera" << std::endl;
      return cv::Mat::zeros(width_, height_, CV_8UC3);
    }

    frameNr_++;

    cv::cvtColor(imageBGR_, imageRGB_, CV_BGR2RGB);
    return imageRGB_;
}
