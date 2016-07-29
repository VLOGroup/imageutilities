#include "opencvsource.h"


namespace iu {



OpenCVSource::OpenCVSource(unsigned int camId, bool gray)
{
    videocapture_ = new cv::VideoCapture;

    if (!videocapture_->open(camId))
    {
        width_ = 0;
        height_ = 0;
        videocapture_ = NULL;
        std::cerr << "OpeCVSource: Could not open device" << std::endl;
    }
    frameNr_ = 0;
    numFrames_ = -1;

    width_ = videocapture_->get(CV_CAP_PROP_FRAME_WIDTH);
    height_ = videocapture_->get(CV_CAP_PROP_FRAME_HEIGHT);

    gray_ = gray;
    if (gray)
        frame_ = cv::Mat(height_, width_, CV_8UC1);
    else
        frame_ = cv::Mat(height_, width_, CV_8UC4);
}


OpenCVSource::OpenCVSource(const std::string &filename, bool gray)
{
    videocapture_ = new cv::VideoCapture;

    if (!videocapture_->open(filename))
    {
        width_ = 0;
        height_ = 0;
        videocapture_ = NULL;
        numFrames_ = -1;
        std::cerr << "OpeCVSource: Could not open file " << filename <<  std::endl;
    }
    frameNr_ = 0;

    width_ = videocapture_->get(CV_CAP_PROP_FRAME_WIDTH);
    height_ = videocapture_->get(CV_CAP_PROP_FRAME_HEIGHT);
    numFrames_ = videocapture_->get(CV_CAP_PROP_FRAME_COUNT);

    gray_ = gray;
    if (gray)
        frame_ = cv::Mat(height_, width_, CV_8UC1);
    else
        frame_ = cv::Mat(height_, width_, CV_8UC4);
}


OpenCVSource::~OpenCVSource()
{
    videocapture_->release();
    delete videocapture_;
}

cv::Mat OpenCVSource::getImage()
{
    cv::Mat capture_img;

    if (!videocapture_)
    {
        std::cerr << "OpenCVSource: videosource not initialized" << std::endl;
        return frame_;
    }


    if (!videocapture_->read(capture_img))
    {
        std::cerr << "OpenCVSource: Could not read frame" << std::endl;
        return frame_;
    }
    frameNr_++;

    if (gray_)
        cvtColor(capture_img, frame_, CV_RGB2GRAY);
    else
        cvtColor(capture_img, frame_, CV_BGR2RGBA);

    return frame_;
}


} // namespace iu
