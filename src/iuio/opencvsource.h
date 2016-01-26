#ifndef OPENCV_SOURCE_H
#define OPENCV_SOURCE_H

#include "videosource.h"
#include <string>

namespace iu {

class OpenCVSource : public VideoSource
{
public:

    OpenCVSource(unsigned int camId=0, bool gray=true);
    OpenCVSource(const std::string& filename, bool gray=true);
    virtual ~OpenCVSource();

    cv::Mat getImage();

    /** get image width */
    unsigned int getWidth() { return width_; }

    /** get image height */
    unsigned int getHeight() { return height_; }

    /** get current frame number */
    unsigned int getCurrentFrameNr() { return frameNr_; }

private:

    cv::VideoCapture* videocapture_;
    int numFrames_;

    cv::Mat frame_;
    bool gray_;
};


} // namespace iu


#endif
