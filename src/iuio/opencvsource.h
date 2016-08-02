#pragma once

#include "videosource.h"
#include <string>

namespace iu {

/** \defgroup VideoIO Video
  * \ingroup IO
  * \brief Read from cameras and video files
  * \{
  */

/**
 * @brief The OpenCVSource class uses OpenCV to read images from cameras or files
 */
class OpenCVSource : public VideoSource
{
public:

    /**
     * @brief OpenCVSource constructor. Initialize the camera \p camId
     * @param camId id of camera to initialize
     * @param gray if true, capture 8-bit grayscale, otherwise 24-bit RGB
     */
    OpenCVSource(unsigned int camId=0, bool gray=true);

    /**
     * @brief OpenCVSource constructor to read from a video file
     * @param filename video file name
     * @param gray if true, capture 8-bit grayscale, otherwise 24-bit RGB
     */
    OpenCVSource(const std::string& filename, bool gray=true);
    virtual ~OpenCVSource();

    /**
     * @brief get new image
     * @return cv::Mat
     */
    cv::Mat getImage();

    /**
     * @brief get image width
     * @return width
     */
    unsigned int getWidth() { return width_; }

    /**
     * @brief get image height
     * @return height
     */
    unsigned int getHeight() { return height_; }

    /**
     * @brief get frame index. Upon camera initilaization, the counter is set to 0
     * @return
     */
    unsigned int getCurrentFrameNr() { return frameNr_; }

private:

    cv::VideoCapture* videocapture_;
    int numFrames_;

    cv::Mat frame_;
    bool gray_;
};

/** \}  */ // end of videoIO



} // namespace iu



