#ifndef VIDEOSOURCE_H
#define VIDEOSOURCE_H

#include <opencv2/opencv.hpp>


namespace iu {

/** \defgroup VideoIO Video
  * \ingroup IO
  * \brief Read from cameras and video files
  * \{
  */

/**
 * @brief The VideoSource class is the abstract base class for video input
 */
class VideoSource
{
public:

    /**
   * @brief get a new image from the source
   * @return cv::Mat
   */
  virtual cv::Mat getImage() = 0;

    /**
   * @brief get image width
   * @return width
   */
  virtual unsigned int getWidth() = 0;

    /**
   * @brief get image height
   * @return height
   */
  virtual unsigned int getHeight() = 0;

    /**
   * @brief get current frame number.
   * @return frame index
   */
  virtual unsigned int getCurrentFrameNr() = 0;

protected:
  unsigned int width_;
  unsigned int height_;
  unsigned int frameNr_;
};

/** \}  */ // end of videoIO


} // namespace iu

#endif // VIDEOSOURCE_H
