#pragma once

#include "videosource.h"


namespace iuprivate {
    class PGRCameraData;
}
namespace FlyCapture2 {
  class Error;
}


namespace iu {

/** \defgroup VideoIO Video
  * \ingroup IO
  * \brief Read from cameras and video files
  * \{
  */

/**
 * @brief The PGRSource class reads from PointGrey Firewire cameras
 */
class PGRSource : public VideoSource
{
public:
   /**
   * @brief PGRSource constructor. Initialize the camera \p camId
   * @param camId id of camera to initialize
   * @param gray if true, capture 8-bit grayscale, otherwise 24-bit RGB
   */
  PGRSource(unsigned int camId=0, bool gray=true, bool use_format7=false);
  virtual ~PGRSource();

  /**
   * @brief grab a new image from the camera. If a new image is not available, block until
   * there is one. If getImage is called in a loop and the camera delivers e.g. 30fps,
   * you will get at most 30 loop iterations per second.
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
  bool init(unsigned int camId);
  bool connectToCamera(unsigned int camId);
  bool startCapture();
  void grab();
  void printError(FlyCapture2::Error* error);

  iuprivate::PGRCameraData* data_;
  bool gray_;
  bool use_format7_;
};

/** \}  */ // end of videoIO


}  // namespace


