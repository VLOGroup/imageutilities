#include "1394source.h"
#include "1394camera.h"

FW1394Source::FW1394Source()
{
  try {
    // create FireWire Camera
    cam_1394_ = new Camera(NULL);
//    cam_1394_->setBayerDecoding(true);
    cam_1394_->setFramerate(DC1394_FRAMERATE_30);
    cam_1394_->setMode(DC1394_VIDEO_MODE_1024x768_MONO8);
    //camera_1394_->setMode(DC1394_VIDEO_MODE_FORMAT7_1);
    cam_1394_->initialize();
    cam_1394_->start();

    fps_ = 0;
    switch (cam_1394_->getFramerate())
    {
    case DC1394_FRAMERATE_1_875:
      fps_ = 1.875;
      break;
    case DC1394_FRAMERATE_3_75:
      fps_ = 3.75;
      break;
    case DC1394_FRAMERATE_7_5:
      fps_ = 7.5;
      break;
    case DC1394_FRAMERATE_15:
      fps_ = 15;
      break;
    case DC1394_FRAMERATE_30:
      fps_ = 30;
      break;
    case DC1394_FRAMERATE_60:
      fps_ = 60;
      break;
    case DC1394_FRAMERATE_120:
      fps_ = 120;
      break;
    case DC1394_FRAMERATE_240:
      fps_ = 240;
      break;
    default:
      fps_ = -1;
    }
    cout << "[1394Source] Init camera, running at  " << fps_ << "fps" << endl;
  }
  catch ( std::exception& e ) {
    std::cerr << "Std Exception: " << e.what() << std::endl;;
  }
  catch (...) {
    std::cerr << "Unknown Exception!!" << std::endl;
  }

  width_ = cam_1394_->getWidth();
  height_ = cam_1394_->getHeight();
  camImageRGB_.create(height_, width_, CV_8UC3);
  frameNr_ = 0;
}


FW1394Source::~FW1394Source()
{
  cout << "destr 1394source: Waiting for camera thread to finish...";
  cam_1394_->quit();
  cam_1394_->wait();
  delete cam_1394_;
  cam_1394_ = NULL;
  cout << "done" << endl;
}


cv::Mat FW1394Source::getImage()
{
    cam_1394_->getImageRGB8((RGB8*)camImageRGB_.data);
    return camImageRGB_;
}

