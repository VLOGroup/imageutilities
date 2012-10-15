#include "pgrsource.h"
#include <FlyCapture2.h>
#include "pgrcameradata.h"

PGRSource::PGRSource(unsigned int camId)
{
  data_ = new PGRCameraData();

  // initializes the camera, set width and height member variables
  if (!this->init(camId))
  {
    width_ = 0;
    height_ = 0;
    std::cerr << "PGRSource: Error initializing camera" << std::endl;
  }
  frameNr_ = 0;
}

PGRSource::~PGRSource()
{
  if(data_->cam_->IsConnected())
  {
    FlyCapture2::Error error = data_->cam_->StopCapture();
    if (error != FlyCapture2::PGRERROR_OK)
      this->printError(&error);

    error = data_->cam_->Disconnect();
    if (error != FlyCapture2::PGRERROR_OK)
      this->printError(&error);
  }

  delete data_;
}



bool PGRSource::init(unsigned int camId)
{
  unsigned int numCams;
  FlyCapture2::Error error = data_->bus_mgr_->GetNumOfCameras(&numCams);
  if (error != FlyCapture2::PGRERROR_OK)
  {
    this->printError(&error);
    return false;
  }

  if (numCams <= 0)
    return false;

  if(camId >= numCams)
  {
    std::cerr << "PGRSource::init(): camId " << camId << " exceeds number of cameras, " <<
                 "use camera 0" << std::endl;
    camId = 0;
  }

  if (!this->connectToCamera(camId))
    return false;

  if (!this->startCapture())
    return false;

  // this sets width and height member variables
  this->grab();
  return true;
}


cv::Mat PGRSource::getImage()
{
  this->grab();
  return cv::Mat(data_->processed_image_->GetRows(), data_->processed_image_->GetCols(),
                 CV_8UC3, data_->processed_image_->GetData(),
                 data_->processed_image_->GetStride());

}



bool PGRSource::connectToCamera(unsigned int camId)
{
  FlyCapture2::PGRGuid guid;
  FlyCapture2::Error error = data_->bus_mgr_->GetCameraFromIndex(camId, &guid);
  if (error != FlyCapture2::PGRERROR_OK)
  {
    this->printError(&error);
    return false;
  }

  error = data_->cam_->Connect(&guid);
  if (error != FlyCapture2::PGRERROR_OK)
  {
    this->printError(&error);
    return false;
  }
  return true;
}




bool PGRSource::startCapture()
{
  FlyCapture2::Error error = data_->cam_->StartCapture();
  if (error != FlyCapture2::PGRERROR_OK)
  {
    this->printError(&error);
    return false;
  }

  error = data_->cam_->RetrieveBuffer(data_->raw_image_);
  if (error != FlyCapture2::PGRERROR_OK)
    printf( "PGRSource::startCapture(): grab error\n  %s\n", error.GetDescription() );

  width_ = data_->raw_image_->GetCols();
  height_ = data_->raw_image_->GetRows();

  return true;
}



void PGRSource::grab()
{
  // grab camera image
  FlyCapture2::Error error = data_->cam_->RetrieveBuffer(data_->raw_image_);
  if (error != FlyCapture2::PGRERROR_OK)
    printf( "PGRSource::grab(): %s\n", error.GetDescription() );

  // convert to rgb, this is the image we'll be using
  error = data_->raw_image_->Convert( FlyCapture2::PIXEL_FORMAT_RGB8,
                                      data_->processed_image_ );
  if (error != FlyCapture2::PGRERROR_OK)
    printf( "PGRSource::grab(): %s\n", error.GetDescription() );

  frameNr_++;

  if (frameNr_ > (1 << 30))    // bound is not tight for an unsigned int but i don't care
    frameNr_ = 0;
}





void PGRSource::printError(FlyCapture2::Error *error)
{
  printf("PGRSource Error: %s\n", error->GetDescription());
}

