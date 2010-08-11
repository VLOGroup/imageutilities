/*
 * Copyright (c) ICG. All rights reserved.
 *
 * Institute for Computer Graphics and Vision
 * Graz University of Technology / Austria
 *
 *
 * This software is distributed WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 * PURPOSE.  See the above copyright notices for more information.
 *
 *
 * Project     : vmgpu
 * Module      : Tools
 * Language    : C++/CUDA
 * Description : Implementation of a Camera using the Fyrefly SDK from Point Grey
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 * comment: Many code parts are taken from the examples of the FlyCapture SDK examples
 */

#include <QObject>
#include <QImage>
#include <FlyCapture2.h>

// includes, local
#include "pgrcameradata.h"
#include "pgrcameraimagegrabber.h"
#include "pgrcameraimageconsumer.h"
#include "pgrcamera.h"

//#include "ImageWriter.h"

using namespace FlyCapture2;

//-----------------------------------------------------------------------------
PGRCamera::PGRCamera(QObject* parent, unsigned int cam_id) :
  QObject(parent),
  ready_(false),
  capture_mode_(false),
//  dimage_(NULL),
  fnr_(0)
{
  data_ = new PGRCameraData();

  unsigned int num_cams = this->getNumOfCameras();
  if(num_cams > 0)
  {
    if(cam_id >= num_cams)
      cam_id = 0;


    this->connectToCamera(cam_id);
    this->startCapture();

    image_grabber_thread_ = new PGRCameraImageGrabber(data_);
    image_consumer_thread_ = new PGRCameraImageConsumer(data_);

    qRegisterMetaType<QImage>("QImage");
    connect(image_consumer_thread_, SIGNAL(signalPushNewQImage(const QImage &, unsigned int&, unsigned int&)),
            this, SLOT(slotRetrieveQImage(const QImage&, unsigned int&, unsigned int&)),
            Qt::DirectConnection);

    image_grabber_thread_->start();
    image_consumer_thread_->start();
    ready_ = true;
  }
}

//-----------------------------------------------------------------------------
PGRCamera::~PGRCamera()
{

  if(ready_)
    this->quit();

  // delete threads
  delete(image_grabber_thread_);
  delete(image_consumer_thread_);

  // delete data structure
  delete(data_);
}

//-----------------------------------------------------------------------------
void PGRCamera::quit()
{
  if(ready_)
  {
    // stop threads.
    image_grabber_thread_->quit();
    image_grabber_thread_->wait();

    image_consumer_thread_->quit();
    data_->no_image_wait_cond_.wakeAll();
    image_consumer_thread_->wait();

    // disconnect and delete camera
    if(data_->cam_->IsConnected())
    {
      if(capture_mode_)
        this->stopCapture();
      this->disconnectCamera();
    }
  }
}

//-----------------------------------------------------------------------------
unsigned int PGRCamera::getNumOfCameras()
{
  unsigned int num_cams;
  FlyCapture2::Error error = data_->bus_mgr_->GetNumOfCameras(&num_cams);
  if (error != PGRERROR_OK)
  {
    this->PrintError(&error);
    return -1;
  }
  return num_cams;
}

//-----------------------------------------------------------------------------
void PGRCamera::printCameraInfo()
{
  if(data_->cam_ == NULL)
  {
    printf("\n *** NO CAMERA HANDLED FOR THE MOMENT ***\n\n");
    return;
  }

  CameraInfo cam_info;
  FlyCapture2::Error error = data_->cam_->GetCameraInfo(&cam_info);
  if (error != PGRERROR_OK)
  {
    this->PrintError(&error);
    return;
  }

  printf(
    "\n*** CAMERA INFORMATION ***\n"
    "Serial number - %u\n"
    "Camera model - %s\n"
    "Camera vendor - %s\n"
    "Sensor - %s\n"
    "Resolution - %s\n"
    "Firmware version - %s\n"
    "Firmware build time - %s\n\n",
    cam_info.serialNumber,
    cam_info.modelName,
    cam_info.vendorName,
    cam_info.sensorInfo,
    cam_info.sensorResolution,
    cam_info.firmwareVersion,
    cam_info.firmwareBuildTime );
}

//-----------------------------------------------------------------------------
unsigned int PGRCamera::getCurImageNumber()
{
  return(fnr_);
}

//-----------------------------------------------------------------------------
//Cuda::DeviceMemoryPitched<Npp32f,2>* PGRCamera::getDImage()
//{
//  return dimage_;
//}


//-----------------------------------------------------------------------------
bool PGRCamera::connectToCamera(unsigned int id)
{
  if(data_->cam_->IsConnected())
  {
    printf("\n ** The camera is already connected to a physical camera. ** \n\n");
    return false;
  }
  PGRGuid guid;
  FlyCapture2::Error error = data_->bus_mgr_->GetCameraFromIndex(id, &guid);
  if (error != PGRERROR_OK)
  {
    this->PrintError(&error);
    return false;
  }

  error = data_->cam_->Connect(&guid);
  if (error != PGRERROR_OK)
  {
    this->PrintError(&error);
    return false;
  }
  return true;
}

//-----------------------------------------------------------------------------
bool PGRCamera::disconnectCamera()
{
  // Disconnect the camera
  FlyCapture2::Error error = data_->cam_->Disconnect();
  if (error != PGRERROR_OK)
  {
    this->PrintError( &error );
    return false;
  }
  return true;
}

//-----------------------------------------------------------------------------
bool PGRCamera::startCapture()
{
  // Start capturing images
  FlyCapture2::Error error = data_->cam_->StartCapture();
  if (error != PGRERROR_OK)
  {
    this->PrintError(&error);
    return false;
  }
  capture_mode_ = true;
  return true;
}

//-----------------------------------------------------------------------------
bool PGRCamera::stopCapture()
{
  // Stop capturing images
  FlyCapture2::Error error = data_->cam_->StopCapture();
  if (error != PGRERROR_OK)
  {
    this->PrintError(&error);
    return false;
  }
  capture_mode_ = false;
  return true;
}

//-----------------------------------------------------------------------------
void PGRCamera::slotRetrieveQImage(const QImage& image, unsigned int& stride,
                                   unsigned int& bits_per_pixel)
{
  if(fnr_ > 60000)
  {
    fnr_ = 0;
    emit signalResetFrameNumber();
  }
  ++fnr_;

  int _stride = stride;
  int _bits = bits_per_pixel;
  emit signalPushNewImage(image, _stride, _bits);
}


//-----------------------------------------------------------------------------
void PGRCamera::PrintError( FlyCapture2::Error* error )
{
  printf( "PGRCamera: %s\n", error->GetDescription() );
}
