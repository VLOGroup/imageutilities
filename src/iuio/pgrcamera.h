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
 * Language    : C++
 * Description : Definition of a Camera using the Fyrefly SDK from Point Grey
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 * comment: Many code parts are taken from the examples of the FlyCapture SDK examples
 */


#ifndef PGRCAMERA_H
#define PGRCAMERA_H

#include <QtCore>
#include <iucore/globaldefs.h>

class PGRCameraData;
class PGRCameraImageGrabber;
class PGRCameraImageConsumer;
namespace FlyCapture2 {
  class Error;
}

/** The PGRCamera class handles Cameras of the manufacturer Point Grey Research via the
 * FlyCapture SDK. The current 'beta' release of version 2.0 supports Windows _and_
 * Linux.
 */
class IU_DLLAPI PGRCamera : public QObject
{
  Q_OBJECT;

public:
  /** Constructor for the camera interface. */
  PGRCamera(QObject* parent = NULL, unsigned int cam_id = 0);

  /** Destructor. */
  virtual ~PGRCamera();

  /** Stops the internal threads and disconnects the physical camera. */
  void quit();

  /** Returns the number of available and supported physical cameras. */
  unsigned int getNumOfCameras();

  /** Prints generic camera information to the console.
   * @param[in] id The camera id to connect to. Per default the first camera is used.
   */
  void printCameraInfo();

  /** Returns the current image number (id?). */
  unsigned int getCurImageNumber();

//  /** Gets shared pointer to current image. */
//  Cuda::DeviceMemoryPitched<Npp32f,2>* getDImage();

signals:
  void signalPushNewImage(const QImage &image, unsigned int stride,
                          unsigned int bits_per_pixel);

  /** If the counter for the framenumber (which can be used as a unique id) is nearby a
   * buffer-overflow this signal is emitted.
   */
  void signalResetFrameNumber();

public slots:
  //void slotRetrieveImage(Cuda::HostMemoryHeap<Npp32f,2>* h_image);
  void slotRetrieveQImage(const QImage& image, unsigned int& stride,
                          unsigned int& bits_per_pixel);

protected:
  /** Connect to a camera.
   * @param[in] id The camera id to connect to. Per default the first camera is used.
   */
  bool connectToCamera(unsigned int id = 0);

  /** Disconnect the current camera. */
  bool disconnectCamera();

  /** Starts capturing images from current camera. */
  bool startCapture();

  /** Stops capturing images from current camera. */
  bool stopCapture();

  /** Prints information about the given \a error. */
  void PrintError( FlyCapture2::Error* error );

private:
  PGRCameraData* data_;
  bool ready_; /**< General 'ready' flag for the camera instance. */
  bool capture_mode_; /**< Camera capture status. */

  //DImagePtrType dimage_; /**< Shared pointer to current device image from camera. */
//  Cuda::DeviceMemoryPitched<Npp32f,2>* dimage_; /**< current DeviceImage */
  unsigned int fnr_; /**< Frame number of PGRCamera object (starts with 0). */

  // producer - consumer threads
  PGRCameraImageGrabber* image_grabber_thread_;
  PGRCameraImageConsumer* image_consumer_thread_;
}; // class PGRCamera

#endif // PGRCAMERA_H
