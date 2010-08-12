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
 * Description : Definition/Implementation of a thread responsible to process and provide images from a camera
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#include "FlyCapture2.h"
#include "pgrcameradata.h"
#include "pgrcameraimageconsumer.h"

using namespace FlyCapture2;

//-----------------------------------------------------------------------------
PGRCameraImageConsumer::PGRCameraImageConsumer(PGRCameraData* data) :
  QThread(),
  data_(data),
  thread_running_(false)
{
}

//-----------------------------------------------------------------------------
PGRCameraImageConsumer::~PGRCameraImageConsumer()
{

}

//-----------------------------------------------------------------------------
void PGRCameraImageConsumer::run()
{
  thread_running_ = true;

  while(thread_running_)
  {
    data_->image_mutex_.lock();
    data_->no_image_wait_cond_.wait(&data_->image_mutex_); // Waits till an image is available

    // check here if thread is still running when it got woken up (When PGRCamera calls
    // quit and the consumer is waiting for an image it would return here and the signal
    // to push images can cause problems...)
    if(!thread_running_)
      break;

    // Convert the raw image
    FlyCapture2::Error error = data_->raw_image_->Convert( PIXEL_FORMAT_MONO8, data_->processed_image_ );
    if (error != PGRERROR_OK)
    {
      printf( "PGRCameraImageConsumer: %s\n", error.GetDescription() );
      data_->image_mutex_.unlock();
      return;
    }

    // just for testing
    //data_->processed_image_->Save( "~/bla.png" );

    // get image dimensions
    unsigned int width, height, stride, bits_per_pixel;
    FlyCapture2::PixelFormat pixel_format;
    FlyCapture2::BayerTileFormat bayer_tile_format;
    data_->processed_image_->GetDimensions(&height, &width, &stride, &pixel_format, &bayer_tile_format);
    bits_per_pixel = data_->processed_image_->GetBitsPerPixel();

    // pack image data into a QImage -> due to implicit sharing beyond thread boundaries!
    QImage image (data_->processed_image_->GetData(), width, height, bits_per_pixel*stride,
                  QImage::Format_Mono);

    data_->image_mutex_.unlock();
    emit signalPushNewQImage(image, stride, bits_per_pixel);
    this->usleep(5);
  }
}

//-----------------------------------------------------------------------------
void PGRCameraImageConsumer::quit()
{
  thread_running_ = false;
  QThread::quit();
}
