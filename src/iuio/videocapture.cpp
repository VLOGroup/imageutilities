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
 * Project     : ImageUtilities
 * Module      : IO
 * Class       : VideoCapture
 * Language    : C++
 * Description : Implementation of a  to capture videos from either files or cameras with OpenCVs VideoCapture.
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#include <iucore/copy.h>
#include "videocapture_private.h"
#include "videocapture.h"

/* ****************************************************************************
 *
 *  private interface implementation
 *
 * ***************************************************************************/

namespace iuprivate {

//-----------------------------------------------------------------------------
VideoCapture::VideoCapture() :
  cv::VideoCapture()
{
}

//-----------------------------------------------------------------------------
VideoCapture::VideoCapture(std::string &filename) :
  cv::VideoCapture(filename)
{
}

//-----------------------------------------------------------------------------
VideoCapture::VideoCapture(int device) :
  cv::VideoCapture(device)
{
}

//-----------------------------------------------------------------------------
VideoCapture::~VideoCapture()
{
}

////-----------------------------------------------------------------------------
//bool grab()
//{
//  return cv::VideoCapture::grab();
//}

//-----------------------------------------------------------------------------
bool VideoCapture::retrieve(cv::Mat &image, int channel)
{
  return cv::VideoCapture::retrieve(image, channel);
}

//-----------------------------------------------------------------------------
IuStatus VideoCapture::retrieve(iu::ImageCpu_8u_C1 *image)
{
  if (!this->isOpened())
  {
    printf("VideoCapture: Capture device not ready.\n");
    return IU_ERROR;
  }

 /*if (!this->grab())
 {
   printf("VideoCapture: No more frames available.\n");
   return IU_ERROR;
 }*/

  if (!this->retrieve(frame_))
  {
    printf("VideoCapture: Frame couldn't be retrieved.\n");
    return IU_ERROR;
  }

  // TODO: check size of image
  if(image->size() != IuSize(frame_.cols, frame_.rows))
    return IU_ERROR;

  cv::Mat mat_8u(image->height(), image->width(), CV_8UC1, image->data(), image->pitch());
  // convert to grayscale image
  cv::cvtColor(frame_, mat_8u, CV_BGR2GRAY);
  return IU_SUCCESS;
}

//-----------------------------------------------------------------------------
IuStatus VideoCapture::retrieve(iu::ImageCpu_32f_C1 *image)
{
  if (!this->isOpened())
  {
    printf("VideoCapture: Capture device not ready.\n");
    return IU_ERROR;
  }

//  if (!this->grab())
//  {
//    printf("VideoCapture: No more frames available.\n");
//    return IU_ERROR;
//  }

  if (!this->retrieve(frame_))
  {
    printf("VideoCapture: Frame couldn't be retrieved.\n");
    return IU_ERROR;
  }

  // TODO: check size of image
  if(image->size() != IuSize(frame_.cols, frame_.rows))
    return IU_ERROR;

  cv::Mat mat_8u;
  // convert to grayscale image
  cv::cvtColor(frame_, mat_8u, CV_BGR2GRAY);

  cv::Mat im_mat(image->height(), image->width(), CV_32FC1, image->data(), image->pitch());
  mat_8u.convertTo(im_mat, im_mat.type(), 1.0f/255.0f, 0);
  return IU_SUCCESS;
}

//-----------------------------------------------------------------------------
IuStatus VideoCapture::retrieve(iu::ImageGpu_32f_C1 *image)
{
  IuSize sz = this->size();
  iu::ImageCpu_32f_C1 cpu_image(sz.width, sz.height);
  IuStatus status = this->retrieve(&cpu_image);
  if(status < IU_SUCCESS)
    return IU_ERROR;
  iuprivate::copy(&cpu_image, image);
  return IU_SUCCESS;
}

//-----------------------------------------------------------------------------
IuSize VideoCapture::size()
{
  int width = static_cast<int>(this->get(CV_CAP_PROP_FRAME_WIDTH));
  int height = static_cast<int>(this->get(CV_CAP_PROP_FRAME_HEIGHT));

  // crappy driver (or opencv?) returns 0 in linux with usb cameras

  // first try to get width/height from the frame member
  if(width == 0 || height == 0)
  {
    width  = frame_.cols;
    height  = frame_.rows;
  }

  // if there is no frame grabbed yet you have to get one. This should only be the last fallback!
  if (width == 0 || height == 0)
  {
    if (this->retrieve(frame_))
    {
      width  = frame_.cols;
      height  = frame_.rows;
    }
    else
    {
      if(this->grab())
      {
        if(this->retrieve(frame_))
        {
          width  = frame_.cols;
          height  = frame_.rows;
        }
        else
          printf("VideoCapture: Frame couldn't be retrieved.\n");
      }
      else
        printf("VideoCapture: Frame couldn't be grabbed.\n");
    }
  }

  //  printf("w/h = %d/%d\n", width, height);
  IuSize sz(width, height);
  return sz;
}

int VideoCapture::getFPS()
{
  if (!this->isOpened())
  {
    printf("VideoCapture: Capture device not ready.\n");
    return 0;
  }
  
  return this->get(CV_CAP_PROP_FPS);
}

int VideoCapture::totalFrameCount()
{
  if (!this->isOpened())
  {
    printf("VideoCapture: Capture device not ready.\n");
    return 0;
  }
  
  return this->get(CV_CAP_PROP_FRAME_COUNT);
}


int VideoCapture::frameIdx()
{
  if (!this->isOpened())
  {
    printf("VideoCapture: Capture device not ready.\n");
    return 0;
  }
  
  return this->get(CV_CAP_PROP_POS_FRAMES);
}

} // namespace iuprivate



/* ****************************************************************************
 *
 *  public interface implementation
 *
 * ***************************************************************************/
namespace iu {

VideoCapture::VideoCapture() { vidcap_ = new iuprivate::VideoCapture(); }
VideoCapture::VideoCapture(std::string& filename) { vidcap_ = new iuprivate::VideoCapture(filename); }
VideoCapture::VideoCapture(int device) { vidcap_ = new iuprivate::VideoCapture(device); }
VideoCapture::~VideoCapture() { delete(vidcap_); }

bool VideoCapture::grab() { return vidcap_->grab(); }

IuStatus VideoCapture::retrieve(iu::ImageCpu_8u_C1 *image) { return vidcap_->retrieve(image); }
IuStatus VideoCapture::retrieve(iu::ImageCpu_32f_C1 *image) { return vidcap_->retrieve(image); }
IuStatus VideoCapture::retrieve(iu::ImageGpu_32f_C1 *image) { return vidcap_->retrieve(image); }

IuSize VideoCapture::size() { return vidcap_->size(); }
int VideoCapture::getFPS() { return vidcap_->getFPS(); }
int VideoCapture::totalFrameCount() { return vidcap_->totalFrameCount(); }
int VideoCapture::frameIdx() { return vidcap_->frameIdx(); }

} // namespace iu
