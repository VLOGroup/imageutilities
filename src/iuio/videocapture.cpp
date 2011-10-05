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
#include <iostream>

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

//-----------------------------------------------------------------------------
bool VideoCapture::retrieve(cv::Mat &image, int channel)
{
  return cv::VideoCapture::retrieve(image, channel);
}

//-----------------------------------------------------------------------------
void VideoCapture::retrieve(iu::ImageCpu_8u_C1 *image)
{
  if (!this->isOpened())
    throw IuException("VideoCapture: Capture device not ready.\n",
                      __FILE__, __FUNCTION__, __LINE__);

  if (!this->retrieve(frame_))
    throw IuException("VideoCapture: Frame could not be fetched.\n",
                      __FILE__, __FUNCTION__, __LINE__);

  // check size of image
  if(image->size() != IuSize(frame_.cols, frame_.rows))
    throw IuException("VideoCapture: Given image size does not match with grabbed frame size. Could not copy data.\n",
                      __FILE__, __FUNCTION__, __LINE__);

  cv::Mat mat_8u(image->height(), image->width(), CV_8UC1, image->data(), image->pitch());
  // convert to grayscale image
  cv::cvtColor(frame_, mat_8u, CV_BGR2GRAY);
}

//-----------------------------------------------------------------------------
void VideoCapture::retrieve(iu::ImageCpu_32f_C1 *image)
{
  if (!this->isOpened())
    throw IuException("VideoCapture: Capture device not ready.\n",
                      __FILE__, __FUNCTION__, __LINE__);

  if (!this->retrieve(frame_))
    throw IuException("VideoCapture: Frame could not be fetched.\n",
                      __FILE__, __FUNCTION__, __LINE__);

  // check size of image
  if(image->size() != IuSize(frame_.cols, frame_.rows))
    throw IuException("VideoCapture: Given image size does not match with grabbed frame size. Could not copy data.\n",
                      __FILE__, __FUNCTION__, __LINE__);

  cv::Mat mat_8u;
  // convert to grayscale image
  cv::cvtColor(frame_, mat_8u, CV_BGR2GRAY);
  cv::Mat im_mat(image->height(), image->width(), CV_32FC1, image->data(), image->pitch());
  mat_8u.convertTo(im_mat, im_mat.type(), 1.0f/255.0f, 0);
}

//-----------------------------------------------------------------------------
void VideoCapture::retrieve(iu::ImageGpu_8u_C1 *image)
{
  IuSize sz = this->size();
  iu::ImageCpu_8u_C1 cpu_image(sz.width, sz.height);
  this->retrieve(&cpu_image);
  iuprivate::copy(&cpu_image, image);
}

//-----------------------------------------------------------------------------
void VideoCapture::retrieve(iu::ImageGpu_32f_C1 *image)
{
  IuSize sz = this->size();
  iu::ImageCpu_32f_C1 cpu_image(sz.width, sz.height);
  this->retrieve(&cpu_image);
  iuprivate::copy(&cpu_image, image);
}

//-----------------------------------------------------------------------------
IuSize VideoCapture::size()
{
  int width = static_cast<int>(this->get(CV_CAP_PROP_FRAME_WIDTH));
  int height = static_cast<int>(this->get(CV_CAP_PROP_FRAME_HEIGHT));

  // crappy driver (or opencv?) returns 0 in linux with usb cameras
  // with 2.2 it works in linux!!!

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

//-----------------------------------------------------------------------------
int VideoCapture::getFPS()
{
  if (!this->isOpened())
    throw IuException("VideoCapture: Capture device not ready.\n",
                      __FILE__, __FUNCTION__, __LINE__);

  return this->get(CV_CAP_PROP_FPS);
}

//-----------------------------------------------------------------------------
int VideoCapture::setFPS(int fps)
{
  if (!this->isOpened())
    throw IuException("VideoCapture: Capture device not ready.\n",
                      __FILE__, __FUNCTION__, __LINE__);

  return this->set(CV_CAP_PROP_FPS, fps);
}

//-----------------------------------------------------------------------------
int VideoCapture::getFrameWidth()
{
  if (!this->isOpened())
    throw IuException("VideoCapture: Capture device not ready.\n",
                      __FILE__, __FUNCTION__, __LINE__);

  return this->get(CV_CAP_PROP_FRAME_WIDTH);
}

//-----------------------------------------------------------------------------
int VideoCapture::setFrameWidth(int width)
{
  if (!this->isOpened())
    throw IuException("VideoCapture: Capture device not ready.\n",
                      __FILE__, __FUNCTION__, __LINE__);

  return this->set(CV_CAP_PROP_FRAME_WIDTH, width);
}

//-----------------------------------------------------------------------------
int VideoCapture::getFrameHeight()
{
  if (!this->isOpened())
    throw IuException("VideoCapture: Capture device not ready.\n",
                      __FILE__, __FUNCTION__, __LINE__);

  return this->get(CV_CAP_PROP_FRAME_HEIGHT);
}

//-----------------------------------------------------------------------------
int VideoCapture::setFrameHeight(int height)
{
  if (!this->isOpened())
    throw IuException("VideoCapture: Capture device not ready.\n",
                      __FILE__, __FUNCTION__, __LINE__);

  return this->set(CV_CAP_PROP_FRAME_HEIGHT, height);
}

//-----------------------------------------------------------------------------
int VideoCapture::totalFrameCount()
{
  if (!this->isOpened())
    throw IuException("VideoCapture: Capture device not ready.\n",
                      __FILE__, __FUNCTION__, __LINE__);

  return this->get(CV_CAP_PROP_FRAME_COUNT);
}

//-----------------------------------------------------------------------------
int VideoCapture::frameIdx()
{
  if (!this->isOpened())
    throw IuException("VideoCapture: Capture device not ready.\n",
                      __FILE__, __FUNCTION__, __LINE__);

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

void VideoCapture::retrieve(iu::ImageCpu_8u_C1 *image) { return vidcap_->retrieve(image); }
void VideoCapture::retrieve(iu::ImageCpu_32f_C1 *image) { return vidcap_->retrieve(image); }
void VideoCapture::retrieve(iu::ImageGpu_32f_C1 *image) { return vidcap_->retrieve(image); }

IuSize VideoCapture::size() { return vidcap_->size(); }
int VideoCapture::getFPS() { return vidcap_->getFPS(); }
int VideoCapture::setFPS(int fps) { return vidcap_->setFPS(fps); }
int VideoCapture::getFrameWidth() { return vidcap_->getFrameWidth(); }
int VideoCapture::setFrameWidth(int width) { return vidcap_->setFrameWidth(width); }
int VideoCapture::getFrameHeight() { return vidcap_->getFrameHeight(); }int VideoCapture::totalFrameCount() { return vidcap_->totalFrameCount(); }
int VideoCapture::setFrameHeight(int height) { return vidcap_->setFrameHeight(height); }
int VideoCapture::frameIdx() { return vidcap_->frameIdx(); }
double VideoCapture::get(int prop_id)  { return vidcap_->get(prop_id); }
bool VideoCapture::set(int prop_id, double value) { return vidcap_->set(prop_id, value); }

} // namespace iu
