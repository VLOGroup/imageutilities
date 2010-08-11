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
 * Module      : Core
 * Class       : ImageNpp
 * Language    : C++
 * Description : Definition of image class for Npp
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUCORE_IMAGE_NPP_H
#define IUCORE_IMAGE_NPP_H

#include <nppdefs.h>
#include "image.h"
#include "image_allocator_npp.h"

namespace iu {

template<typename PixelType, unsigned int NumChannels, class Allocator>
class ImageNpp : public Image
{
public:
  ImageNpp() :
      data_(0), pitch_(0), n_channels_(NumChannels), ext_data_pointer_(false)
  {
  }

  virtual ~ImageNpp()
  {
    if(!ext_data_pointer_)
    {
      // do not delete externally handeled data pointers.
      Allocator::free(data_);
      data_ = 0;
    }
    pitch_ = 0;
  }

  ImageNpp(unsigned int _width, unsigned int _height) :
      Image(_width, _height), data_(0), pitch_(0), n_channels_(NumChannels),
      ext_data_pointer_(false)
  {
    data_ = Allocator::alloc(_width, _height, &pitch_);
  }

  ImageNpp(const NppiSize& size) :
      Image(size.width, size.height), data_(0), pitch_(0), n_channels_(NumChannels),
      ext_data_pointer_(false)
  {
    data_ = Allocator::alloc(size.width, size.height, &pitch_);
  }

  ImageNpp(const IuSize& size) :
      Image(size), data_(0), pitch_(0), n_channels_(NumChannels),
      ext_data_pointer_(false)
  {
    data_ = Allocator::alloc(width(), height(), &pitch_);
  }

  ImageNpp(const ImageNpp<PixelType, NumChannels, Allocator>& from) :
      Image(from), data_(0), pitch_(0), n_channels_(NumChannels),
      ext_data_pointer_(false)
  {
    data_ = Allocator::alloc(width(), height(), &pitch_);
    Allocator::copy(from.data(), from.pitch(), data_, pitch_, this->nppSize());
    this->setRoi(from.roi());
  }

  ImageNpp(PixelType* _data, unsigned int _width, unsigned int _height,
           size_t _pitch, bool ext_data_pointer = false) :
      Image(_width, _height), data_(0), pitch_(0), n_channels_(NumChannels),
      ext_data_pointer_(ext_data_pointer)
  {
    if(ext_data_pointer_)
    {
      // This uses the external data pointer as internal data pointer.
      data_ = _data;
      pitch_ = _pitch;
    }
    else
    {
      // allocates an internal data pointer and copies the external data onto it.
      if(_data == 0)
        return;

      data_ = Allocator::alloc(width(), height(), &pitch_);
      Allocator::copy(_data, _pitch, data_, pitch_, this->nppSize());
    }
  }

  // :TODO:
  //ImageNpp& operator= (const ImageNpp<PixelType, numChannels, Allocator>& from);

  /** Returns the total amount of bytes saved in the data buffer. */
  size_t bytes() const
  {
    return height()*pitch_;
  }

  /** Returns the distance in bytes between starts of consecutive rows. */
  size_t pitch() const
  {
    return pitch_;
  }

  /** Returns the distnace in pixels between starts of consecutive rows. */
  size_t stride() const
  {
    return pitch_/sizeof(PixelType);
  }

  /** Returns the number of channels. */
  virtual unsigned int nChannels() const
  {
    return n_channels_;
  }

  /** Returns the bit depth of the data pointer. */
  virtual unsigned int bitDepth() const
  {
    return 8*sizeof(PixelType);
  }

  /** Returns flag if the image data resides on the device/GPU (TRUE) or host/GPU (FALSE) */
  virtual bool onDevice() const
  {
    return true;
  }

  /** Returns internal size converted to NppiSize. */
  NppiSize nppSize() const
  {
    NppiSize s = {width(), height()};
    return s;
  }

  /** Returns internal ROI converted to NppiRect. */
  NppiRect nppRoi() const
  {
    NppiRect r = {roi().x, roi().y, roi().width, roi().height};
    return r;
  }

  /** Returns a pointer to the pixel data.
   * The pointer can be offset to position \a (ox/oy).
   * @param[in] ox Horizontal offset of the pointer array.
   * @param[in] oy Vertical offset of the pointer array.
   * @return Pointer to the pixel array.
   */
  PixelType* data(int ox = 0, int oy = 0)
  {
    return &data_[oy * stride() + ox * n_channels_];
  }
  const PixelType* data(int ox = 0, int oy = 0) const
  {
    return reinterpret_cast<const PixelType*>(
        &data_[oy * stride() + ox * n_channels_]);
  }

protected:

private:
  PixelType* data_;
  size_t pitch_;
  unsigned int n_channels_;
  bool ext_data_pointer_; /**< Flag if data pointer is handled outside the image class. */
};

} // namespace iuprivate

#endif // IUCORE_IMAGE_NPP_H
