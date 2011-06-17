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
 * Module      : IPP-Connector
 * Class       : ImageIpp
 * Language    : C++
 * Description : Definition of image class for Ipp
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUCORE_IMAGE_IPP_H
#define IUCORE_IMAGE_IPP_H

#include <iudefs.h>
#include <iucore/image.h>
#include <iucore/image_cpu.h>
#include <ippdefs.h>
#include "image_allocator_ipp.h"

namespace iu {

template<typename PixelType, unsigned int NumChannels, class Allocator, IuPixelType _pixel_type>
class ImageIpp : public virtual Image
{
public:
  ImageIpp() :
    Image(_pixel_type, _pixel_type),
    data_(0), pitch_(0), ext_data_pointer_(false), n_channels_(NumChannels)
  {
  }

  virtual ~ImageIpp()
  {
    if(!ext_data_pointer_)
    {
      // do not delete externally handeled data pointers.
      Allocator::free(data_);
      data_ = 0;
    }
    pitch_ = 0;
  }

  ImageIpp(unsigned int _width, unsigned int _height) :
    Image(_pixel_type, _width, _height), data_(0), pitch_(0), ext_data_pointer_(false),
    n_channels_(NumChannels)
  {
    data_ = Allocator::alloc(_width, _height, &pitch_);
  }

  ImageIpp(const IppiSize& size) :
    Image(_pixel_type, size.width, size.height), data_(0), pitch_(0), ext_data_pointer_(false),
    n_channels_(NumChannels)
  {
    data_ = Allocator::alloc(size.width, size.height, &pitch_);
  }

  ImageIpp(const IuSize& size) :
    Image(_pixel_type, size.width, size.height), data_(0), pitch_(0), ext_data_pointer_(false),
    n_channels_(NumChannels)
  {
    data_ = Allocator::alloc(width(), height(), &pitch_);
  }

  ImageIpp(const ImageIpp<PixelType, NumChannels, Allocator, _pixel_type>& from) :
    Image(from), data_(0), pitch_(0),
    n_channels_(NumChannels)
  {
    data_ = Allocator::alloc(width(), height(), &pitch_);
    Allocator::copy(from.data(), from.pitch(), data_, pitch_, this->size());
  }

  ImageIpp(PixelType* _data, unsigned int _width, unsigned int _height,
           size_t _pitch, bool ext_data_pointer = false) :
    Image(_pixel_type, _width, _height), data_(0), pitch_(0),
    ext_data_pointer_(ext_data_pointer), n_channels_(NumChannels)
  {
    if(ext_data_pointer_)
    {
      // This uses the external data pointer as internal data pointer.
      data_ = _data;
      pitch_ = _pitch;
    }
    else
    {
      data_ = Allocator::alloc(width(), height(), &pitch_);
      Allocator::copy(_data, _pitch, data_, pitch_, this->size());
    }
  }
  // :TODO:
  //ImageIpp& operator= (const ImageIpp<PixelType, numChannels, Allocator>& from);

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

  /** Returns the bit depth of the data pointer. */
  virtual unsigned int bitDepth() const
  {
    return 8*sizeof(PixelType);
  }

  /** Returns flag if the image data resides on the device/GPU (TRUE) or host/GPU (FALSE) */
  virtual bool onDevice() const
  {
    return false;
  }

  /** Returns internal size converted to IppiSize. */
  IppiSize ippSize() const
  {
    IppiSize s = {width(), height()};
    return s;
  }

  /** Returns internal  roi converted to IppiRect. */
  IppiRect ippRoi() const
  {
    IppiRect r = {roi().x, roi().y, roi().width, roi().height};
    return r;
  }

  /** Returns the number of channels. */
  unsigned int nChannels() const
  {
    return n_channels_;
  }

  /** Returns a pointer to the pixel data.
   * The pointer can be offset to position \a (ox/oy).
   * @param[in] ox Horizontal offset of the pointer array.
   * @param[in] oy Vertical offset of the pointer array.
   * @return Pointer to the pixel array.
   */
  PixelType* data(int ox = 0, int oy = 0, int channel = 0)
  {
    return &data_[oy * stride() + ox*n_channels_ + channel];
  }
  const PixelType* data(int ox = 0, int oy = 0, int channel = 0) const
  {
    return reinterpret_cast<const PixelType*>(
          &data_[oy * stride() + ox*n_channels_ + channel]);
  }

private:
  PixelType* data_;
  size_t pitch_;
  bool ext_data_pointer_; /**< Flag if data pointer is handled outside the image class. */
  unsigned int n_channels_;
};

} // namespace iuprivate

#endif // IUCORE_IMAGE_IPP_H
