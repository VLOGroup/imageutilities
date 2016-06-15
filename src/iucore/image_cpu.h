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
 * Class       : ImageCpu
 * Language    : C++
 * Description : Definition of image class for Ipp
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IMAGE_CPU_H
#define IMAGE_CPU_H

#include <thrust/memory.h>
#include "image.h"
#include "image_allocator_cpu.h"

namespace iu {

template<typename PixelType, class Allocator>
class ImageCpu : public Image
{
public:
  ImageCpu() :
    Image(),
    data_(0), pitch_(0), ext_data_pointer_(false)
  {
  }

  virtual ~ImageCpu()
  {
    if(!ext_data_pointer_)
    {
      // do not delete externally handeled data pointers.
      Allocator::free(data_);
      data_ = 0;
    }
    pitch_ = 0;
  }

  ImageCpu(unsigned int _width, unsigned int _height) :
    Image(_width, _height), data_(0), pitch_(0),
    ext_data_pointer_(false)
  {
    data_ = Allocator::alloc(_width, _height, &pitch_);
  }

  ImageCpu(const IuSize& size) :
    Image(size.width, size.height), data_(0), pitch_(0),
    ext_data_pointer_(false)
  {
    data_ = Allocator::alloc(size.width, size.height, &pitch_);
  }

  ImageCpu(const ImageCpu<PixelType, Allocator>& from) :
    Image(from), data_(0), pitch_(0),
    ext_data_pointer_(false)
  {
      if (from.ext_data_pointer_)      // external image stays external when copied
      {
          data_ = from.data_;
          pitch_ = from.pitch_;
          ext_data_pointer_ = from.ext_data_pointer_;
      }
      else
      {
          data_ = Allocator::alloc(width(), height(), &pitch_);
          Allocator::copy(from.data(), from.pitch(), data_, pitch_, this->size());
      }
  }

  ImageCpu(PixelType* _data, unsigned int _width, unsigned int _height,
           size_t _pitch, bool ext_data_pointer = false) :
    Image(_width, _height), data_(0), pitch_(0),
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
      data_ = Allocator::alloc(width(), height(), &pitch_);
      Allocator::copy(_data, _pitch, data_, pitch_, this->size());
    }
  }

  ImageCpu& operator=(const ImageCpu<PixelType, Allocator>& from)
  {
      // destructor of ImageCpu does not free data in case ext_data_pointer_ is true.
      // We support assignment of ImageCpu that wrap external memory by simply copying the data pointer. Reasoning: can have
      // as many ImageCpu's referencing the external memory as we like, since we are not responsible for that data.
      // Support of assignment for non external data would require deep copy or reference counting.
      if (!(from.ext_data_pointer_))
      {
          printf("Error: ImageCpu supports assignment operator only for externally handled data\n");
          throw IuException("ImageCpu supports assignment operator only for externally handled data");
      }

      Image::operator=(from);
      data_ = from.data_;      // copy external (!) data pointer
      pitch_ = from.pitch();
      ext_data_pointer_ = from.ext_data_pointer_;

      return *this;
  }

  /** Returns a pointer to the pixel data.
   * The pointer can be offset to position \a (ox/oy).
   * @param[in] ox Horizontal offset of the pointer array.
   * @param[in] oy Vertical offset of the pointer array.
   * @return Pointer to the pixel array.
   */
  PixelType* data(int ox = 0, int oy = 0)
  {
    return &data_[oy * stride() + ox];
  }
  const PixelType* data(int ox = 0, int oy = 0) const
  {
    return reinterpret_cast<const PixelType*>(
          &data_[oy * stride() + ox]);
  }

  /** Get Pixel value at position x,y. */
  PixelType getPixel(unsigned int x, unsigned int y)
  {
    return *data(x, y);
  }

  /** Get Pointer to beginning of row \a row (y index).
   * This enables the usage of [y][x] operator.
   */
  PixelType* operator[](unsigned int row)
  {
    return data_+row*stride();
  }

  // :TODO:
  //ImageCpu& operator= (const ImageCpu<PixelType, Allocator>& from);

  /** Returns the total amount of bytes saved in the data buffer. */
  virtual size_t bytes() const
  {
    return height()*pitch_;
  }

  /** Returns the distance in bytes between starts of consecutive rows. */
  virtual size_t pitch() const
  {
    return pitch_;
  }

  /** Returns the distnace in pixels between starts of consecutive rows. */
  virtual size_t stride() const
  {
    return pitch_/sizeof(PixelType);
  }

  /** Returns the bit depth of the data pointer. */
  virtual unsigned int bitDepth() const
  {
    return 8*sizeof(PixelType);
  }

  thrust::pointer<PixelType, thrust::host_system_tag> begin(void)
  {
      return thrust::pointer<PixelType, thrust::host_system_tag>(data());
  }

  thrust::pointer<PixelType, thrust::host_system_tag> end(void)
  {
      return thrust::pointer<PixelType, thrust::host_system_tag>(data()+stride()*height());
  }

  /** Returns flag if the image data resides on the device/GPU (TRUE) or host/GPU (FALSE) */
  virtual bool onDevice() const
  {
    return false;
  }

protected:
  PixelType* data_;
  size_t pitch_;
  bool ext_data_pointer_; /**< Flag if data pointer is handled outside the image class. */
};

} // namespace iuprivate


#endif // IMAGE_CPU_H
