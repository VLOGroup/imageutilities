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
 * Class       : ImageGpu
 * Language    : C++
 * Description : Definition of image class for Gpu
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUCORE_IMAGE_GPU_H
#define IUCORE_IMAGE_GPU_H

#include "image.h"
#include "image_allocator_gpu.h"

namespace iu {

template<typename PixelType, class Allocator, IuPixelType _pixel_type>
class ImageGpu : public Image
{
public:
  ImageGpu() :
    Image(_pixel_type),
    data_(0), pitch_(0), ext_data_pointer_(false)
  {
  }

  virtual ~ImageGpu()
  {
    if(!ext_data_pointer_)
    {
      // do not delete externally handeled data pointers.
      Allocator::free(data_);
      data_ = 0;
    }
    pitch_ = 0;
  }

  ImageGpu(unsigned int _width, unsigned int _height) :
      Image(_pixel_type, _width, _height), data_(0), pitch_(0),
      ext_data_pointer_(false)
  {
    data_ = Allocator::alloc(this->size(), &pitch_);
  }

  ImageGpu(const IuSize& size) :
      Image(_pixel_type, size), data_(0), pitch_(0),
      ext_data_pointer_(false)
  {
    data_ = Allocator::alloc(size, &pitch_);
  }

  ImageGpu(const ImageGpu<PixelType, Allocator, _pixel_type>& from) :
      Image(from), data_(0), pitch_(0),
      ext_data_pointer_(false)
  {
    data_ = Allocator::alloc(from.size(), &pitch_);
    Allocator::copy(from.data(), from.pitch(), data_, pitch_, this->size());
    this->setRoi(from.roi());
  }

  ImageGpu(PixelType* _data, unsigned int _width, unsigned int _height,
           size_t _pitch, bool ext_data_pointer = false) :
      Image(_pixel_type, _width, _height), data_(0), pitch_(0), ext_data_pointer_(ext_data_pointer)
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

      data_ = Allocator::alloc(this->size(), &pitch_);
      Allocator::copy(_data, _pitch, data_, pitch_, this->size());
    }
  }

  PixelType getPixel(unsigned int x, unsigned int y)
  {
    PixelType value;
    cudaMemcpy2D(&value, sizeof(PixelType), &data_[y*stride()+x], pitch_,
                 sizeof(PixelType), 1, cudaMemcpyDeviceToHost);
    return value;
  }

  // :TODO:
  //ImageGpu& operator= (const ImageGpu<PixelType, Allocator>& from);

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

  /** Returns the distance in pixels between starts of consecutive rows. 
   * @todo: returned value is invalid if sizeof(PixelType) % 2 != 0, e.g. for *_C3 images. Use
   * pitch in this case to calculate memory addresses by hand.
   */
  virtual size_t stride() const
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
    return true;
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

protected:
  PixelType* data_;
  size_t pitch_;
  bool ext_data_pointer_; /**< Flag if data pointer is handled outside the image class. */
};

} // namespace iuprivate

#endif // IUCORE_IMAGE_GPU_H

