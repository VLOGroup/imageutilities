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
 * Class       : VolumeCpu
 * Language    : C++
 * Description : Definition of volume class for Npp
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUCORE_VOLUME_CPU_H
#define IUCORE_VOLUME_CPU_H

#include "volume.h"
#include "volume_allocator_cpu.h"

namespace iu {

template<typename PixelType, class Allocator>
class VolumeCpu : public Volume
{
public:
  VolumeCpu() :
    Volume(),
    data_(0), pitch_(0), ext_data_pointer_(false)
  {
  }

  virtual ~VolumeCpu()
  {
    if(!ext_data_pointer_)
    {
      // do not delete externally handeled data pointers.
      Allocator::free(data_);
      data_ = 0;
    }
    pitch_ = 0;
  }

  VolumeCpu(unsigned int _width, unsigned int _height, unsigned int _depth) :
    Volume(_width, _height, _depth),
    data_(0), pitch_(0),
    ext_data_pointer_(false)
  {
    data_ = Allocator::alloc(_width, _height, _depth, &pitch_);
  }

  VolumeCpu(const IuSize& size) :
    Volume(size), data_(0), pitch_(0),
    ext_data_pointer_(false)
  {
    data_ = Allocator::alloc(size.width, size.height, size.depth, &pitch_);
  }

  VolumeCpu(const VolumeCpu<PixelType, Allocator>& from) :
    Volume(from),
    data_(0), pitch_(0), ext_data_pointer_(false)
  {
    data_ = Allocator::alloc(from.width(), from.height(), from.depth(), &pitch_);
    Allocator::copy(from.data(), from.pitch(), data_, pitch_, this->size());
    this->setRoi(from.roi());
  }

  VolumeCpu(PixelType* _data, unsigned int _width, unsigned int _height, unsigned int _depth,
            size_t _pitch, bool ext_data_pointer = false) :
    Volume(_width, _height, _depth),
    data_(0), pitch_(0), ext_data_pointer_(ext_data_pointer)
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

      data_ = Allocator::alloc(_width, _height, _depth, &pitch_);
      Allocator::copy(_data, _pitch, data_, pitch_, this->size());
    }
  }

  // :TODO:
  //VolumeCpu& operator= (const VolumeCpu<PixelType, Allocator>& from);

  /** Returns the total amount of bytes saved in the data buffer. */
  size_t bytes() const
  {
    return depth()*height()*pitch_;
  }

  /** Returns the distance in bytes between starts of consecutive rows. */
  size_t pitch() const
  {
    return pitch_;
  }

  /** Returns the distance in bytes between starts of consecutive slices. */
  size_t slice_pitch() const
  {
    return height()*pitch_;
  }

  /** Returns the distnace in pixels between starts of consecutive rows. */
  size_t stride() const
  {
    return pitch_/sizeof(PixelType);
  }

  /** Returns the distnace in pixels between starts of consecutive slices. */
  size_t slice_stride() const
  {
    return height()*pitch_/sizeof(PixelType);
  }

  /** Returns the bit depth of the data pointer. */
  virtual unsigned int bitDepth() const
  {
    return 8*sizeof(PixelType);
  }

  /** Returns flag if the volume data resides on the device/GPU (TRUE) or host/GPU (FALSE) */
  virtual bool onDevice() const
  {
    return false;
  }


  /** Returns a pointer to the pixel data.
   * The pointer can be offset to position \a (ox/oy).
   * @param[in] ox Horizontal offset of the pointer array.
   * @param[in] oy Vertical offset of the pointer array.
   * @return Pointer to the pixel array.
   */
  PixelType* data(int ox = 0, int oy = 0, int oz = 0)
  {
    return &data_[oz*slice_stride() + oy*stride() + ox];
  }
  const PixelType* data(int ox = 0, int oy = 0, int oz = 0) const
  {
    return reinterpret_cast<const PixelType*>(
          &data_[oz*slice_stride() + oy*stride() + ox]);
  }

  /** Returns a volume slice given by a z-offset as ImageCpu
    * @param[in] oz z-offset into the volume
    * @return volume slice at depth z as ImageCpu. Note, the ImageCpu merely holds a pointer to the volume data at depth z,
    * i.e. it does not manage its own data -> changes to the Image are transparent to the volume and vice versa.
    */
  ImageCpu<PixelType, iuprivate::ImageAllocatorCpu<PixelType> > getSlice(int oz)
  {
    return ImageCpu<PixelType, iuprivate::ImageAllocatorCpu<PixelType> >(&data_[oz*slice_stride()], width(), height(), pitch_, true);
  }
  
  /** Get Pixel value at position x,y,z. */
  PixelType getPixel(unsigned int x, unsigned int y, unsigned int z)
  {
    return *data(x, y, z);
  }

protected:

private:
  PixelType* data_;
  size_t pitch_;
  bool ext_data_pointer_; /**< Flag if data pointer is handled outside the volume class. */
};

} // namespace iuprivate

#endif // IUCORE_VOLUME_CPU_H
