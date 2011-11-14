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
 * Class       : VolumeGpu
 * Language    : C++
 * Description : Definition of volume class for Npp
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUCORE_VOLUME_GPU_H
#define IUCORE_VOLUME_GPU_H

#include "volume.h"
#include "volume_allocator_gpu.h"

namespace iu {

template<typename PixelType, class Allocator, IuPixelType _pixel_type>
class VolumeGpu : public Volume
{
public:
  VolumeGpu() :
    Volume(_pixel_type),
    data_(0), pitch_(0), ext_data_pointer_(false)
  {
  }

  virtual ~VolumeGpu()
  {
    if(!ext_data_pointer_)
    {
      // do not delete externally handeled data pointers.
      Allocator::free(data_);
      data_ = 0;
    }
    pitch_ = 0;
  }

  VolumeGpu(unsigned int _width, unsigned int _height, unsigned int _depth) :
    Volume(_pixel_type, _width, _height, _depth), data_(0), pitch_(0),
    ext_data_pointer_(false)
  {
    data_ = Allocator::alloc(this->size(), &pitch_);
  }

  VolumeGpu(const IuSize& size) :
    Volume(_pixel_type, size), data_(0), pitch_(0),
    ext_data_pointer_(false)
  {
    data_ = Allocator::alloc(this->size(), &pitch_);
  }

  VolumeGpu(const VolumeGpu<PixelType, Allocator, _pixel_type>& from) :
    Volume(from), data_(0), pitch_(0),
    ext_data_pointer_(false)
  {
    data_ = Allocator::alloc(this->size(), &pitch_);
    Allocator::copy(from.data(), from.pitch(), data_, pitch_, this->size());
    this->setRoi(from.roi());
  }

  VolumeGpu(PixelType* _data, unsigned int _width, unsigned int _height, unsigned int _depth,
            size_t _pitch, bool ext_data_pointer = false) :
    Volume(_pixel_type, _width, _height, _depth), data_(0), pitch_(0),
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

      data_ = Allocator::alloc(this->size(), &pitch_);
      Allocator::copy(_data, _pitch, data_, pitch_, this->size());
    }
  }

  // :TODO:
  //VolumeGpu& operator= (const VolumeGpu<PixelType, Allocator>& from);

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
    return true;
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

  /** Returns a volume slice given by a z-offset as ImageGpu
    * @param[in] oz z-offset into the volume
    * @return volume slice at depth z as ImageGpu. Note, the ImageGpu merely holds a pointer to the volume data at depth z,
    * i.e. it does not manage its own data -> changes to the Image are transparent to the volume and vice versa.
    */
  ImageGpu<PixelType, iuprivate::ImageAllocatorGpu<PixelType>, _pixel_type> getSlice(int oz)
  {
    return ImageGpu<PixelType, iuprivate::ImageAllocatorGpu<PixelType>, _pixel_type>(
          &data_[oz*slice_stride()], width(), height(), pitch_, true);
  }

protected:

private:
  PixelType* data_;
  size_t pitch_;
  bool ext_data_pointer_; /**< Flag if data pointer is handled outside the volume class. */
};

} // namespace iuprivate

#endif // IUCORE_VOLUME_GPU_H
