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
 * Module      : Volume base class
 * Class       : Volume
 * Language    : C++
 * Description : Implementation of Volume base class
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at

 *
 */

#ifndef IUCORE_VOLUME_H
#define IUCORE_VOLUME_H

#include "globaldefs.h"
#include "coredefs.h"

namespace iu{

//! \todo We maybe do not want to have the Volume class in the public dll interface??
class Volume
{
public:
  Volume(IuPixelType pixel_type) :
    pixel_type_(pixel_type), size_()
  {
  }

  virtual ~Volume()
  {
  }

  Volume(const Volume &from) :
    pixel_type_(from.pixelType()), size_(from.size_)
  {
  }

  Volume(IuPixelType pixel_type, unsigned int width, unsigned int height, unsigned int depth) :
    pixel_type_(pixel_type), size_(width, height, depth)
  {
  }

  Volume(IuPixelType pixel_type, const IuSize &size) :
    pixel_type_(pixel_type), size_(size)
  {
  }

  Volume& operator= (const Volume &from)
  {
    // TODO == operator
    this->pixel_type_ = from.pixel_type_;
    this->size_ = from.size_;
    return *this;
  }

  IuPixelType pixelType() const
  {
    return pixel_type_;
  }

  IuSize size() const
  {
    return size_;
  }

  unsigned int width() const
  {
    return size_.width;
  }

  unsigned int height() const
  {
    return size_.height;
  }

  unsigned int depth() const
  {
    return size_.depth;
  }

  /** Returns the number of pixels in the Volume. */
  size_t numel() const
  {
    return (size_.width * size_.height * size_.depth);
  }



  /** Returns the total amount of bytes saved in the data buffer. */
  virtual size_t bytes() const {return 0;};

  /** Returns the distance in bytes between starts of consecutive rows. */
  virtual size_t pitch() const {return 0;};

  /** Returns the distnace in pixels between starts of consecutive rows. */
  virtual size_t stride() const {return 0;};

  /** Returns the bit depth of the data pointer. */
  virtual unsigned int bitDepth() const {return 0;};

  /** Returns flag if the Volume data resides on the device/GPU (TRUE) or host/GPU (FALSE) */
  virtual bool onDevice() const {return false;};

private:
  IuPixelType pixel_type_;
  IuSize size_;
};

} // namespace iuprivate

#endif // IUCORE_Volume_H
