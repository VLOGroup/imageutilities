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
 * Project     : VMLibraries
 * Module      : Image base class
 * Class       : Image
 * Language    : C++
 * Description : Implementation of image base class
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at

 *
 */

#ifndef IUCORE_IMAGE_H
#define IUCORE_IMAGE_H

#include "globaldefs.h"
#include "coredefs.h"

#include <ostream>

namespace iu{

//! \todo We maybe do not want to have the Image class in the public dll interface??
class Image
{
public:
  Image() :
   size_(0,0)
  {
  }

  virtual ~Image()
  {
  }

  Image(const Image &from) :
    size_(from.size_)
  {
  }

  Image( unsigned int width, unsigned int height) :
       size_(width, height)
  {
  }

  Image( const IuSize &size) :
      size_(size)
  {
  }

  Image& operator= (const Image &from)
  {
    // TODO == operator
    this->size_ = from.size_;
    return *this;
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

  /** Returns the number of pixels in the image. */
  size_t numel() const
  {
    return (size_.width * size_.height);
  }

  /** Returns the total amount of bytes saved in the data buffer. */
  virtual size_t bytes() const {return 0;};

  /** Returns the distance in bytes between starts of consecutive rows. */
  virtual size_t pitch() const {return 0;};

  /** Returns the distance in pixels between starts of consecutive rows. */
  virtual size_t stride() const {return 0;};

  /** Returns the bit depth of the data pointer. */
  virtual unsigned int bitDepth() const {return 0;};

  /** Returns flag if the image data resides on the device/GPU (TRUE) or host/GPU (FALSE) */
  virtual bool onDevice() const {return false;};

  /** Operator<< overloading. Output of Image class. */
  friend std::ostream& operator<<(std::ostream & out,
                                  Image const& image)
  {
    out << "Image: " << image.size() << " stride="
        << image.stride() << " onDevice=" << image.onDevice();
    return out;
  }

protected:
  IuSize size_;
};

} // namespace iuprivate

#endif // IUCORE_IMAGE_H
