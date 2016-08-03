#pragma once

#include "coredefs.h"

#include <ostream>
#include <typeinfo>

namespace iu{

/** \defgroup Image Pitched memory: Image
 *  \ingroup MemoryManagement
 *  \brief Memory management for Image classes.
 *
 *  This handles the memory management for following pitched memory classes:
 *  - ImageCpu
 *  - ImageGpu
 *
 * The device memory class can be easily passed to CUDA kernels using a special
 * struct. This struct gives the possibility to not only access the data pointer
 * of the image but also other useful information such as size and strides of the
 * object.
 * - ImageGpu::KernelData
 *
 * \todo We maybe do not want to have the Image class in the public dll interface??
 *
 * \{
 */

/** \brief Base class for 2D images (pitched memory). */
class Image
{
public:
  /** Constructor. */
  Image() :
   size_(0,0)
  {
  }

  /** Destructor. */
  virtual ~Image()
  {
  }

  /** Special constructor.
   *  @param width Width of the Image
   *  @param height Height of the Image
   */
  Image( unsigned int width, unsigned int height) :
       size_(width, height)
  {
  }

  /** Special constructor.
   *  @param size Size of the Image
   */
  Image( const IuSize &size) :
      size_(size)
  {
  }

  /** Get the size of the Image
   *  @return Size of the Image
   */
  IuSize size() const
  {
    return size_;
  }

  /** Get the width of the Image.
   *  @return Width of the Image
   */
  unsigned int width() const
  {
    return size_.width;
  }

  /** Get the height of the Image.
   *  @return Height of the Image
   */
  unsigned int height() const
  {
    return size_.height;
  }

  /** Compares the Image type to a target Image.
   *  @param from Target Image.
   *  @return Returns true if target class is of the same type (using RTTI).
   */
  bool sameType(const Image &from)
  {
      return typeid(from)==typeid(*this);
  }

  /** Returns the number of pixels in the Image. */
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
  /** Size of the Image. */
  IuSize size_;

private:
  /** Private copy constructor. */
  Image(const Image&);
  /** Private copy assignment operator. */
  Image& operator=(const Image&);
};

/** \} */ // end of Image

} // namespace iu

