#pragma once

#include "coredefs.h"

#include <ostream>
#include <typeinfo>

namespace iu{
/** \defgroup Volume Pitched Memory: Volume
 *  \ingroup MemoryManagement
 *  \brief Memory management for Volume classes.
 *
 *  This handles the memory management for following pitched memory classes:
 *  - VolumeCpu
 *  - VolumeGpu
 *
 * The device memory class can be easily passed to CUDA kernels using a special
 * struct. This struct gives the possibility to not only access the data pointer
 * of the image but also other useful information such as size and strides of the
 * object.
 * - VolumeGpu::KernelData
 *
 * \todo We maybe do not want to have the Volume class in the public dll interface??
 *
 * \{
 */

/** \brief Base class for 3D volumes (pitched memory). */
class Volume
{
public:
  /** Constructor. */
  Volume() :
    size_()
  {
  }

  /** Destructor. */
  virtual ~Volume()
  {
  }

  /** Special constructor.
   *  @param width Width of the Volume
   *  @param height Height of the Volume
   *  @param depth Depth of the Volume
   */
  Volume(unsigned int width, unsigned int height, unsigned int depth) :
    size_(width, height, depth)
  {
  }

  /** Special constructor.
   *  @param size Size of the Volume
   */
  Volume(const IuSize &size) :
    size_(size)
  {
  }

  /** Compares the Volume type to a target Volume.
   *  @param from Target Volume.
   *  @return Returns true if target class is of the same type (using RTTI).
   */
  bool sameType(const Volume &from)
  {
      return typeid(from)==typeid(*this);
  }

  /** Get the size of the Volume
   *  @return Size of the Volume
   */
  IuSize size() const
  {
    return size_;
  }

  /** Get the width of the Volume
   *  @return Width of the Volume
   */
  unsigned int width() const
  {
    return size_.width;
  }

  /** Get the height of the Volume
   *  @return Height of the Volume
   */
  unsigned int height() const
  {
    return size_.height;
  }

  /** Get the depth of the Volume
   *  @return Depth of the Volume
   */
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

  /** Returns the distance in pixels between starts of consecutive rows. */
  virtual size_t stride() const {return 0;};

  /** Returns the distance in pixels between starts of consecutive slices. */
  virtual size_t slice_stride() const {return 0;};

  /** Returns the bit depth of the data pointer. */
  virtual unsigned int bitDepth() const {return 0;};

  /** Returns flag if the Volume data resides on the device/GPU (TRUE) or host/GPU (FALSE) */
  virtual bool onDevice() const {return false;};

  /** Operator<< overloading. Output of Volume class. */
  friend std::ostream& operator<<(std::ostream & out,
                                  Volume const& volume)
  {
    out << "Volume: " << volume.size() << " stride="
        << volume.stride() << " slice_stride=" << volume.slice_stride()
        << " onDevice=" << volume.onDevice();
    return out;
  }

private:
  /** Size of the Volume. */
  IuSize size_;

private:
  /** Private copy constructor. */
  Volume(const Volume&);
  /** Private copy assignment operator. */
  Volume& operator=(const Volume&);
};

/** \} */ // end of Volume

} // namespace iuprivate

