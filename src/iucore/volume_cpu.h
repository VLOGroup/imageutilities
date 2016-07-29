
#ifndef IUCORE_VOLUME_CPU_H
#define IUCORE_VOLUME_CPU_H

#include "volume.h"
#include "volume_allocator_cpu.h"

template<typename, int> class ndarray_ref;

namespace iu {
/** \brief Host 3D volume class (pitched memory).
 *  \ingroup Volume
 */
template<typename PixelType, class Allocator>
class VolumeCpu : public Volume
{
public:
  /** Constructor. */
  VolumeCpu() :
    Volume(),
    data_(0), pitch_(0), ext_data_pointer_(false)
  {
  }

  /** Destructor. */
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

  /** Special constructor.
   *  @param _width Width of the Volume
   *  @param _height Height of the Volume
   *  @param _depth Depth of the Volume
   */
  VolumeCpu(unsigned int _width, unsigned int _height, unsigned int _depth) :
    Volume(_width, _height, _depth),
    data_(0), pitch_(0),
    ext_data_pointer_(false)
  {
    data_ = Allocator::alloc(_width, _height, _depth, &pitch_);
  }

  /** Special constructor.
   *  @param size Size of the Volume
   */
  VolumeCpu(const IuSize& size) :
    Volume(size), data_(0), pitch_(0),
    ext_data_pointer_(false)
  {
    data_ = Allocator::alloc(size.width, size.height, size.depth, &pitch_);
  }

  /** Special constructor.
   *  @param _data Host data pointer
   *  @param _width Width of the Volume
   *  @param _height Height of the Volume
   *  @param _depth Height of the Volume
   *  @param _pitch Distance in bytes between starts of consecutive rows.
   *  @param ext_data_pointer Use external data pointer as internal data pointer
   */
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

  /** Returns the distance in pixels between starts of consecutive rows. */
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
   * @param ox Horizontal offset of the pointer array.
   * @param oy Vertical offset of the pointer array.
   * @return Pointer to the pixel array.
   */
  PixelType* data(int ox = 0, int oy = 0, int oz = 0)
  {
    return &data_[oz*slice_stride() + oy*stride() + ox];
  }

  /** Returns a const pointer to the pixel data.
   * The pointer can be offset to position \a (ox/oy).
   * @param ox Horizontal offset of the pointer array.
   * @param oy Vertical offset of the pointer array.
   * @return Const pointer to the pixel array.
   */
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

  /** Returns a thrust pointer that can be used in custom operators
    @return Thrust pointer of the begin of the host memory
   */
  thrust::pointer<PixelType, thrust::host_system_tag> begin(void)
  {
      return thrust::pointer<PixelType, thrust::host_system_tag>(data());
  }

  /** Returns a thrust pointer that can be used in custom operators
    @return Thrust pointer of the end of the host memory
   */
  thrust::pointer<PixelType, thrust::host_system_tag> end(void)
  {
	  return thrust::pointer<PixelType, thrust::host_system_tag>(data()+slice_stride()*depth());
  }

  /** convert to ndarray_ref -- include ndarray/ndarray_iu.h*/
  ndarray_ref<PixelType,3> ref() const;

  /** construct from ndarray_ref  -- include ndarray/ndarray_iu.h*/
  VolumeCpu(const ndarray_ref<PixelType,3> &x);

protected:

private:
  /** Pointer to host buffer. */
  PixelType* data_;
  /** Distance in bytes between starts of consecutive rows. */
  size_t pitch_;
  /** Flag if data pointer is handled outside the volume class. */
  bool ext_data_pointer_;

private:
  /** Private copy constructor. */
  VolumeCpu(const VolumeCpu&);
  /** Private copy assignment operator. */
  VolumeCpu& operator=(const VolumeCpu&);
};

} // namespace iuprivate

#endif // IUCORE_VOLUME_CPU_H
