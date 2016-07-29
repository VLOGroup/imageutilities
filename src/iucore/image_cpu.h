
#ifndef IMAGE_CPU_H
#define IMAGE_CPU_H

#include <thrust/memory.h>
#include "image.h"
#include "image_allocator_cpu.h"

//#include "ndarray/ndarray_ref.host.h"
template<typename type, int dims> class ndarray_ref;

namespace boost
{
    namespace python
    {
        namespace api
        {
            class object;
        }
    }
}


namespace iu {

template<typename PixelType, class Allocator>
/** \brief Host 2D image class (pitched memory).
 *  \ingroup Image
 */
class ImageCpu : public Image
{
public:
  /** Constructor. */
  ImageCpu() :
    Image(),
    data_(0), pitch_(0), ext_data_pointer_(false)
  {
  }

  /** Destructor. */
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

  /** Special constructor.
   *  @param _width Width of the Image
   *  @param _height Height of the Image
   */
  ImageCpu(unsigned int _width, unsigned int _height) :
    Image(_width, _height), data_(0), pitch_(0),
    ext_data_pointer_(false)
  {
    data_ = Allocator::alloc(_width, _height, &pitch_);
  }

  /** Special constructor.
   *  @param size Size of the Image
   */
  ImageCpu(const IuSize& size) :
    Image(size.width, size.height), data_(0), pitch_(0),
    ext_data_pointer_(false)
  {
    data_ = Allocator::alloc(size.width, size.height, &pitch_);
  }

  /** Special constructor.
   *  @param _data Host data pointer
   *  @param _width Width of the Image
   *  @param _height Height of the Image
   *  @param _pitch Distance in bytes between starts of consecutive rows.
   *  @param ext_data_pointer Use external data pointer as internal data pointer
   */
  ImageCpu(PixelType* _data, unsigned int _width, unsigned int _height,
           size_t _pitch, bool ext_data_pointer = false) :
    Image(_width, _height), data_(0), pitch_(0),
    ext_data_pointer_(ext_data_pointer)
  {
    if(ext_data_pointer_)
    {
      data_ = _data;
      pitch_ = _pitch;
    }
    else
    {
      data_ = Allocator::alloc(width(), height(), &pitch_);
      Allocator::copy(_data, _pitch, data_, pitch_, this->size());
    }
  }

  /** Returns a pointer to the pixel data.
   * The pointer can be offset to position (ox/oy).
   * @param ox Horizontal offset of the pointer array.
   * @param oy Vertical offset of the pointer array.
   * @return Pointer to the pixel array.
   */
  PixelType* data(int ox = 0, int oy = 0)
  {
    return &data_[oy * stride() + ox];
  }

  /** Returns a const pointer to the pixel data.
   * The pointer can be offset to position (ox/oy).
   * @param ox Horizontal offset of the pointer array.
   * @param oy Vertical offset of the pointer array.
   * @return Const pointer to the pixel array.
   */
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

  /** Get Pointer to beginning of row (y index).
   * This enables the usage of [y][x] operator.
   * \param row y index / beginning of row
   */
  PixelType* operator[](unsigned int row)
  {
    return data_+row*stride();
  }

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

  /** Returns the distance in pixels between starts of consecutive rows. */
  virtual size_t stride() const
  {
    return pitch_/sizeof(PixelType);
  }

  /** Returns the bit depth of the data pointer. */
  virtual unsigned int bitDepth() const
  {
    return 8*sizeof(PixelType);
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
      return thrust::pointer<PixelType, thrust::host_system_tag>(data()+stride()*height());
  }

  /** Returns flag if the image data resides on the device/GPU (TRUE) or host/GPU (FALSE) */
  virtual bool onDevice() const
  {
    return false;
  }

  /** convert to ndarray_ref -- include ndarray/ndarray_iu.h*/
  ndarray_ref<PixelType,2> ref() const;

  /** construct from ndarray_ref  -- include ndarray/ndarray_iu.h*/
   ImageCpu(const ndarray_ref<PixelType,2> &x);

   /**
    * ImageCpu constructor from numpy array. It wraps the numpy data pointer
    * i.e. does not perform a deep copy. This means that changes to the ImageCpu are transparent to python.
    * <b>Attention:</b> It performs just a basic check if the datatyps are compatible: If you have a numpy array with
    * uint8 and you try to construct a ImageCpu_32f_C1 from it, it will throw an exception. Constructing a ImageCpu_32u_C1
    * from a numpy float32 array will not give an error (both are 32-bit datatypes)!
    *
    * include iuypthon.h!
    * @param py_arr a boost::python::object representing a numpy array
    */
   ImageCpu(boost::python::api::object& py_arr);

protected:
  /** Pointer to host buffer. */
  PixelType* data_;
  /** Distance in bytes between starts of consecutive rows. */
  size_t pitch_;
  /** Flag if data pointer is handled outside the image class. */
  bool ext_data_pointer_;

private:
  /** Private copy constructor. */
  ImageCpu(const ImageCpu&);
  /** Private copy assignment operator. */
  ImageCpu& operator=(const ImageCpu&);
};

} // namespace iu


#endif // IMAGE_CPU_H
