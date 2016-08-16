#pragma once

#include "volume.h"
#include "volume_allocator_gpu.h"
#include <thrust/device_ptr.h>

template<typename, int> class ndarray_ref;

namespace iu {

template<typename PixelType, class Allocator>
/** \brief Device 3D volume class (pitched memory).
 *  \ingroup Volume
 */
class VolumeGpu : public Volume
{
public:
  /** Constructor. */
  VolumeGpu() :
    Volume(),
    data_(0), pitch_(0), ext_data_pointer_(false)
  {
  }

  /** Destructor. */
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

  /** Special constructor.
   *  @param _width Width of the Volume
   *  @param _height Height of the Volume
   *  @param _depth Depth of the Volume
   */
  VolumeGpu(unsigned int _width, unsigned int _height, unsigned int _depth) :
    Volume(_width, _height, _depth), data_(0), pitch_(0),
    ext_data_pointer_(false)
  {
    data_ = Allocator::alloc(this->size(), &pitch_);
  }

  /** Special constructor.
   *  @param size Size of the Volume
   */
  VolumeGpu(const iu::Size<3>& size) :
    Volume(size), data_(0), pitch_(0),
    ext_data_pointer_(false)
  {
    data_ = Allocator::alloc(this->size(), &pitch_);
  }

  /** Special constructor.
   *  @param _data Device data pointer
   *  @param _width Width of the Volume
   *  @param _height Height of the Volume
   *  @param _depth Height of the Volume
   *  @param _pitch Distance in bytes between starts of consecutive rows.
   *  @param ext_data_pointer Use external data pointer as internal data pointer
   */
  VolumeGpu(PixelType* _data, unsigned int _width, unsigned int _height, unsigned int _depth,
            size_t _pitch, bool ext_data_pointer = false) :
    Volume(_width, _height, _depth), data_(0), pitch_(0),
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

  /** Returns the distance in pixels between starts of consecutive slices. */
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

  /** Returns a volume slice given by a z-offset as ImageGpu
    * @param[in] oz z-offset into the volume
    * @return volume slice at depth z as ImageGpu. Note, the ImageGpu merely holds a pointer to the volume data at depth z,
    * i.e. it does not manage its own data -> changes to the Image are transparent to the volume and vice versa.
    */
  ImageGpu<PixelType, iuprivate::ImageAllocatorGpu<PixelType> > getSlice(int oz)
  {
    return ImageGpu<PixelType, iuprivate::ImageAllocatorGpu<PixelType> >(
          &data_[oz*slice_stride()], width(), height(), pitch_, true);
  }

  /** Returns a thrust pointer that can be used in custom operators
    @return Thrust pointer of the begin of the device memory
   */
  thrust::device_ptr<PixelType> begin(void)
  {
      return thrust::device_ptr<PixelType>(data());
  }

  /** Returns a thrust pointer that can be used in custom operators
    @return Thrust pointer of the end of the device memory
   */
  thrust::device_ptr<PixelType> end(void)
  {
      return thrust::device_ptr<PixelType>(data()+slice_stride()*depth());
  }

  /** \todo cuda texture for VolumeGpu */
  
  /** \brief Struct pointer KernelData that can be used in CUDA kernels
   *
   *  This struct provides the device data pointer as well as important class
   *  properties.
   *  @code
   *  template<typename PixelType, class Allocator>
   *  __global__ void cudaFunctionKernel(VolumeGpu<PixelType, Allocator>::KernelData img, PixelType value)
   *  {
   *     const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
   *     const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
   *     const unsigned int z = threadIdx.z + blockIdx.z * blockDim.z;
   *
   *     if (x < img.width_ && y < img.height_ && z < img.depth_)
   *     {
   *        img(x, y, z) += value;
   *     }
   *  }
   *
   * template<typename PixelType, class Allocator>
   * void doSomethingWithCuda(iu::VolumeGpu<PixelType, Allocator> *img, PixelType value)
   * {
   *     dim3 dimBlock(8,8,4);
   *     dim3 dimGrid(iu::divUp(img->width(), dimBlock.x),
   *                  iu::divUp(img->height(), dimBlock.y),
   *                  iu::divUp(img->depth(), dimBlock.z));
   *     cudaFunctionKernel<PixelType, Allocator><<<dimGrid, dimBlock>>>(*img, value);
   *     IU_CUDA_CHECK;
   * }
   * @endcode
   */
  struct KernelData
  {
      /** Pointer to device buffer. */
      PixelType* data_;
      /** Width of Volume. */
      int width_;
      /** Height of Volume. */
      int height_;
      /** Depth of Volume. */
      int depth_;
      /** Distance in pixels between starts of consecutive rows. */
      int stride_;

      /** Access the image via the () operator.
       * @param x Position in x.
       * @param y Position in y.
       * @param z Position in z.
       * @return value at position (x,y,z).
       */
      __device__ PixelType& operator()(int x, int y, int z)
      {
          return data_[z*height_*stride_ + y*stride_ + x];
      }

      /** Constructor. */
      __host__ KernelData(const VolumeGpu<PixelType, Allocator> &vol)
          : data_(const_cast<PixelType*>(vol.data())), width_(vol.width()), height_(vol.height()),
            depth_(vol.depth()), stride_(vol.stride())
      { }
  };

  /** convert to ndarray_ref -- include ndarray/ndarray_iu.h*/
  ndarray_ref<PixelType,3> ref() const;

  /** construct from ndarray_ref  -- include ndarray/ndarray_iu.h*/
  VolumeGpu(const ndarray_ref<PixelType,3> &x);

protected:

private:
  /** Pointer to device buffer. */
  PixelType* data_;
  /** Distance in bytes between starts of consecutive rows. */
  size_t pitch_;
  /** Flag if data pointer is handled outside the volume class. */
  bool ext_data_pointer_;

private:
  /** Private copy constructor. */
  VolumeGpu(const VolumeGpu&);
  /** Private copy assignment operator. */
  VolumeGpu& operator=(const VolumeGpu&);
};

} // namespace iu


