#pragma once

#include "image.h"
#include "image_allocator_gpu.h"
#include <thrust/device_ptr.h>

template<typename type, int dims> class ndarray_ref;

namespace iu {

template<typename PixelType, class Allocator>
/** \brief Device 2D image class (pitched memory).
 *  \ingroup Image
 */
class ImageGpu : public Image
{
public:
  /** Define the current pixel type. */
  typedef PixelType pixel_type;

  /** Constructor. */
  ImageGpu() :
    Image(),
    data_(0), pitch_(0), ext_data_pointer_(false), texture_(0)
  {
  }

  /** Destructor. */
  virtual ~ImageGpu()
  {
    if(!ext_data_pointer_)
    {
      // do not delete externally handeled data pointers.
      Allocator::free(data_);
      data_ = 0;
    }
    pitch_ = 0;

    if (texture_)
      IU_CUDA_SAFE_CALL(cudaDestroyTextureObject(texture_));
  }

  /** Special constructor.
   *  @param _width Width of the Image
   *  @param _height Height of the Image
   */
  ImageGpu(unsigned int _width, unsigned int _height) :
      Image(_width, _height), data_(0), pitch_(0),
      ext_data_pointer_(false), texture_(0)
  {
    data_ = Allocator::alloc(this->size(), &pitch_);
  }

  /** Special constructor.
   *  @param size Size of the Image
   */
  ImageGpu(const iu::Size<2>& size) :
      Image(size), data_(0), pitch_(0),
      ext_data_pointer_(false), texture_(0)
  {
    data_ = Allocator::alloc(size, &pitch_);
  }

  /** Special constructor.
   *  @param _data Device data pointer
   *  @param _width Width of the Image
   *  @param _height Height of the Image
   *  @param _pitch Distance in bytes between starts of consecutive rows.
   *  @param ext_data_pointer Use external data pointer as internal data pointer
   */
  ImageGpu(PixelType* _data, unsigned int _width, unsigned int _height,
           size_t _pitch, bool ext_data_pointer = false) :
      Image(_width, _height), data_(0), pitch_(0), ext_data_pointer_(ext_data_pointer), texture_(0)
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

  /** Get Pixel value at position x,y. */
  PixelType getPixel(unsigned int x, unsigned int y)
  {
    PixelType value;
    IU_CUDA_SAFE_CALL(cudaMemcpy2D(&value, sizeof(PixelType), &data_[y*stride()+x], pitch_,
                 sizeof(PixelType), 1, cudaMemcpyDeviceToHost));
    return value;
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

  /** Returns the distance in pixels between starts of consecutive rows. 
   * @todo: returned value is invalid if sizeof(PixelType) == 3, e.g. for *_C3 images. Use
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
      return thrust::device_ptr<PixelType>(data()+stride()*height());
  }

  /** Prepare CUDA texture
   * @param readMode CUDA texture read mode
   * @param filterMode CUDA texture filter mode
   * @param addressMode CUDA texture address mode
   */
  void prepareTexture(cudaTextureReadMode readMode = cudaReadModeElementType,
                      cudaTextureFilterMode filterMode = cudaFilterModeLinear,
                      cudaTextureAddressMode addressMode = cudaAddressModeClamp)
  {
      if (texture_)             // delete if already exists
        IU_CUDA_SAFE_CALL(cudaDestroyTextureObject(texture_));

      cudaResourceDesc resDesc;
      memset(&resDesc, 0, sizeof(resDesc));

      cudaTextureDesc texDesc;
      memset(&texDesc, 0, sizeof(texDesc));

      resDesc.resType = cudaResourceTypePitch2D;
      resDesc.res.pitch2D.devPtr = data();
      resDesc.res.pitch2D.pitchInBytes = pitch();
      resDesc.res.pitch2D.width = width();
      resDesc.res.pitch2D.height = height();
      resDesc.res.pitch2D.desc = cudaCreateChannelDesc<PixelType>();

      texDesc.readMode = readMode;
      texDesc.normalizedCoords = (addressMode == cudaAddressModeClamp) ? false : true;
      texDesc.addressMode[0] = addressMode;
      texDesc.addressMode[1] = addressMode;
      texDesc.filterMode = filterMode;

      IU_CUDA_SAFE_CALL(cudaCreateTextureObject(&texture_, &resDesc, &texDesc, NULL));
  }

  /** Get CUDA texture object */
  cudaTextureObject_t getTexture()
  {
      if (!texture_)        // create texture object implicitly
          this->prepareTexture();

      return texture_;
  }

  /** Get CUDA texture object
   * \note Use case: ImageGPU is passed as const into a function and the method
   * getTexture() is used as a parameter in a cuda kernel call. In that case the
   * user has to explicitly call prepareTexture() in advance such that the texture
   * object exists, because an implicit call to prepareTexture() here would
   * obviously violate method constness.
   */
  inline cudaTextureObject_t getTexture() const
  {
      if (!texture_)
          throw IuException("Warning: getTexture() on const image requires explicit call to prepareTexture(),"
                            " returned cudaTextureObject will be invalid\n", __FILE__, __FUNCTION__, __LINE__);

      return texture_;
  }

  /** \brief Struct pointer KernelData that can be used in CUDA kernels.
   *
   *  This struct provides the device data pointer as well as important class
   *  properties.
   *  @code
   *  template<typename PixelType, class Allocator>
   *  __global__ void cudaFunctionKernel(ImageGpu<PixelType, Allocator>::KernelData img, PixelType value)
   *  {
   *     const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
   *     const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
   *
   *     if (x < img.width_ && y < img.height_)
   *     {
   *        img(x, y) += value;
   *     }
   *  }
   *
   * template<typename PixelType, class Allocator>
   * void doSomethingWithCuda(iu::ImageGpu<PixelType, Allocator> *img, PixelType value)
   * {
   *     dim3 dimBlock(32,32);
   *     dim3 dimGrid(iu::divUp(img->width(), dimBlock.x),
   *                  iu::divUp(img->height(), dimBlock.y));
   *     cudaFunctionKernel<PixelType, Allocator><<<dimGrid, dimBlock>>>(*img, value);
   *     IU_CUDA_CHECK;
   * }
   * @endcode
   */
  struct KernelData
  {
      /** Pointer to device buffer. */
      PixelType* data_;
      /** Width of the Image. */
      int width_;
      /** Height of the Image. */
      int height_;
      /** Distance in pixels between starts of consecutive rows. */
      int stride_;

      /** Access the image via the () operator.
       * @param x Position in x.
       * @param y Position in y.
       * @return value at position (x,y).
       */
      __device__ PixelType& operator()(int x, int y)
      {
          return data_[y*stride_ + x];
      }

      /** Constructor */
      __host__ KernelData(const ImageGpu<PixelType, Allocator> &im)
          : data_(const_cast<PixelType*>(im.data())), width_(im.width()), height_(im.height()),
            stride_(im.stride())
      { }
  };

  /** convert to ndarray_ref -- include ndarray/ndarray_iu.h*/
  ndarray_ref<PixelType,2> ref() const;

  /** construct from ndarray_ref  -- include ndarray/ndarray_iu.h*/
   ImageGpu(const ndarray_ref<PixelType,2> &x);

protected:
  /** Pointer to device buffer. */
  PixelType* data_;
  /** Distance in bytes between starts of consecutive rows. */
  size_t pitch_;
  /** Flag if data pointer is handled outside the image class. */
  bool ext_data_pointer_;
  /** CUDA texture object */
  cudaTextureObject_t texture_;

private:
  /** Private copy constructor. */
  ImageGpu(const ImageGpu&);
  /** Private copy assignment operator. */
  ImageGpu& operator=(const ImageGpu&);
};

} // namespace iu


