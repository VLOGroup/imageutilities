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
 * Class       : ImageGpu
 * Language    : C++
 * Description : Definition of image class for Gpu
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUCORE_IMAGE_GPU_H
#define IUCORE_IMAGE_GPU_H

#include "image.h"
#include "image_allocator_gpu.h"
#include <thrust/device_ptr.h>

namespace iu {

template<typename PixelType, class Allocator, IuPixelType _pixel_type>
class ImageGpu : public Image
{
public:
  ImageGpu() :
    Image(_pixel_type),
    data_(0), pitch_(0), ext_data_pointer_(false), texture_(0)
  {
  }

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
        cudaDestroyTextureObject(texture_);
  }

  ImageGpu(unsigned int _width, unsigned int _height) :
      Image(_pixel_type, _width, _height), data_(0), pitch_(0),
      ext_data_pointer_(false), texture_(0)
  {
    data_ = Allocator::alloc(this->size(), &pitch_);
  }

  ImageGpu(const IuSize& size) :
      Image(_pixel_type, size), data_(0), pitch_(0),
      ext_data_pointer_(false), texture_(0)
  {
    data_ = Allocator::alloc(size, &pitch_);
  }

  ImageGpu(const ImageGpu<PixelType, Allocator, _pixel_type>& from) :
      Image(from), data_(0), pitch_(0),
      ext_data_pointer_(false), texture_(0)
  {
      if (from.ext_data_pointer_)  // external image stays external when copied
      {
          data_ = from.data_;
          pitch_ = from.pitch_;
          ext_data_pointer_ = from.ext_data_pointer_;
      }
      else
      {
          data_ = Allocator::alloc(from.size(), &pitch_);
          Allocator::copy(from.data(), from.pitch(), data_, pitch_, this->size());
      }
  }

  ImageGpu(PixelType* _data, unsigned int _width, unsigned int _height,
           size_t _pitch, bool ext_data_pointer = false) :
      Image(_pixel_type, _width, _height), data_(0), pitch_(0), ext_data_pointer_(ext_data_pointer), texture_(0)
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

	ImageGpu& operator= (const ImageGpu<PixelType, Allocator, _pixel_type> &from)
	{
		this->size_ = from.size();
//		this->roi_ = from.roi();
		this->pixel_type_ = from.pixelType();
		this->data_ = Allocator::alloc(from.size(), &(this->pitch_));
		cudaMemcpy2D(data_, pitch_, from.data(), from.pitch(), from.width(), from.height(), cudaMemcpyDeviceToDevice);
		// pitch_ = from.pitch(); // handled by allocator
		this->ext_data_pointer_ = false; 
        this->texture_ = 0;
		return *this;
	}

  PixelType getPixel(unsigned int x, unsigned int y)
  {
    PixelType value;
    cudaMemcpy2D(&value, sizeof(PixelType), &data_[y*stride()+x], pitch_,
                 sizeof(PixelType), 1, cudaMemcpyDeviceToHost);
    return value;
  }

  // :TODO:
  //ImageGpu& operator= (const ImageGpu<PixelType, Allocator>& from);

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
   * The pointer can be offset to position \a (ox/oy).
   * @param[in] ox Horizontal offset of the pointer array.
   * @param[in] oy Vertical offset of the pointer array.
   * @return Pointer to the pixel array.
   */
  PixelType* data(int ox = 0, int oy = 0)
  {
    return &data_[oy * stride() + ox];
  }
  const PixelType* data(int ox = 0, int oy = 0) const
  {
    return reinterpret_cast<const PixelType*>(
        &data_[oy * stride() + ox]);
  }

  thrust::device_ptr<PixelType> begin(void)
  {
      return thrust::device_ptr<PixelType>(data());
  }

  thrust::device_ptr<PixelType> end(void)
  {
      return thrust::device_ptr<PixelType>(data()+stride()*height());
  }


  void prepareTexture(cudaTextureReadMode readMode = cudaReadModeElementType,
                      cudaTextureFilterMode filterMode = cudaFilterModeLinear,
                      cudaTextureAddressMode addressMode = cudaAddressModeClamp)
  {
      if (texture_)             // delete if already exists
          cudaDestroyTextureObject(texture_);

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

      cudaCreateTextureObject(&texture_, &resDesc, &texDesc, NULL);
  }

  cudaTextureObject_t getTexture()
  {
      if (!texture_)        // create texture object implicitly
          this->prepareTexture();

      return texture_;
  }

  // Use case: ImageGPU is passed as const into a function and the method getTexture() is used as
  // a parameter in a cuda kernel call. In that case the user has to explicitly call prepareTexture()
  // in advance such that the texture object exists, because an implicit call to prepareTexture()
  // here would obviously violate method constness
  cudaTextureObject_t getTexture() const
  {
      if (!texture_)
          throw IuException("Warning: getTexture() on const image requires explicit call to prepareTexture(),"
                            " returned cudaTextureObject will be invalid\n", __FILE__, __FUNCTION__, __LINE__);

      return texture_;
  }

  struct KernelData
  {
      PixelType* data_;
      int width_;
      int height_;
      int stride_;

      __device__ PixelType& operator()(int x, int y)
      {
          return data_[y*stride_ + x];
      }



      __host__ KernelData(const ImageGpu<PixelType, Allocator, _pixel_type> &im)
          : data_(const_cast<PixelType*>(im.data())), width_(im.width()), height_(im.height()),
            stride_(im.stride())
      { }
  };

protected:
  PixelType* data_;
  size_t pitch_;
  bool ext_data_pointer_; /**< Flag if data pointer is handled outside the image class. */

  cudaTextureObject_t texture_;
};

} // namespace iuprivate

#endif // IUCORE_IMAGE_GPU_H

