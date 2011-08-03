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
 * Class       : LinearDeviceMemory
 * Language    : C++
 * Description : Inline implementation of a linear device memory class
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUCORE_LINEARDEVICEMEMORY_H
#define IUCORE_LINEARDEVICEMEMORY_H

#include <cuda_runtime_api.h>
#include "linearmemory.h"

namespace iu {

template<typename PixelType>
class LinearDeviceMemory : public LinearMemory
{
public:
  LinearDeviceMemory() :
    LinearMemory(),
    data_(0), ext_data_pointer_(false)
  {
  }

  virtual ~LinearDeviceMemory()
  {
    if((!ext_data_pointer_) && (data_!=NULL))
    {
      cudaFree(data_);
      data_ = 0;
    }
  }

  LinearDeviceMemory(const unsigned int& length) :
    LinearMemory(length),
    data_(0), ext_data_pointer_(false)
  {
    cudaMalloc((void**)&data_, this->length()*sizeof(PixelType));
    if (data_ == 0) throw std::bad_alloc();
  }

  LinearDeviceMemory(const LinearDeviceMemory<PixelType>& from) :
    LinearMemory(from),
    data_(0), ext_data_pointer_(false)
  {
    if (from.data_==0) throw IuException("input data not valid", __FILE__, __FUNCTION__, __LINE__);
    cudaMalloc((void**)&data_, this->length()*sizeof(PixelType));
    if (data_ == 0) throw std::bad_alloc();
    cudaMemcpy(data_, from.data_, this->length() * sizeof(PixelType), cudaMemcpyDeviceToDevice);
  }

  LinearDeviceMemory(PixelType* device_data, const unsigned int& length, bool ext_data_pointer = false) :
    LinearMemory(length),
    data_(0), ext_data_pointer_(ext_data_pointer)
  {
    if (device_data==0) throw IuException("input data not valid", __FILE__, __FUNCTION__, __LINE__);
    if(ext_data_pointer_)
    {
      // This uses the external data pointer as internal data pointer.
      data_ = device_data;
    }
    else
    {
      // allocates an internal data pointer and copies the external data onto it.
      cudaMalloc((void**)&data_, this->length()*sizeof(PixelType));
      if (data_ == 0) throw std::bad_alloc();
      cudaMemcpy(data_, device_data, this->length() * sizeof(PixelType), cudaMemcpyHostToDevice);
    }
  }

  // :TODO: operator=

  /** Returns a pointer to the device buffer.
   * The pointer can be offset to position \a offset.
   * @param[in] offset Offset of the pointer array.
   * @return Pointer to the device buffer.
   */
  PixelType* data(int offset = 0)
  {
    if (offset > (int)this->length()) throw IuException("offset not in range", __FILE__, __FUNCTION__, __LINE__);
    return &(data_[offset]);
  }

  /** Returns a const pointer to the device buffer.
   * The pointer can be offset to position \a offset.
   * @param[in] offset Offset of the pointer array.
   * @return Const pointer to the device buffer.
   */
  const PixelType* data(int offset = 0) const
  {
    if (offset > (int)this->length()) throw IuException("offset not in range", __FILE__, __FUNCTION__, __LINE__);
    return reinterpret_cast<const PixelType*>(&(data_[offset]));
  }

  /** Returns the total amount of bytes saved in the data buffer. */
  virtual size_t bytes() const
  {
    return this->length()*sizeof(PixelType);
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

protected:


private:
  PixelType* data_; /**< Pointer to device buffer. */
  bool ext_data_pointer_; /**< Flag if data pointer is handled outside the image class. */

};

} // namespace iuprivate

#endif // LINEARDEVICEMEMORY_H
