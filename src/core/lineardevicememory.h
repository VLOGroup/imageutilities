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

namespace iu {

template<typename PixelType>
class LinearDeviceMemory
{
public:
  LinearDeviceMemory() :
      data_(0), length_(0), ext_data_pointer_(false)
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
      data_(0), length_(length), ext_data_pointer_(false)
  {
    cudaMalloc((void**)&data_, length_*sizeof(PixelType));
  }

  LinearDeviceMemory(const LinearDeviceMemory<PixelType>& from) :
      data_(0), length_(from.length_), ext_data_pointer_(false)
  {
    if(from.data_ == NULL)
      return;

    cudaMalloc((void**)&data_, length_*sizeof(PixelType));
    cudaMemcpy(data_, from.data_, length_ * sizeof(PixelType), cudaMemcpyDeviceToDevice);
  }

  LinearDeviceMemory(PixelType* host_data, const unsigned int& length, bool ext_data_pointer = false) :
      data_(0), length_(length), ext_data_pointer_(ext_data_pointer)
  {
    if(ext_data_pointer_)
    {
      // This uses the external data pointer as internal data pointer.
      data_ = host_data;
    }
    else
    {
      // allocates an internal data pointer and copies the external data onto it.
      if(host_data == 0)
        return;

      cudaMalloc((void**)&data_, length_*sizeof(PixelType));
      cudaMemcpy(data_, host_data, length_ * sizeof(PixelType), cudaMemcpyHostToDevice);
    }
  }

  // :TODO: operator=

  /** Returns the number of elements saved in the device buffer. (length of device buffer) */
  unsigned int length() const
  {
    return length_;
  }

  /** Returns a pointer to the device buffer.
   * The pointer can be offset to position \a offset.
   * @param[in] offset Offset of the pointer array.
   * @return Pointer to the device buffer.
   */
  PixelType* data(int offset = 0)
  {
    return &(data_[offset]);
  }

  /** Returns a const pointer to the device buffer.
   * The pointer can be offset to position \a offset.
   * @param[in] offset Offset of the pointer array.
   * @return Const pointer to the device buffer.
   */
  const PixelType* data(int offset = 0) const
  {
    return reinterpret_cast<const PixelType*>(&(data_[offset]));
  }

  /** Returns the total amount of bytes saved in the data buffer. */
  size_t bytes() const
  {
    return length_*sizeof(PixelType);
  }

protected:


private:
  PixelType* data_; /**< Pointer to device buffer. */
  unsigned int length_; /**< Buffer length (number of elements). */
  bool ext_data_pointer_; /**< Flag if data pointer is handled outside the image class. */

};

} // namespace iuprivate

#endif // LINEARDEVICEMEMORY_H
