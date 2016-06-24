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
#include <thrust/device_ptr.h>
#include "../iucutil.h"

namespace iu {

template<typename PixelType>
/** \addtogroup LinearMemory
 *  \{
 */

/**  \brief Linear device memory class.
 */
class LinearDeviceMemory : public LinearMemory
{
public:
  /** Constructor. */
  LinearDeviceMemory() :
    LinearMemory(),
    data_(0), ext_data_pointer_(false)
  {
  }

  /** Destructor. */
  virtual ~LinearDeviceMemory()
  {
    if((!ext_data_pointer_) && (data_!=NULL))
    {
      IU_CUDA_SAFE_CALL(cudaFree(data_));
      data_ = 0;
    }
  }

  /** Special constructor.
   *  @param length Length of linear memory
   */
  LinearDeviceMemory(const unsigned int& length) :
    LinearMemory(length),
    data_(0), ext_data_pointer_(false)
  {
    IU_CUDA_SAFE_CALL(cudaMalloc((void**)&data_, this->length()*sizeof(PixelType)));
    if (data_ == 0) throw std::bad_alloc();
  }

  /** Special constructor.
   *  @param device_data Device data pointer
   *  @param length Length of the memory
   *  @param ext_data_pointer Use external data pointer as internal data pointer
   */
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
      IU_CUDA_SAFE_CALL(cudaMalloc((void**)&data_, this->length()*sizeof(PixelType)));
      if (data_ == 0) throw std::bad_alloc();
      IU_CUDA_SAFE_CALL(cudaMemcpy(data_, device_data, this->length() * sizeof(PixelType), cudaMemcpyHostToDevice));
    }
  }

  /** Returns a pointer to the device buffer.
   * The pointer can be offset to position \a offset.
   * @param offset Offset of the pointer array.
   * @return Pointer to the device buffer.
   */
  PixelType* data(int offset = 0)
  {
    if (offset > (int)this->length()) throw IuException("offset not in range", __FILE__, __FUNCTION__, __LINE__);
    return &(data_[offset]);
  }

  /** Returns a const pointer to the device buffer.
   * The pointer can be offset to position \a offset.
   * @param offset Offset of the pointer array.
   * @return Const pointer to the device buffer.
   */
  const PixelType* data(int offset = 0) const
  {
    if (offset > (int)this->length()) throw IuException("offset not in range", __FILE__, __FUNCTION__, __LINE__);
    return reinterpret_cast<const PixelType*>(&(data_[offset]));
  }

  /** Returns a thrust device pointer that can be used in custom operators
    @return Thrust pointer of the begin of the memory
    */
  thrust::device_ptr<PixelType> begin(void)
  {
      return thrust::device_ptr<PixelType>(data());
  }

  /** Returns a thrust device pointer that can be used in custom operators
    @return Thrust pointer of the end of the memory
    */
  thrust::device_ptr<PixelType> end(void)
  {
      return thrust::device_ptr<PixelType>(data() + length());
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

  /** \brief Struct pointer KernelData that can be used in CUDA kernels.
   *
   *  This struct provides the device data pointer as well as important class
   *  properties.
   *  @code
   *  template<typename PixelType>
   *  __global__ void cudaFunctionKernel(iu::LinearDeviceMemory<PixelType>::KernelData memory, PixelType value)
   *  {
   *     const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
   *
   *     if (x < memory.length_ )
   *     {
   *        img(x) += value;
   *     }
   *  }
   *
   * template<typename PixelType>
   * void doSomethingWithCuda(iu::LinearDeviceMemory<PixelType> *memory, PixelType value)
   * {
   *     dim3 dimBlock(32,1);
   *     dim3 dimGrid(iu::divUp(img->length(), dimBlock.x), 1);
   *     cudaFunctionKernel<PixelType><<<dimGrid, dimBlock>>>(*memory, value);
   *     IU_CUDA_CHECK;
   * }
   * @endcode
   */
  struct KernelData
  {
      /** Pointer to device buffer. */
      PixelType* data_;

      /** Length of the memory.*/
      int length_;

      /** Access the memory via the () operator.
       * @param idx Index to access.
       * @return value at index.
       */
      __device__ PixelType& operator()(int idx)
      {
          return data_[idx];
      }

      /** Constructor */
      __host__ KernelData(const LinearDeviceMemory<PixelType> &mem)
          : data_(const_cast<PixelType*>(mem.data())), length_(mem.length())
      { }
  };

protected:


private:
  /** Pointer to device buffer. */
  PixelType* data_;
  /** Flag if data pointer is handled outside the LinearDeviceMemory class. */
  bool ext_data_pointer_;

private:
  /** Private copy constructor. */
  LinearDeviceMemory(const LinearDeviceMemory&);
  /** Private copy assignment operator. */
  LinearDeviceMemory& operator=(const LinearDeviceMemory&);
};

/** \} */ // end of Linear Memory

} // namespace iu

#endif // LINEARDEVICEMEMORY_H
