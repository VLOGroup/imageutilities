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

template<typename, int> class ndarray_ref;

namespace iu {

/**  \brief Linear device memory class.
 *   \ingroup LinearMemory
 */
template<typename PixelType, int Ndim = 1>
class LinearDeviceMemory : public LinearMemory<Ndim>
{
public:
  /** Constructor. */
  LinearDeviceMemory() :
    LinearMemory<Ndim>(),
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
   *  @param numel numel of linear memory
   */
  LinearDeviceMemory(const Size<Ndim>& size) :
    LinearMemory<Ndim>(size),
    data_(0), ext_data_pointer_(false)
  {
    IU_CUDA_SAFE_CALL(cudaMalloc((void**)&data_, this->numel()*sizeof(PixelType)));
    if (data_ == 0) throw std::bad_alloc();
  }

  /** Special constructor.
   *  @param device_data Device data pointer
   *  @param size size of the memory
   *  @param ext_data_pointer Use external data pointer as internal data pointer
   */
  LinearDeviceMemory(PixelType* device_data, const Size<Ndim>& size, bool ext_data_pointer = false) :
    LinearMemory<Ndim>(size),
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
      IU_CUDA_SAFE_CALL(cudaMalloc((void**)&data_, this->numel()*sizeof(PixelType)));
      if (data_ == 0) throw std::bad_alloc();
      IU_CUDA_SAFE_CALL(cudaMemcpy(data_, device_data, this->numel() * sizeof(PixelType), cudaMemcpyHostToDevice));
    }
  }

  /** Returns a pointer to the device buffer.
   * The pointer can be offset to position \a offset.
   * @param offset Offset of the pointer array.
   * @return Pointer to the device buffer.
   */
  PixelType* data(int offset = 0)
  {
    if (offset > (int)this->numel()) throw IuException("offset not in range", __FILE__, __FUNCTION__, __LINE__);
    return &(data_[offset]);
  }

  /** Returns a const pointer to the device buffer.
   * The pointer can be offset to position \a offset.
   * @param offset Offset of the pointer array.
   * @return Const pointer to the device buffer.
   */
  const PixelType* data(int offset = 0) const
  {
    if (offset > (int)this->numel()) throw IuException("offset not in range", __FILE__, __FUNCTION__, __LINE__);
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
      return thrust::device_ptr<PixelType>(data() + this->numel());
  }

  /** Returns the total amount of bytes saved in the data buffer. */
  virtual size_t bytes() const
  {
    return this->numel()*sizeof(PixelType);
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
   *     if (x < memory.numel_ )
   *     {
   *        img(x) += value;
   *     }
   *  }
   *
   * template<typename PixelType>
   * void doSomethingWithCuda(iu::LinearDeviceMemory<PixelType> *memory, PixelType value)
   * {
   *     dim3 dimBlock(32,1);
   *     dim3 dimGrid(iu::divUp(img->numel(), dimBlock.x), 1);
   *     cudaFunctionKernel<PixelType><<<dimGrid, dimBlock>>>(*memory, value);
   *     IU_CUDA_CHECK;
   * }
   * @endcode
   */
  struct KernelData
  {
      /** Pointer to device buffer. */
      PixelType* data_;

      /** numel of the memory.*/
      int numel_;

      /** size of the memory (on device) */
      int* size_;

      /** Access the memory via the () operator.
       * @param idx Index to access.
       * @return value at index.
       */
      __device__ PixelType& operator()(int idx0, int idx1=0, int idx2=0, int idx3=0, int idx4=0)
      {
          int idx = idx0;
          if(Ndim > 1)
            idx += size_[0]*idx1;
          if(Ndim > 2)
            idx += size_[0]*size_[1]*idx2;
          if(Ndim > 3)
            idx += size_[0]*size_[1]*size_[2]*idx3;
          if(Ndim > 4)
            idx += size_[0]*size_[1]*size_[2]*size_[3]*idx4;
          return data_[idx];
      }

      /** Constructor */
      __host__ KernelData(const LinearDeviceMemory<PixelType, Ndim> &mem)
          : data_(const_cast<PixelType*>(mem.data())), numel_(mem.numel())
      {
        IU_CUDA_SAFE_CALL(cudaMalloc((void**)&size_, Ndim*sizeof(int)));
        IU_CUDA_SAFE_CALL(cudaMemcpy(size_, mem.size().ptr(), Ndim * sizeof(int), cudaMemcpyHostToDevice));
      }

      /** Destructor */
      __host__ ~KernelData()
      {
        IU_CUDA_SAFE_CALL(cudaFree(size_));
        size_ = 0;
      }
  };

  /** convert to ndarray_ref -- include ndarray/ndarray_iu.h*/
  ndarray_ref<PixelType,1> ref() const;

  /** construct from ndarray_ref  -- include ndarray/ndarray_iu.h*/
  LinearDeviceMemory(const ndarray_ref<PixelType,1> &x);

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

} // namespace iu

#endif // LINEARDEVICEMEMORY_H
