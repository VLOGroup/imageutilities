
#ifndef IUCORE_LINEARDEVICEMEMORY_H
#define IUCORE_LINEARDEVICEMEMORY_H

#include <cuda_runtime_api.h>
#include "linearmemory.h"
#include <thrust/device_ptr.h>
#include "../iucutil.h"
#include <type_traits>

template<typename, int> class ndarray_ref;

namespace iu {

/**  \brief Linear device memory class.
 *   \ingroup LinearMemory
 */
template<typename PixelType, unsigned int Ndim>
class LinearDeviceMemory: public LinearMemory<Ndim>
{
public:
  /** Constructor. */
  LinearDeviceMemory() :
      LinearMemory<Ndim>(), data_(0), ext_data_pointer_(false)
  {
  }

  /** Destructor. */
  virtual ~LinearDeviceMemory()
  {
    if ((!ext_data_pointer_) && (data_ != NULL))
    {
      IU_CUDA_SAFE_CALL(cudaFree(data_));
      data_ = 0;
    }
  }

  /** Special constructor.
   *  @param size size of linear memory
   */
  LinearDeviceMemory(const Size<Ndim>& size) :
      LinearMemory<Ndim>(size), data_(0), ext_data_pointer_(false)
  {
    IU_CUDA_SAFE_CALL(
        cudaMalloc((void** )&data_, this->numel() * sizeof(PixelType)));
    if (data_ == 0)
      throw std::bad_alloc();
  }

  /** Special constructor.
   *  @param numel Number of elements of linear memory
   */
  LinearDeviceMemory(const unsigned int& numel) :
      LinearMemory<Ndim>(numel), data_(0), ext_data_pointer_(false)
  {
    IU_CUDA_SAFE_CALL(
        cudaMalloc((void** )&data_, this->numel() * sizeof(PixelType)));
    if (data_ == 0)
      throw std::bad_alloc();
  }

  /** Special constructor.
   *  @param device_data Device data pointer
   *  @param size size of the memory
   *  @param ext_data_pointer Use external data pointer as internal data pointer
   */
  LinearDeviceMemory(PixelType* device_data, const Size<Ndim>& size,
                     bool ext_data_pointer = false) :
      LinearMemory<Ndim>(size), data_(0), ext_data_pointer_(ext_data_pointer)
  {
    if (device_data == 0)
      throw IuException("input data not valid", __FILE__, __FUNCTION__,
                        __LINE__);
    if (ext_data_pointer_)
    {
      // This uses the external data pointer as internal data pointer.
      data_ = device_data;
    }
    else
    {
      // allocates an internal data pointer and copies the external data onto it.
      IU_CUDA_SAFE_CALL(
          cudaMalloc((void** )&data_, this->numel() * sizeof(PixelType)));
      if (data_ == 0)
        throw std::bad_alloc();
      IU_CUDA_SAFE_CALL(
          cudaMemcpy(data_, device_data, this->numel() * sizeof(PixelType),
                     cudaMemcpyHostToDevice));
    }
  }

  /** Special constructor.
   *  @param device_data Device data pointer
   *  @param numel Number of elements of the memory
   *  @param ext_data_pointer Use external data pointer as internal data pointer
   */
  LinearDeviceMemory(PixelType* device_data, const unsigned int& numel,
                     bool ext_data_pointer = false) :
      LinearMemory<Ndim>(numel), data_(0), ext_data_pointer_(ext_data_pointer)
  {
    if (device_data == 0)
      throw IuException("input data not valid", __FILE__, __FUNCTION__,
                        __LINE__);
    if (ext_data_pointer_)
    {
      // This uses the external data pointer as internal data pointer.
      data_ = device_data;
    }
    else
    {
      // allocates an internal data pointer and copies the external data onto it.
      IU_CUDA_SAFE_CALL(
          cudaMalloc((void** )&data_, this->numel() * sizeof(PixelType)));
      if (data_ == 0)
        throw std::bad_alloc();
      IU_CUDA_SAFE_CALL(
          cudaMemcpy(data_, device_data, this->numel() * sizeof(PixelType),
                     cudaMemcpyHostToDevice));
    }
  }

  /** Returns a pointer to the device buffer.
   * The pointer can be offset to position \a offset.
   * @param offset Offset of the pointer array.
   * @return Pointer to the device buffer.
   */
  PixelType* data(unsigned int offset = 0)
  {
    if (offset >= this->numel())
      throw IuException("offset not in range", __FILE__, __FUNCTION__,
                        __LINE__);
    return &(data_[offset]);
  }

  /** Returns a const pointer to the device buffer.
   * The pointer can be offset to position \a offset.
   * @param offset Offset of the pointer array.
   * @return Const pointer to the device buffer.
   */
  const PixelType* data(unsigned int offset = 0) const
  {
    if (offset >= this->numel())
      throw IuException("offset not in range", __FILE__, __FUNCTION__,
                        __LINE__);
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
    return this->numel() * sizeof(PixelType);
  }

  /** Returns the bit depth of the data pointer. */
  virtual unsigned int bitDepth() const
  {
    return 8 * sizeof(PixelType);
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

    /** stride of the memory (on device) */
    int* stride_;

    /** Constructor */
    __host__ KernelData(const LinearDeviceMemory<PixelType, Ndim> &mem) :
        data_(const_cast<PixelType*>(mem.data())), numel_(mem.numel())
    {
      IU_CUDA_SAFE_CALL(cudaMalloc((void** )&size_, Ndim * sizeof(unsigned int)));
      IU_CUDA_SAFE_CALL(
          cudaMemcpy(size_, mem.size().ptr(), Ndim * sizeof(unsigned int),
                     cudaMemcpyHostToDevice));
      IU_CUDA_SAFE_CALL(cudaMalloc((void** )&stride_, Ndim * sizeof(unsigned int)));
      IU_CUDA_SAFE_CALL(
          cudaMemcpy(stride_, mem.stride().ptr(), Ndim * sizeof(unsigned int),
                     cudaMemcpyHostToDevice));
    }

    /** Destructor */
    __host__ ~KernelData()
    {
      IU_CUDA_SAFE_CALL(cudaFree(size_));
      size_ = 0;
      IU_CUDA_SAFE_CALL(cudaFree(stride_));
      stride_ = 0;
    }

    /** Get pixel position for a linear index
     * @param[in] idx0 Position at dimension 0
     * @param[in] idx1 Position at dimension 1
     * @param[out] linear_idx Linear index
     */
    template<typename ResultType = void>
    __device__ typename std::enable_if<(Ndim == 2), ResultType>::type getPosition(
        const unsigned int& linear_idx, unsigned int& idx0, unsigned int& idx1)
    {
      idx1 = linear_idx / stride_[1];
      idx0 = linear_idx % stride_[1];
    }

    /** Get pixel position for a linear index
     * @param[in] idx0 Position at dimension 0
     * @param[in] idx1 Position at dimension 1
     * @param[in] idx2 Position at dimension 2
     * @param[out] linear_idx Linear index
     */
    template<typename ResultType = void>
    __device__ typename std::enable_if<(Ndim == 3), ResultType>::type getPosition(
        const unsigned int& linear_idx, unsigned int& idx0, unsigned int& idx1,
        unsigned int& idx2)
    {
      idx2 = linear_idx / stride_[2];
      idx1 = (linear_idx % stride_[2]) / stride_[1];
      idx0 = (linear_idx % stride_[2]) % stride_[1];
    }

    /** Get pixel position for a linear index
     * @param[in] idx0 Position at dimension 0
     * @param[in] idx1 Position at dimension 1
     * @param[in] idx2 Position at dimension 2
     * @param[in] idx3 Position at dimension 3
     * @param[out] linear_idx Linear index
     */
    template<typename ResultType = void>
    __device__ typename std::enable_if<(Ndim == 4), ResultType>::type getPosition(
        const unsigned int& linear_idx, unsigned int& idx0, unsigned int& idx1,
        unsigned int& idx2, unsigned int& idx3)
    {
      idx3 = linear_idx / stride_[3];
      idx2 = (linear_idx % stride_[3]) / stride_[2];
      idx1 = ((linear_idx % stride_[3]) % stride_[2]) / stride_[1];
      idx0 = ((linear_idx % stride_[3]) % stride_[2]) % stride_[1];
    }

    /** Get pixel position for a linear index
     * @param[in] idx0 Position at dimension 0
     * @param[in] idx1 Position at dimension 1
     * @param[in] idx2 Position at dimension 2
     * @param[in] idx3 Position at dimension 3
     * @param[in] idx4 Position at dimension 4
     * @param[out] linear_idx Linear index
     */
    template<typename ResultType = void>
    __device__ typename std::enable_if<(Ndim == 5), ResultType>::type getPosition(
        const unsigned int& linear_idx, unsigned int& idx0, unsigned int& idx1,
        unsigned int& idx2, unsigned int& idx3, unsigned int& idx4)
    {
      idx4 = linear_idx / stride_[4];
      idx3 = (linear_idx % stride_[4]) / stride_[3];
      idx2= ((linear_idx % stride_[4]) % stride_[3]) / stride_[2];
      idx1 = (((linear_idx % stride_[4]) % stride_[3]) % stride_[2]) / stride_[1];
      idx0 = (((linear_idx % stride_[4]) % stride_[3]) % stride_[2]) % stride_[1];
    }

    /** Convert pixel position to linear index
     * @param idx0 Position at dimension 0
     * @param idx1 Position at dimension 1
     * @return Linear index
     */
    template<typename ResultType = unsigned int>
    __device__ typename std::enable_if<(Ndim > 1), ResultType>::type getLinearIndex(
        const unsigned int& idx0, const unsigned int& idx1)
    {
      unsigned int linear_idx = idx0;
      linear_idx += stride_[1] * idx1;
      return linear_idx;
    }

    /** Convert pixel position to linear index
     * @param idx0 Position at dimension 0
     * @param idx1 Position at dimension 1
     * @param idx2 Position at dimension 2
     * @return Linear index
     */
    template<typename ResultType = unsigned int>
    __device__ typename std::enable_if<(Ndim > 2), ResultType>::type getLinearIndex(
        const unsigned int& idx0, const unsigned int& idx1, const unsigned int& idx2)
    {
      unsigned int linear_idx = idx0;
      linear_idx += stride_[1] * idx1;
      linear_idx += stride_[2] * idx2;
      return linear_idx;
    }

    /** Convert pixel position to linear index
     * @param idx0 Position at dimension 0
     * @param idx1 Position at dimension 1
     * @param idx2 Position at dimension 2
     * @param idx3 Position at dimension 3
     * @return Linear index
     */
    template<typename ResultType = unsigned int>
    __device__ typename std::enable_if<(Ndim > 3), ResultType>::type getLinearIndex(
        const unsigned int& idx0, const unsigned int& idx1, const unsigned int& idx2, const unsigned int& idx3)
    {
      unsigned int linear_idx = idx0;
      linear_idx += stride_[1] * idx1;
      linear_idx += stride_[2] * idx2;
      linear_idx += stride_[3] * idx3;
      return linear_idx;
    }

    /** Convert pixel position to linear index
     * @param idx0 Position at dimension 0
     * @param idx1 Position at dimension 1
     * @param idx2 Position at dimension 2
     * @param idx3 Position at dimension 3
     * @param idx4 Position at dimension 4
     * @return Linear index
     */
    template<typename ResultType = unsigned int>
    __device__ typename std::enable_if<(Ndim > 4), ResultType>::type getLinearIndex(
        const unsigned int& idx0, const unsigned int& idx1, const unsigned int& idx2, const unsigned int& idx3, const unsigned int& idx4)
    {
      unsigned int linear_idx = idx0;
      linear_idx += stride_[1] * idx1;
      linear_idx += stride_[2] * idx2;
      linear_idx += stride_[3] * idx3;
      linear_idx += stride_[4] * idx4;
      return linear_idx;
    }

    /** Access the memory via the () operator.
     * @param idx Index to access.
     * @return value at index.
     */
    __device__ PixelType& operator()(const unsigned int& idx)
    {
      return data_[idx];
    }

    /** Access the memory via the () operator.
     * @param idx0 Index at position 0 to access.
     * @param idx1 Index at position 1 to access.
     * @return value at index.
     */
    template<typename ResultType = PixelType>
    __device__ typename std::enable_if<(Ndim > 1), ResultType&>::type operator()(
        const unsigned int& idx0, const unsigned int& idx1)
    {
      return data_[getLinearIndex(idx0, idx1)];
    }

    /** Access the memory via the () operator.
     * @param idx0 Index at position 0 to access.
     * @param idx1 Index at position 1 to access.
     * @param idx2 Index at position 2 to access.
     * @return value at index.
     */
    template<typename ResultType = PixelType>
    __device__ typename std::enable_if<(Ndim > 2), ResultType&>::type operator()(
        const unsigned int& idx0, const unsigned int& idx1, const unsigned int& idx2)
    {
      return data_[getLinearIndex(idx0, idx1, idx2)];
    }

    /** Access the memory via the () operator.
     * @param idx0 Index at position 0 to access.
     * @param idx1 Index at position 1 to access.
     * @param idx2 Index at position 2 to access.
     * @param idx3 Index at position 3 to access.
     * @return value at index.
     */
    template<typename ResultType = PixelType>
    __device__ typename std::enable_if<(Ndim > 3), ResultType&>::type operator()(
        const unsigned int& idx0, const unsigned int& idx1, const unsigned int& idx2, const unsigned int& idx3)
    {
      return data_[getLinearIndex(idx0, idx1, idx2, idx3)];
    }

    /** Access the memory via the () operator.
     * @param idx0 Index at position 0 to access.
     * @param idx1 Index at position 1 to access.
     * @param idx2 Index at position 2 to access.
     * @param idx3 Index at position 3 to access.
     * @param idx4 Index at position 4 to access.
     * @return value at index.
     */
    template<typename ResultType = PixelType>
    __device__ typename std::enable_if<(Ndim > 4), ResultType&>::type operator()(
        const unsigned int& idx0, const unsigned int& idx1, const unsigned int& idx2, const unsigned int& idx3, const unsigned int& idx4)
    {
      return data_[getLinearIndex(idx0, idx1, idx2, idx3, idx4)];
    }
  };

protected:

protected:
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

template<typename PixelType>
class LinearDeviceMemory1d : public LinearDeviceMemory<PixelType, 1>
{
public:
  LinearDeviceMemory1d(const unsigned int& numel) : LinearDeviceMemory<PixelType, 1>(numel) {}

  /** convert to ndarray_ref -- include ndarray/ndarray_iu.h*/
  ndarray_ref<PixelType, 1> ref() const;

  /** construct from ndarray_ref  -- include ndarray/ndarray_iu.h*/
  LinearDeviceMemory1d(const ndarray_ref<PixelType, 1> &x);
};

}  // namespace iu

#endif // LINEARDEVICEMEMORY_H
