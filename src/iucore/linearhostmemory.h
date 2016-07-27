
#ifndef IUCORE_LINEARHOSTMEMORY_H
#define IUCORE_LINEARHOSTMEMORY_H

#include <stdio.h>
#include <assert.h>
#include <cstdlib>
#include <string.h>         // memcpy
#include <thrust/memory.h>
#include <type_traits>

#include "linearmemory.h"
#include "iuvector.h"

template<typename, int> class ndarray_ref;

namespace iu {

/**  \brief Linear host memory class.
 *   \ingroup LinearMemory
 */
template<typename PixelType, unsigned int Ndim>
class LinearHostMemory: public LinearMemory<Ndim>
{
public:
  /** Constructor. */
  LinearHostMemory() :
      LinearMemory<Ndim>(), data_(0), ext_data_pointer_(false)
  {
  }

  /** Destructor. */
  virtual ~LinearHostMemory()
  {
    if ((!ext_data_pointer_) && (data_ != NULL))
    {
      free(data_);
      data_ = 0;
    }
  }

  /** Special constructor.
   *  @param size size of linear memory
   */
  LinearHostMemory(const Size<Ndim>& size) :
      LinearMemory<Ndim>(size), data_(0), ext_data_pointer_(false)
  {
    data_ = (PixelType*) malloc(this->numel() * sizeof(PixelType));
    if (data_ == 0)
      throw std::bad_alloc();
  }

  /** Special constructor.
   *  @param numel number of elements of linear memory. Size[0] equals the number of elements,
   *  the other dimensions are set to 1.
   */
  LinearHostMemory(const unsigned int& numel) :
      LinearMemory<Ndim>(numel), data_(0), ext_data_pointer_(false)
  {
    data_ = (PixelType*) malloc(this->numel() * sizeof(PixelType));
    if (data_ == 0)
      throw std::bad_alloc();
  }

  /** Special constructor.
   *  @param host_data Host data pointer
   *  @param size size of the linear memory
   *  @param ext_data_pointer Use external data pointer as internal data pointer
   */
  LinearHostMemory(PixelType* host_data, const Size<Ndim>& size,
                   bool ext_data_pointer = false) :
      LinearMemory<Ndim>(size), data_(0), ext_data_pointer_(ext_data_pointer)
  {
    if (host_data == 0)
      throw IuException("input data not valid", __FILE__, __FUNCTION__,
      __LINE__);
    if (ext_data_pointer_)
    {
      // This uses the external data pointer as internal data pointer.
      data_ = host_data;
    }
    else
    {
      // allocates an internal data pointer and copies the external data onto it.
      data_ = (PixelType*) malloc(this->numel() * sizeof(PixelType));
      if (data_ == 0)
        throw std::bad_alloc();
      memcpy(data_, host_data, this->numel() * sizeof(PixelType));
    }
  }

  /** Special constructor.
   *  @param host_data Host data pointer
   *  @param numel Number of elements of the linear memory
   *  @param ext_data_pointer Use external data pointer as internal data pointer
   */
  LinearHostMemory(PixelType* host_data, const unsigned int& numel,
                   bool ext_data_pointer = false) :
      LinearMemory<Ndim>(numel), data_(0), ext_data_pointer_(ext_data_pointer)
  {
    if (host_data == 0)
      throw IuException("input data not valid", __FILE__, __FUNCTION__,
      __LINE__);
    if (ext_data_pointer_)
    {
      // This uses the external data pointer as internal data pointer.
      data_ = host_data;
    }
    else
    {
      // allocates an internal data pointer and copies the external data onto it.
      data_ = (PixelType*) malloc(this->numel() * sizeof(PixelType));
      if (data_ == 0)
        throw std::bad_alloc();
      memcpy(data_, host_data, this->numel() * sizeof(PixelType));
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
    {
      std::stringstream msg;
      msg << "Index (" << offset << ") out of range (" << this->numel() << ").";
      throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
    }
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
    {
      std::stringstream msg;
      msg << "Offset (" << offset << ") out of range (" << this->numel() << ").";
      throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
    }
    return reinterpret_cast<const PixelType*>(&(data_[offset]));
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
    return thrust::pointer<PixelType, thrust::host_system_tag>(
        data() + this->numel());
  }

  /** Returns flag if the image data resides on the device/GPU (TRUE) or host/GPU (FALSE) */
  virtual bool onDevice() const
  {
    return false;
  }

  /** Get pixel value at a certain position.
   * @param idx Index
   * @return Pixel value
   */
  PixelType& getPixel(const unsigned int& idx)
  {
    if (idx >= this->numel())
    {
      std::stringstream msg;
      msg << "Index (" << idx << ") out of range (" << this->numel() << ").";
      throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
    }

    return this->data_[idx];
  }

  /** Get pixel value at a certain position.
   * @param idx0 Position at dimension 0
   * @param idx1 Position at dimension 1
   * @return Pixel value
   */
  template<typename ResultType = PixelType>
  typename std::enable_if<(Ndim > 1), ResultType&>::type getPixel(
      const unsigned int& idx0, const unsigned int& idx1)
  {
    return data_[getLinearIndex(idx0, idx1)];
  }

  /** Get pixel value at a certain position.
   * @param idx0 Position at dimension 0
   * @param idx1 Position at dimension 1
   * @param idx2 Position at dimension 2
   * @return Pixel value
   */
  template<typename ResultType = PixelType>
  typename std::enable_if<(Ndim > 2), ResultType&>::type getPixel(
      const unsigned int& idx0, const unsigned int& idx1,
      const unsigned int& idx2)
  {
    return data_[getLinearIndex(idx0, idx1, idx2)];
  }

  /** Get pixel value at a certain position.
   * @param idx0 Position at dimension 0
   * @param idx1 Position at dimension 1
   * @param idx2 Position at dimension 2
   * @param idx3 Position at dimension 3
   * @return Pixel value
   */
  template<typename ResultType = PixelType>
  typename std::enable_if<(Ndim > 3), ResultType&>::type getPixel(
      const unsigned int& idx0, const unsigned int& idx1,
      const unsigned int& idx2, const unsigned int& idx3)
  {
    return data_[getLinearIndex(idx0, idx1, idx2, idx3)];
  }

  /** Get pixel value at a certain position.
   * @param idx0 Position at dimension 0
   * @param idx1 Position at dimension 1
   * @param idx2 Position at dimension 2
   * @param idx3 Position at dimension 3
   * @param idx4 Position at dimension 4
   * @return Pixel value
   */
  template<typename ResultType = PixelType>
  typename std::enable_if<(Ndim > 4), ResultType&>::type getPixel(
      const unsigned int& idx0, const unsigned int& idx1,
      const unsigned int& idx2, const unsigned int& idx3,
      const unsigned int& idx4)
  {
    return data_[getLinearIndex(idx0, idx1, idx2, idx3, idx4)];
  }

protected:

protected:
  /** Pointer to host buffer. */
  PixelType* data_;
  /** Flag if data pointer is handled outside the LinearHostMemory class. */
  bool ext_data_pointer_;

private:
  /** Private copy constructor. */
  LinearHostMemory(const LinearHostMemory&);
  /** Private copy assignment operator. */
  LinearHostMemory& operator=(const LinearHostMemory&);

  /** Convert pixel position to linear index
   * @param idx0 Position at dimension 0
   * @param idx1 Position at dimension 1
   * @return Linear index
   */
  template<typename ResultType = unsigned int>
  typename std::enable_if<(Ndim > 1), ResultType>::type getLinearIndex(
      const unsigned int& idx0, const unsigned int& idx1)
  {
    if (idx0 >= this->size()[0] || idx1 >= this->size()[1])
    {
      std::stringstream msg;
      msg <<  "Index (" << idx0 << ", " << idx1 << ") out of range ("
          << this->size() << ").";
      throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
    }

    unsigned int linear_idx = idx0;
    linear_idx += this->stride()[1] * idx1;
    return linear_idx;
  }

  /** Convert pixel position to linear index
   * @param idx0 Position at dimension 0
   * @param idx1 Position at dimension 1
   * @param idx2 Position at dimension 2
   * @return Linear index
   */
  template<typename ResultType = unsigned int>
  typename std::enable_if<(Ndim > 2), ResultType>::type getLinearIndex(
      const unsigned int& idx0, const unsigned int& idx1,
      const unsigned int& idx2)
  {
    if (idx0 >= this->size()[0] || idx1 >= this->size()[1]
        || idx2 >= this->size()[2])
    {
      std::stringstream msg;
      msg <<  "Index (" << idx0 << ", " << idx1 << ", " << idx2
          << ") out of range (" << this->size() << ").";
      throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
    }

    unsigned int linear_idx = idx0;
    linear_idx += this->stride()[1] * idx1;
    linear_idx += this->stride()[2] * idx2;
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
  typename std::enable_if<(Ndim > 3), ResultType>::type getLinearIndex(
      const unsigned int& idx0, const unsigned int& idx1,
      const unsigned int& idx2, const unsigned int& idx3)
  {
    if (idx0 >= this->size()[0] || idx1 >= this->size()[1]
        || idx2 >= this->size()[2] || idx3 >= this->size()[3])
    {
      std::stringstream msg;
      msg << "Index (" << idx0 << ", " << idx1 << ", " << idx2 << ", " << idx3
          << ") out of range (" << this->size() << ").";
      throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
    }

    unsigned int linear_idx = idx0;
    linear_idx += this->stride()[1] * idx1;
    linear_idx += this->stride()[2] * idx2;
    linear_idx += this->stride()[3] * idx3;
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
  typename std::enable_if<(Ndim > 4), ResultType>::type getLinearIndex(
      const unsigned int& idx0, const unsigned int& idx1,
      const unsigned int& idx2, const unsigned int& idx3,
      const unsigned int& idx4)
  {
    if (idx0 >= this->size()[0] || idx1 >= this->size()[1]
        || idx2 >= this->size()[2] || idx3 >= this->size()[3]
        || idx4 >= this->size()[4])

    {
      std::stringstream msg;
      msg << "Index (" << idx0 << ", " << idx1 << ", " << idx2 << ", " << idx3
          << ", " << idx4 << ") out of range (" << this->size() << ").";
      throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
    }

    unsigned int linear_idx = idx0;
    linear_idx += this->stride()[1] * idx1;
    linear_idx += this->stride()[2] * idx2;
    linear_idx += this->stride()[3] * idx3;
    linear_idx += this->stride()[4] * idx4;
    return linear_idx;
  }
};

template<typename PixelType>
class LinearHostMemory1d : public LinearHostMemory<PixelType, 1>
{
public:
  LinearHostMemory1d(const unsigned int& numel) : LinearHostMemory<PixelType, 1>(numel) {}

  /** convert to ndarray_ref -- include ndarray/ndarray_iu.h*/
  ndarray_ref<PixelType, 1> ref() const;

  /** construct from ndarray_ref  -- include ndarray/ndarray_iu.h*/
  LinearHostMemory1d(const ndarray_ref<PixelType, 1> &x);
};

}  // namespace iu

#endif // IU_LINEARHOSTMEMORY_H
