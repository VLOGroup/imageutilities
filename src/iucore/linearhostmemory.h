
#ifndef IUCORE_LINEARHOSTMEMORY_H
#define IUCORE_LINEARHOSTMEMORY_H

#include <stdio.h>
#include <assert.h>
#include <cstdlib>
#include <string.h>         // memcpy
#include <thrust/memory.h>

#include "linearmemory.h"

template<typename, int> class ndarray_ref;

namespace iu {

/**  \brief Linear host memory class.
 *   \ingroup LinearMemory
 */
template<typename PixelType>
class LinearHostMemory : public LinearMemory
{
public:
  /** Constructor. */
  LinearHostMemory() :
    LinearMemory(),
    data_(0), ext_data_pointer_(false)
  {
  }

  /** Destructor. */
  virtual ~LinearHostMemory()
  {
    if((!ext_data_pointer_) && (data_!=NULL))
    {
      free(data_);
      data_ = 0;
    }
  }

  /** Special constructor.
   *  @param length Length of linear memory
   */
  LinearHostMemory(const unsigned int& length) :
    LinearMemory(length),
    data_(0), ext_data_pointer_(false)
  {
    data_ = (PixelType*)malloc(this->length()*sizeof(PixelType));
    if (data_ == 0) throw std::bad_alloc();
  }

  /** Special constructor.
   *  @param host_data Host data pointer
   *  @param length Length of the memory
   *  @param ext_data_pointer Use external data pointer as internal data pointer
   */
  LinearHostMemory(PixelType* host_data, const unsigned int& length, bool ext_data_pointer = false) :
    LinearMemory(length),
    data_(0), ext_data_pointer_(ext_data_pointer)
  {
    if (host_data==0) throw IuException("input data not valid", __FILE__, __FUNCTION__, __LINE__);
    if(ext_data_pointer_)
    {
      // This uses the external data pointer as internal data pointer.
      data_ = host_data;
    }
    else
    {
      // allocates an internal data pointer and copies the external data onto it.
      data_ = (PixelType*)malloc(this->length()*sizeof(PixelType));
      if (data_ == 0) throw std::bad_alloc();
      memcpy(data_, host_data, this->length() * sizeof(PixelType));
    }
  }

  /** Returns a pointer to the device buffer.
   * The pointer can be offset to position \a offset.
   * @param offset Offset of the pointer array.
   * @return Pointer to the device buffer.
   */
  PixelType* data(int offset = 0)
  {
    if ((size_t)offset > this->length()) throw IuException("offset not in range", __FILE__, __FUNCTION__, __LINE__);
    return &(data_[offset]);
  }

  /** Returns a const pointer to the device buffer.
   * The pointer can be offset to position \a offset.
   * @param offset Offset of the pointer array.
   * @return Const pointer to the device buffer.
   */
  const PixelType* data(int offset = 0) const
  {
    if ((size_t)offset > this->length()) throw IuException("offset not in range", __FILE__, __FUNCTION__, __LINE__);
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
      return thrust::pointer<PixelType, thrust::host_system_tag>(data()+length());
  }

  /** Returns flag if the image data resides on the device/GPU (TRUE) or host/GPU (FALSE) */
  virtual bool onDevice() const
  {
    return false;
  }

  /** convert to ndarray_ref -- include ndarray/ndarray_iu.h*/
    ndarray_ref<PixelType,1> ref() const;

    /** construct from ndarray_ref  -- include ndarray/ndarray_iu.h*/
    LinearHostMemory(const ndarray_ref<PixelType,1> &x);

protected:

private:
  /** Pointer to host buffer. */
  PixelType* data_;
  /** Flag if data pointer is handled outside the LinearHostMemory class. */
  bool ext_data_pointer_;

private:
  /** Private copy constructor. */
  LinearHostMemory(const LinearHostMemory&);
  /** Private copy assignment operator. */
  LinearHostMemory& operator=(const LinearHostMemory&);

};
} // namespace iu

#endif // IU_LINEARHOSTMEMORY_H
