#pragma once

#include <typeinfo>

#include "coredefs.h"
#include "vector.h"

namespace iu {
/** \defgroup MemoryManagement Memory management
 *  \ingroup Core
 *  \brief Handles memory management of different types of host and device classes.
 *
 * - LinearMemory
 *    - LinearHostMemory: linear memory with ND size layout
 *    - LinearDeviceMemory: linear memory with ND size layout
 *    - TensorCpu: Specialization with 1D size layout. Wrapper for special 4D tensor.
 *                 Same access convention as python.
 *    - TensorGpu: Specialization with 1D size layout. Wrapper for special 4D tensor.
 *                 Same access convention as python.
 * - Pitched memory: Image
 *    - ImageCpu
 *    - ImageGpu
 * - Pitched memory: Volume
 *    - VolumeCpu
 *    - VolumeGpu
 *
 * The device memory classes can be easily passed to CUDA kernels using a special
 * struct. This struct gives the possibility to not only access the data pointer
 * of the image but also other useful information such as numel/size of the
 * object.
 * - LinearDeviceMemory::KernelData
 * - TensorGpu::TensorKernelData
 * - ImageGpu::KernelData
 * - VolumeGpu::KernelData
 * \{
 */

/** \defgroup LinearMemory Linear memory
 *  \ingroup MemoryManagement
 *  \brief Memory management for LinearMemory classes.
 *
 *  This handles the memory management for following linear memory classes:
 *    - LinearHostMemory: linear memory with ND size layout
 *    - LinearDeviceMemory: linear memory with ND size layout
 *    - TensorCpu: Specialization with 1D size layout. Wrapper for special 4D tensor.
 *                 Same access convention as python.
 *    - TensorGpu: Specialization with 1D size layout. Wrapper for special 4D tensor.
 *                 Same access convention as python.
 *
 * The device memory classes can be easily passed to CUDA kernels using a special
 * struct. This struct gives the possibility to not only access the data pointer
 * of the image but also other useful information such as numel of the
 * object.
 * - LinearDeviceMemory::KernelData
 * - TensorGpu::TensorKernelData
 *  \{
 */

/** \brief Base class for linear memory classes. */
template<unsigned int Ndim = 1>
class LinearMemory
{
IU_ASSERT(Ndim > 0)
public:
  /** Constructor. */
  LinearMemory() :
      size_(), stride_()
  {
  }

  /** Special constructor.
   *  @param size size of the linear memory
   */
  LinearMemory(const Size<Ndim>& size) :
      size_(size)
  {
    computeStride();
  }

  /** Special constructor.
   *  @param numel number of elements of linear memory. Size[0] equals the number of elements,
   *  the other dimensions are 1.
   */
  LinearMemory(const unsigned int& numel) :
      size_()
  {
    size_[0] = numel;
    computeStride();
  }
  /** Compares the LinearMemory type to a target LinearMemory.
   *  @param from Target LinearMemory.
   *  @return Returns true if target class is of the same type (using RTTI).
   */
  bool sameType(const LinearMemory &from)
  {
    return typeid(from) == typeid(*this);
  }

  /** Destructor. */
  virtual ~LinearMemory()
  {
  }

  /** Returns the number of elements saved in the buffer. (numel of buffer) */
  unsigned int numel() const
  {
    return size_.numel();
  }

//  /** Returns the number of elements saved in the buffer. (numel of buffer) */
//  unsigned int length() const
//  {
//#pragma message("LinearMemory::length() is deprecated and will be removed in the future. Use numel() instead.")
////    std::cout
////        << "Warning: LinearMemory::length() is deprecated and will be removed in the future. Use numel() instead."
////        << std::endl;
//    return size_.numel();
//  }

  /** Returns size of the linear memory */
  Size<Ndim> size() const
  {
    return size_;
  }

  /** Returns size of the linear memory */
  Size<Ndim> stride() const
  {
    return stride_;
  }

  /** Returns the total amount of bytes saved in the data buffer. */
  virtual size_t bytes() const
  {
    return 0;
  }

  /** Returns the bit depth of the data pointer. */
  virtual unsigned int bitDepth() const
  {
    return 0;
  }

  /** Returns flag if the image data resides on the device/GPU (TRUE) or host/GPU (FALSE) */
  virtual bool onDevice() const
  {
    return false;
  }

  /** Operator<< overloading. Output of LinearMemory class. */
  friend std::ostream& operator<<(std::ostream & out,
                                  LinearMemory const& linmem)
  {
    out << "LinearMemory: size=" << linmem.size() << " strides="
        << linmem.stride() << " numel=" << linmem.numel() << " onDevice="
        << linmem.onDevice();
    return out;
  }

protected:
  /** size of the memory.*/
  Size<Ndim> size_;

  /** Compute the strides of the memory*/
  void computeStride()
  {
    for (unsigned int i = 0; i < Ndim; i++)
    {
      if (i == 0)
        stride_[i] = 1;
      else
        stride_[i] = stride_[i - 1] * size_[i - 1];
    }
  }

private:
  /** Stride of the memory. First dimension is always one.*/
  Size<Ndim> stride_;

private:
  /** Private copy constructor. */
  LinearMemory(const LinearMemory&);
  /** Private copy assignment operator. */
  LinearMemory& operator=(const LinearMemory&);
};

/** \} */  // end of Memory Management
/** \} */// end of Linear Memory
}// namespace iu


