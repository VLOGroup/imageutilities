
#ifndef LINEARMEMORY_H
#define LINEARMEMORY_H


#include "globaldefs.h"
#include "coredefs.h"
#include <typeinfo>

namespace iu {

/** \defgroup MemoryManagement Memory management
 *  \ingroup Core
 *  \brief Handles memory management of different types of host and device classes.
 *
 * - LinearMemory and specialized 4D tensor
 *    - LinearHostMemory
 *    - LinearDeviceMemory
 *    - TensorCpu
 *    - TensorGpu
 * - Pitched memory: Image
 *    - ImageCpu
 *    - ImageGpu
 * - Pitched memory: Volume
 *    - VolumeCpu
 *    - VolumeGpu
 *
 * The device memory classes can be easily passed to CUDA kernels using a special
 * struct. This struct gives the possibility to not only access the data pointer
 * of the image but also other useful information such as length/size of the
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
 *  - LinearHostMemory
 *  - LinearDeviceMemory
 *  - TensorCpu
 *  - TensorGpu
 *
 * The device memory classes can be easily passed to CUDA kernels using a special
 * struct. This struct gives the possibility to not only access the data pointer
 * of the image but also other useful information such as length of the
 * object.
 * - LinearDeviceMemory::KernelData
 * - TensorGpu::TensorKernelData
 *  \{
 */

/** \brief Base class for linear memory classes. */
class LinearMemory
{
public:
  /** Constructor. */
  LinearMemory() :
    length_(0)
  { }

  /** Special constructor.
   *  @param length Length of linear memory
   */
  LinearMemory(const unsigned int& length) :
    length_(length)
  { }

  /** Compares the LinearMemory type to a target LinearMemory.
   *  @param from Target LinearMemory.
   *  @return Returns true if target class is of the same type (using RTTI).
   */
  bool sameType(const LinearMemory &from)
  {
      return typeid(from)==typeid(*this);
  }

  /** Destructor. */
  virtual ~LinearMemory()
  { }

  /** Returns the number of elements saved in the device buffer. (length of device buffer) */
  unsigned int length() const
  {
    return length_;
  }

  /** Returns the total amount of bytes saved in the data buffer. */
  virtual size_t bytes() const {return 0;}

  /** Returns the bit depth of the data pointer. */
  virtual unsigned int bitDepth() const {return 0;}

  /** Returns flag if the image data resides on the device/GPU (TRUE) or host/GPU (FALSE) */
  virtual bool onDevice() const {return false;}

  /** Operator<< overloading. Output of LinearMemory class. */
  friend std::ostream& operator<<(std::ostream & out,
                                  LinearMemory const& linmem)
  {
    out << "LinearMemory: length=" << linmem.length() << " onDevice=" << linmem.onDevice();
    return out;
  }

private:
  /** Length of the memory.*/
  unsigned int length_;

private:
  /** Private copy constructor. */
  LinearMemory(const LinearMemory&);
  /** Private copy assignment operator. */
  LinearMemory& operator=(const LinearMemory&);
};

/** \} */ // end of Memory Management
/** \} */ // end of Linear Memory

} // namespace iu

#endif // LINEARMEMORY_H
