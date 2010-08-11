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
 * Class       : LinearHostMemory
 * Language    : C++
 * Description : Inline implementation of a linear host memory class
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUCORE_LINEARHOSTMEMORY_H
#define IUCORE_LINEARHOSTMEMORY_H

#include <stdio.h>
#include <assert.h>
#include <cstdlib>

namespace iu {

template<typename PixelType>
class LinearHostMemory
{
public:
  LinearHostMemory() :
      data_(0), length_(0), ext_data_pointer_(false)
  {
  }

  virtual ~LinearHostMemory()
  {
    if((!ext_data_pointer_) && (data_!=NULL))
    {
      free(data_);
      data_ = 0;
    }
  }

  LinearHostMemory(const unsigned int& length) :
      data_(0), length_(length), ext_data_pointer_(false)
  {
    data_ = (PixelType*)malloc(length_*sizeof(PixelType));
    IU_ASSERT(data_!=NULL);
    if(data_ == NULL)
    {
      fprintf(stderr, "iu::LinearHostMemory: OUT OF MEMORY\n");
      length_ = NULL;
    }
  }

  LinearHostMemory(const LinearHostMemory<PixelType>& from) :
      data_(0), length_(from.length_), ext_data_pointer_(false)
  {
    if(from.data_ == NULL)
      return;

    data_ = (PixelType*)malloc(length_*sizeof(PixelType));
    IU_ASSERT(data_!=NULL);
    if(data_ == NULL)
    {
      fprintf(stderr, "iu::LinearHostMemory: OUT OF MEMORY\n");
      length_ = NULL;
    }
    memcpy(data_, from.data_, length_ * sizeof(PixelType));
  }

  LinearHostMemory(PixelType* host_data, const unsigned int& length, bool ext_data_pointer = false) :
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

      data_ = (PixelType*)malloc(length_*sizeof(PixelType));
      IU_ASSERT(data_!=NULL);
      if(data_ == NULL)
      {
        fprintf(stderr, "iu::LinearHostMemory: OUT OF MEMORY\n");
        length_ = NULL;
      }
      memcpy(data_, host_data, length_ * sizeof(PixelType));
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

} // namespace iu

#endif // IU_LINEARHOSTMEMORY_H
