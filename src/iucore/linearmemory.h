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
 * Project     : Utilities for IPP and NPP images
 * Module      : Memory (linear) base class
 * Class       : LinearMemory
 * Language    : C++
 * Description : Implementation of linear memory base class
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at

 *
 */

#ifndef LINEARMEMORY_H
#define LINEARMEMORY_H


#include "globaldefs.h"
#include "coredefs.h"
#include <typeinfo>

namespace iu {

/** \brief Base class for linear memory classes.
  */
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

} // namespace iu

#endif // LINEARMEMORY_H
