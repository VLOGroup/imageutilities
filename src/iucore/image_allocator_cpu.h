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
 * Class       : ImageAllocatorCpu
 * Language    : C++
 * Description : Image allocation functions for Cpu images.
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IMAGE_ALLOCATOR_CPU_H
#define IMAGE_ALLOCATOR_CPU_H

#include <cstring>
#include "coredefs.h"

namespace iuprivate {

//--------------------------------------------------------------------------
template <typename PixelType>
class ImageAllocatorCpu
{
public:
  static PixelType* alloc(unsigned int width, unsigned int height, size_t *pitch)
  {
    IU_ASSERT(width * height > 0);
    PixelType *buffer = new PixelType[width * height];
    IU_ASSERT(buffer != 0);
    *pitch = width * sizeof(PixelType);
    return buffer;
  }

  static void free(PixelType *buffer)
  {
    delete[] buffer;
  }

  static void copy(const PixelType *src, size_t src_pitch,
                   PixelType *dst, size_t dst_pitch, IuSize size)
  {
    size_t src_stride = src_pitch/sizeof(PixelType);
    size_t dst_stride = src_pitch/sizeof(PixelType);

    for(unsigned int y=0; y< size.height; ++y)
    {
      for(unsigned int x=0; x<size.width; ++x)
      {
        dst[y*dst_stride+x] = src[y*src_stride+x];
      }
    }
  }
};

} // namespace iuprivate

#endif // IMAGE_ALLOCATOR_CPU_H
