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
template <typename PixelType, unsigned int NumChannels>
class ImageAllocatorCpu
{
public:
  static PixelType* alloc(unsigned int width, unsigned int height, size_t *pitch)
  {
    IU_ASSERT(width * height > 0);
    PixelType *buffer = new PixelType[width * NumChannels * height];
    IU_ASSERT(buffer != 0);
    *pitch = width * sizeof(PixelType) * NumChannels;
    return buffer;
  }

  static void free(PixelType *buffer)
  {
    delete[] buffer;
  }

  static void copy(const PixelType *src, size_t src_pitch,
                   PixelType *dst, size_t dst_pitch, IuSize size)
  {
    size_t src_stride = src_pitch/sizeof(*src);
    size_t dst_stride = src_pitch/sizeof(*src);

    for(unsigned int y=0; y< size.height; ++y)
    {
      for(unsigned int x=0; x<size.width; ++x)
      {
        for(unsigned int channel=0; channel<NumChannels; ++channel)
        {
          dst[y*dst_stride+x*NumChannels+channel] = src[y*src_stride+x*NumChannels+channel];
        }
      }
    }
  }
};

} // namespace iuprivate

#endif // IMAGE_ALLOCATOR_CPU_H
