#pragma once

#include <assert.h>
#include <cuda_runtime.h>
#include "coredefs.h"

namespace iuprivate {

//--------------------------------------------------------------------------
template <typename PixelType>
class VolumeAllocatorCpu
{
public:
  static PixelType* alloc(unsigned int width, unsigned int height, unsigned int depth, size_t *pitch)
  {
    if ((width==0) || (height==0) || (depth==0))
      throw IuException("width, height or depth is 0", __FILE__,__FUNCTION__, __LINE__);
    PixelType *buffer = new PixelType[width*height*depth];
    *pitch = width * sizeof(PixelType);
    return buffer;
  }

  static void free(PixelType *buffer)
  {
    delete[] buffer;
  }

  static void copy(const PixelType *src, size_t src_pitch,
                   PixelType *dst, size_t dst_pitch, iu::Size<3> size)
  {
    size_t src_stride = src_pitch/sizeof(PixelType);
    size_t dst_stride = dst_pitch/sizeof(PixelType);

    for(unsigned int z=0; z<size.depth; ++z)
    {
      for(unsigned int y=0; y<size.height; ++y)
      {
        for(unsigned int x=0; x<size.width; ++x)
        {
          dst[z*dst_stride*size.height + y*dst_stride + x] =
              src[z*src_stride*size.height + y*src_stride + x];
        }
      }
    }
  }
};

} // namespace iuprivate

