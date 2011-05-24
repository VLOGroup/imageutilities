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
 * Class       : none
 * Language    : C
 * Description : Implementation of CUDA functions for clamping
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */


#ifndef IUCORE_CLAMP_CU
#define IUCORE_CLAMP_CU

#include "iucutil.h"
#include "memorydefs.h"
#include "coredefs.h"
#include "iutextures.cuh"


namespace iuprivate {


///////////////////////////////////////////////////////////////////////////////
// thresholding/clamping kernels

// kernel: 1D set values; 1D
template<class T>
__global__ void cuClampKernel(T min, T max, T* dst, int length)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if(x<length)
  {
    dst[x] = clamp(min, max, dst[x]);
  }
}

// kernel: 2D set values;
template<class T>
__global__ void cuClampKernel_32f_C1(T min, T max, T* dst, size_t stride,
                                     int xoff, int yoff, int width, int height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  const int c =  y*stride+x;

  float xx = x + xoff + 0.5f;
  float yy = y + yoff + 0.5f;

  if(x>=0 && y>=0 && x<width && y<height)
  {
    dst[c] = clamp(tex2D(tex1_32f_C1__, xx, yy), min, max);
  }
}

///////////////////////////////////////////////////////////////////////////////

IuStatus cuClamp(const float& min, const float& max,
                  iu::ImageGpu_32f_C1 *srcdst, const IuRect &roi)
{
  // bind textures
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
  cudaBindTexture2D(0, &tex1_32f_C1__, srcdst->data(), &channel_desc, srcdst->width(), srcdst->height(), srcdst->pitch());

  // fragmentation
  const unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(roi.width, dimBlock.x),
               iu::divUp(roi.height, dimBlock.y));

  cuClampKernel_32f_C1 <<< dimGrid, dimBlock >>> (
      min, max, srcdst->data(roi.x, roi.y), srcdst->stride(),
      roi.x, roi.y, roi.width, roi.height);

  IU_CHECK_AND_RETURN_CUDA_ERRORS();
}

} // namespace iuprivate

#endif // IUCORE_CLAMP_CU
