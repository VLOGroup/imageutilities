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
 * Description : Implementation of CUDA functions to set a value to GPU memory
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */


#ifndef IUCORE_SETVALUE_CU
#define IUCORE_SETVALUE_CU

#include "coredefs.h"
#include "memorydefs.h"
#include "iucutil.h"

namespace iuprivate {

/* ****************************************************************************
 *  1D
 * ****************************************************************************/

//-----------------------------------------------------------------------------
// kernel: 1D set values; 1D
template<class T>
__global__ void cuSetValueKernel(T value, T* dst, int length)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if(x<length)
  {
    dst[x] = value;
  }
}

//-----------------------------------------------------------------------------
// wrapper: set values; 1D; 8-bit
/** Sets values of 1D linear gpu memory.
 * \param value The pixel value to be set.
 * \param buffer Pointer to the buffer
 */
IuStatus cuSetValue(const unsigned char& value, iu::LinearDeviceMemory_8u_C1* dst)
{
  // fragmentation
  const unsigned int block_width = 512;
  dim3 dimBlock(block_width, 1, 1);
  dim3 dimGrid(iu::divUp(dst->length(), dimBlock.x), 1);

  cuSetValueKernel <<< dimGrid, dimBlock >>> (
      value, dst->data(), dst->length());

  IU_CHECK_AND_RETURN_CUDA_ERRORS();
}

//-----------------------------------------------------------------------------
// wrapper: set values; 1D; 32-bit
/** Sets values of 1D linear gpu memory.
 * \param value The pixel value to be set.
 * \param buffer Pointer to the buffer
 */
IuStatus cuSetValue(const int& value, iu::LinearDeviceMemory_32s_C1* dst)
{
  // fragmentation
  const unsigned int block_width = 512;
  dim3 dimBlock(block_width, 1, 1);
  dim3 dimGrid(iu::divUp(dst->length(), dimBlock.x), 1);

  cuSetValueKernel <<< dimGrid, dimBlock >>> (
      value, dst->data(), dst->length());

  IU_CHECK_AND_RETURN_CUDA_ERRORS();
}

//-----------------------------------------------------------------------------
// wrapper: set values; 1D; 32-bit
/** Sets values of 1D linear gpu memory.
 * \param value The pixel value to be set.
 * \param buffer Pointer to the buffer
 */
IuStatus cuSetValue(const float& value, iu::LinearDeviceMemory_32f_C1* dst)
{
  // fragmentation
  const unsigned int block_width = 512;
  dim3 dimBlock(block_width, 1, 1);
  dim3 dimGrid(iu::divUp(dst->length(), dimBlock.x), 1);

  cuSetValueKernel <<< dimGrid, dimBlock >>> (
      value, dst->data(), dst->length());

  IU_CHECK_AND_RETURN_CUDA_ERRORS();
}


/* ****************************************************************************
 *  2D
 * ****************************************************************************/

//-----------------------------------------------------------------------------
// kernel: 2D set values; multi-channel
template<class T>
__global__ void cuSetValueKernel(T value, T* dst, size_t stride,
                                 int xoff, int yoff, int width, int height)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  const int c = y*stride+x;

  // add xoff for checks after calculating the output pixel location c
  x+=xoff;
  y+=yoff;

  if(x>=0 && y>=0 && x<width && y<height)
  {
    dst[c] = value;
  }
}

//-----------------------------------------------------------------------------
// templated wrapper: set value; 2D;
template<typename PixelType, class Allocator>
IuStatus cuSetValueTemplate(const PixelType &value,
                             iu::ImageGpu<PixelType, Allocator> *dst, const IuRect& roi)
{
  // fragmentation
  const unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(roi.width, dimBlock.x),
               iu::divUp(roi.height, dimBlock.y));

  cuSetValueKernel <<< dimGrid, dimBlock >>> (
      value, dst->data(roi.x, roi.y), dst->stride(),
      roi.x, roi.y, roi.width, roi.height);

  IU_CHECK_AND_RETURN_CUDA_ERRORS();
}

//-----------------------------------------------------------------------------
// specialized wrapper: set values (single value); 2D; 8-bit;
IuStatus cuSetValue(const unsigned char& value, iu::ImageGpu_8u_C1 *dst, const IuRect &roi)
{ return cuSetValueTemplate(value, dst, roi); }
IuStatus cuSetValue(const uchar2& value, iu::ImageGpu_8u_C2 *dst, const IuRect &roi)
{ return cuSetValueTemplate(value, dst, roi); }
IuStatus cuSetValue(const uchar3& value, iu::ImageGpu_8u_C3 *dst, const IuRect &roi)
{ return cuSetValueTemplate(value, dst, roi); }
IuStatus cuSetValue(const uchar4& value, iu::ImageGpu_8u_C4 *dst, const IuRect &roi)
{ return cuSetValueTemplate(value, dst, roi); }
// wrapper: set values (single value); 2D; 32-bit;
IuStatus cuSetValue(const int& value, iu::ImageGpu_32s_C1 *dst, const IuRect &roi)
{ return cuSetValueTemplate(value, dst, roi); }
// wrapper: set values (single value); 2D; 32-bit;
IuStatus cuSetValue(const float& value, iu::ImageGpu_32f_C1 *dst, const IuRect &roi)
{ return cuSetValueTemplate(value, dst, roi); }
IuStatus cuSetValue(const float2& value, iu::ImageGpu_32f_C2 *dst, const IuRect &roi)
{ return cuSetValueTemplate(value, dst, roi); }
IuStatus cuSetValue(const float3& value, iu::ImageGpu_32f_C3 *dst, const IuRect &roi)
{ return cuSetValueTemplate(value, dst, roi); }
IuStatus cuSetValue(const float4& value, iu::ImageGpu_32f_C4 *dst, const IuRect &roi)
{ return cuSetValueTemplate(value, dst, roi); }


/* ****************************************************************************
 *  3D
 * ****************************************************************************/

//-----------------------------------------------------------------------------
// kernel: 3D set values; multi-channel
template<class T>
__global__ void cuSetValueKernel(T value, T* dst, size_t stride, size_t slice_stride,
                                 int xoff, int yoff, int zoff, int width, int height, int depth)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  const int c =  y*stride+x;

  // add xoff for checks after calculating the output pixel location c
  x+=xoff;
  y+=yoff;

  if(x>=0 && y>=0 && x<width && y<height)
  {
    for(int z = -min(0,zoff); z<depth; ++z)
      dst[c+z*slice_stride] = value;
  }
}

//-----------------------------------------------------------------------------
// templated wrapper: set values (single value); 3D; ...
template<typename PixelType, class Allocator>
IuStatus cuSetValueTemplate(const PixelType &value,
                             iu::VolumeGpu<PixelType, Allocator> *dst, const IuCube& roi)
{
  // fragmentation
  const unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(roi.width, dimBlock.x),
               iu::divUp(roi.height, dimBlock.y));

  cuSetValueKernel <<< dimGrid, dimBlock >>> (
      value, dst->data(roi.x, roi.y, roi.z), dst->stride(), dst->slice_stride(),
      roi.x, roi.y, roi.z, roi.width, roi.height, roi.depth);

  IU_CHECK_AND_RETURN_CUDA_ERRORS();
}

//-----------------------------------------------------------------------------
// specialized wrapper: set values (single value); 3D; 8-bit;
IuStatus cuSetValue(const unsigned char& value, iu::VolumeGpu_8u_C1 *dst, const IuCube &roi)
{ return cuSetValueTemplate(value, dst, roi); }
IuStatus cuSetValue(const uchar2& value, iu::VolumeGpu_8u_C2 *dst, const IuCube &roi)
{ return cuSetValueTemplate(value, dst, roi); }
IuStatus cuSetValue(const uchar4& value, iu::VolumeGpu_8u_C4 *dst, const IuCube &roi)
{ return cuSetValueTemplate(value, dst, roi); }
// wrapper: set values (single value); 3D; 32-bit;
IuStatus cuSetValue(const float& value, iu::VolumeGpu_32f_C1 *dst, const IuCube &roi)
{ return cuSetValueTemplate(value, dst, roi); }
IuStatus cuSetValue(const float2& value, iu::VolumeGpu_32f_C2 *dst, const IuCube &roi)
{ return cuSetValueTemplate(value, dst, roi); }
IuStatus cuSetValue(const float4& value, iu::VolumeGpu_32f_C4 *dst, const IuCube &roi)
{ return cuSetValueTemplate(value, dst, roi); }



} // namespace iuprivate

#endif // IUCORE_SETVALUE_CU
