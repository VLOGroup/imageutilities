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
 * Description : Implementation of CUDA wrappers for memory modifications
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */


#ifndef IUCORE_MEMORY_MODIFICATION_CU
#define IUCORE_MEMORY_MODIFICATION_CU

#include <iucutil.h>
#include "memory_modification.cuh"
#include "memory_modification_kernels.cu"

namespace iuprivate {

///////////////////////////////////////////////////////////////////////////////

// wrapper: set values; 1D; 8-bit
IuStatus cuSetValue(const unsigned char& value, iu::LinearDeviceMemory_8u_C1* dst)
{
  // fragmentation
  const unsigned int block_width = 512;
  dim3 dimBlock(block_width, 1, 1);
  dim3 dimGrid(iu::divUp(dst->length(), dimBlock.x), 1);

  cuSetValueKernel <<< dimGrid, dimBlock >>> (
      value, dst->data(), dst->length());

  IU_CHECK_AND_RETURN_CUDA_ERRORS();
  return IU_SUCCESS;
}

// wrapper: set values; 1D; 32-bit
IuStatus cuSetValue(const float& value, iu::LinearDeviceMemory_32f_C1* dst)
{
  // fragmentation
  const unsigned int block_width = 512;
  dim3 dimBlock(block_width, 1, 1);
  dim3 dimGrid(iu::divUp(dst->length(), dimBlock.x), 1);

  cuSetValueKernel <<< dimGrid, dimBlock >>> (
      value, dst->data(), dst->length());

  IU_CHECK_AND_RETURN_CUDA_ERRORS();
  return IU_SUCCESS;
}


///////////////////////////////////////////////////////////////////////////////

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
  return IU_SUCCESS;
}

// wrapper: set values (single value); 2D; 8-bit;
IuStatus cuSetValue(const unsigned char& value, iu::ImageGpu_8u_C1 *dst, const IuRect &roi)
{ return cuSetValueTemplate(value, dst, roi); }
IuStatus cuSetValue(const uchar2& value, iu::ImageGpu_8u_C2 *dst, const IuRect &roi)
{ return cuSetValueTemplate(value, dst, roi); }
IuStatus cuSetValue(const uchar3& value, iu::ImageGpu_8u_C3 *dst, const IuRect &roi)
{ return cuSetValueTemplate(value, dst, roi); }
IuStatus cuSetValue(const uchar4& value, iu::ImageGpu_8u_C4 *dst, const IuRect &roi)
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


///////////////////////////////////////////////////////////////////////////////

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
  return IU_SUCCESS;
}

// wrapper: set values (single value); 3D; 8-bit;
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
  return IU_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////

// convert 32f_C3 -> 32f_C4
IuStatus cuConvert(const iu::ImageGpu_32f_C3* src, const IuRect& src_roi, iu::ImageGpu_32f_C4* dst, const IuRect& dst_roi)
{
  // fragmentation
  const unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(dst_roi.width - dst_roi.x, dimBlock.x),
               iu::divUp(dst_roi.height - dst_roi.y, dimBlock.y));

  cuConvertC3ToC4Kernel <<< dimGrid, dimBlock >>> (
      src->data(src_roi.x, src_roi.y), src->stride(), src_roi.width, src_roi.height,
      dst->data(dst_roi.x, dst_roi.y), dst->stride(), dst_roi.width, dst_roi.height);

  IU_CHECK_AND_RETURN_CUDA_ERRORS();
  return IU_SUCCESS;
}

// convert 32f_C4 -> 32f_C3
IuStatus cuConvert(const iu::ImageGpu_32f_C4* src, const IuRect& src_roi, iu::ImageGpu_32f_C3* dst, const IuRect& dst_roi)
{
  // fragmentation
  const unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(dst_roi.width - dst_roi.x, dimBlock.x),
               iu::divUp(dst_roi.height - dst_roi.y, dimBlock.y));

  cuConvertC4ToC3Kernel <<< dimGrid, dimBlock >>> (
      src->data(src_roi.x, src_roi.y), src->stride(), src_roi.width, src_roi.height,
      dst->data(dst_roi.x, dst_roi.y), dst->stride(), dst_roi.width, dst_roi.height);

  IU_CHECK_AND_RETURN_CUDA_ERRORS();
  return IU_SUCCESS;
}


} // namespace iuprivate

#endif // IUCORE_MEMORY_MODIFICATION_CU
