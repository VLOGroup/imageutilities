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

#include <global/cudadefs.h>
#include "memory_modification.cuh"
#include "memory_modification_kernels.cu"

namespace iuprivate {

///////////////////////////////////////////////////////////////////////////////

// wrapper: set values; 1D; 8-bit
NppStatus cuSetValue(const Npp8u& value, iu::LinearDeviceMemory_8u* dst)
{
  // fragmentation
  const unsigned int block_width = 512;
  dim3 dimBlock(block_width, 1, 1);
  dim3 dimGrid(iu::divUp(dst->length(), dimBlock.x), 1);

  cuSetValueKernel <<< dimGrid, dimBlock >>> (
      value, dst->data(), dst->length());

  IU_CHECK_CUDA_ERRORS();
  return NPP_SUCCESS;
}

// wrapper: set values; 1D; 32-bit
NppStatus cuSetValue(const Npp32f& value, iu::LinearDeviceMemory_32f* dst)
{
  // fragmentation
  const unsigned int block_width = 512;
  dim3 dimBlock(block_width, 1, 1);
  dim3 dimGrid(iu::divUp(dst->length(), dimBlock.x), 1);

  cuSetValueKernel <<< dimGrid, dimBlock >>> (
      value, dst->data(), dst->length());

  IU_CHECK_CUDA_ERRORS();
  return NPP_SUCCESS;
}

// wrapper: set values (single value); 2D; ...
template<typename PixelType, unsigned int NumChannels, class Allocator>
NppStatus cuSetValueTemplate(const PixelType &value,
                             iu::ImageNpp<PixelType, NumChannels, Allocator> *dst, const IuRect& roi)
{
  // fragmentation
  const unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(roi.width - roi.x, dimBlock.x),
               iu::divUp(roi.height - roi.y, dimBlock.y));


  cuSetValueKernel <<< dimGrid, dimBlock >>> (
      value, dst->data(roi.x, roi.y), dst->stride(), NumChannels,
      roi.x, roi.y, roi.width, roi.height);

  IU_CHECK_CUDA_ERRORS();
  return NPP_SUCCESS;
}

// wrapper: set values (single value); 2D; 8-bit;
NppStatus cuSetValue(const Npp8u& value, iu::ImageNpp_8u_C1 *dst, const IuRect &roi)
{ return cuSetValueTemplate(value, dst, roi); }
NppStatus cuSetValue(const Npp8u& value, iu::ImageNpp_8u_C2 *dst, const IuRect &roi)
{ return cuSetValueTemplate(value, dst, roi); }
NppStatus cuSetValue(const Npp8u& value, iu::ImageNpp_8u_C3 *dst, const IuRect &roi)
{ return cuSetValueTemplate(value, dst, roi); }
NppStatus cuSetValue(const Npp8u& value, iu::ImageNpp_8u_C4 *dst, const IuRect &roi)
{ return cuSetValueTemplate(value, dst, roi); }
// wrapper: set values (single value); 2D; 8-bit;
NppStatus cuSetValue(const Npp32f& value, iu::ImageNpp_32f_C1 *dst, const IuRect &roi)
{ return cuSetValueTemplate(value, dst, roi); }
NppStatus cuSetValue(const Npp32f& value, iu::ImageNpp_32f_C2 *dst, const IuRect &roi)
{ return cuSetValueTemplate(value, dst, roi); }
NppStatus cuSetValue(const Npp32f& value, iu::ImageNpp_32f_C3 *dst, const IuRect &roi)
{ return cuSetValueTemplate(value, dst, roi); }
NppStatus cuSetValue(const Npp32f& value, iu::ImageNpp_32f_C4 *dst, const IuRect &roi)
{ return cuSetValueTemplate(value, dst, roi); }

///////////////////////////////////////////////////////////////////////////////

NppStatus cuClamp(const Npp32f& min, const Npp32f& max,
                  iu::ImageNpp_32f_C1 *srcdst, const IuRect &roi)
{
  // bind textures
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float1>();
  cudaBindTexture2D(0, &tex1_32f_C1__, srcdst->data(), &channel_desc, srcdst->width(), srcdst->height(), srcdst->pitch());

  // fragmentation
  const unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(roi.width - roi.x, dimBlock.x),
               iu::divUp(roi.height - roi.y, dimBlock.y));


  cuClampKernel_32f_C1 <<< dimGrid, dimBlock >>> (
      min, max, srcdst->data(roi.x, roi.y), srcdst->stride(),
      roi.x, roi.y, roi.width, roi.height);

  IU_CHECK_CUDA_ERRORS();
  return NPP_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////

// convert 32f_C3 -> 32f_C4
NppStatus cuConvert(const iu::ImageNpp_32f_C3* src, const IuRect& src_roi, iu::ImageNpp_32f_C4* dst, const IuRect& dst_roi)
{
  // fragmentation
  const unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(dst_roi.width - dst_roi.x, dimBlock.x),
               iu::divUp(dst_roi.height - dst_roi.y, dimBlock.y));

  cuConvertC3ToC4Kernel <<< dimGrid, dimBlock >>> (
      src->data(src_roi.x, src_roi.y), src->stride(), src_roi.width, src_roi.height,
      dst->data(dst_roi.x, dst_roi.y), dst->stride(), dst_roi.width, dst_roi.height);

  IU_CHECK_CUDA_ERRORS();
  return NPP_SUCCESS;
}

// convert 32f_C4 -> 32f_C3
NppStatus cuConvert(const iu::ImageNpp_32f_C4* src, const IuRect& src_roi, iu::ImageNpp_32f_C3* dst, const IuRect& dst_roi)
{
  // fragmentation
  const unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(dst_roi.width - dst_roi.x, dimBlock.x),
               iu::divUp(dst_roi.height - dst_roi.y, dimBlock.y));

  cuConvertC4ToC3Kernel <<< dimGrid, dimBlock >>> (
      src->data(src_roi.x, src_roi.y), src->stride(), src_roi.width, src_roi.height,
      dst->data(dst_roi.x, dst_roi.y), dst->stride(), dst_roi.width, dst_roi.height);

  IU_CHECK_CUDA_ERRORS();
  return NPP_SUCCESS;
}


} // namespace iuprivate

#endif // IUCORE_MEMORY_MODIFICATION_CU
