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
 * Project     : VMLibraries
 * Module      : ImageUtilities; Matlab connector
 * Class       : none
 * Language    : C
 * Description : Interface for Matlab and iu memory layouts
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */


#ifndef IUPRIVATE_IUMATLABCONNECTOR_H
#define IUPRIVATE_IUMATLABCONNECTOR_H

//
//  W A R N I N G
//  -------------
//
// This file is not part of the IU API.  It exists purely as an
// implementation detail.  This header file may change from version to
// version without notice, or even be removed.
//

#include <iostream>
#include <iudefs.h>
#include <iucutil.h>
#include <iucore/memorydefs.h>
#include <iucore/copy.h>
#include <mex.h>

namespace iuprivate {

//-----------------------------------------------------------------------------
// [host] conversion from matlab to ImageCpu memory layout
template<typename PixelType, class Allocator, IuPixelType _pixel_type>
IuStatus convertMatlabToCpu(double* matlab_src_buffer, unsigned int width, unsigned int height,
                            iu::ImageCpu<PixelType, Allocator, _pixel_type> *dst)

{
  if(width > dst->width() || height > dst->height())
  {
    std::cerr << "Error in convertMatlabToCpu: memory dimensions mismatch!" << std::endl;
    return IU_MEM_COPY_ERROR;
  }

  // iterate over the smaller block of input and output
  for (unsigned int y = IUMAX(0, dst->roi().y); y<IUMIN(height, dst->roi().height); ++y)
  {
    for (unsigned int x = IUMAX(0, dst->roi().x); x<IUMIN(width, dst->roi().width); ++x)
    {
      *dst->data(x,y) = static_cast<float>(matlab_src_buffer[y + x*height]);
    }
  }

  return IU_NO_ERROR;
}

//-----------------------------------------------------------------------------
// [host] conversion from matlab to ImageCpu memory layout - float/single version
template<typename PixelType, class Allocator, IuPixelType _pixel_type>
IuStatus convertMatlabToCpu(float* matlab_src_buffer, unsigned int width, unsigned int height,
                            iu::ImageCpu<PixelType, Allocator, _pixel_type> *dst)

{
  if(width > dst->width() || height > dst->height())
  {
    std::cerr << "Error in convertMatlabToCpu: memory dimensions mismatch!" << std::endl;
    return IU_MEM_COPY_ERROR;
  }

  // iterate over the smaller block of input and output
  for (unsigned int y = IUMAX(0, dst->roi().y); y<IUMIN(height, dst->roi().height); ++y)
  {
    for (unsigned int x = IUMAX(0, dst->roi().x); x<IUMIN(width, dst->roi().width); ++x)
    {
      *dst->data(x,y) = matlab_src_buffer[y + x*height];
    }
  }

  return IU_NO_ERROR;
}

//-----------------------------------------------------------------------------
// [host] conversion from matlab to ImageCpu memory layout - int version
template<typename PixelType, class Allocator, IuPixelType _pixel_type>
IuStatus convertMatlabToCpu(int* matlab_src_buffer, unsigned int width, unsigned int height,
                            iu::ImageCpu<PixelType, Allocator, _pixel_type> *dst)

{
  if(width > dst->width() || height > dst->height())
  {
    std::cerr << "Error in convertMatlabToCpu: memory dimensions mismatch!" << std::endl;
    return IU_MEM_COPY_ERROR;
  }

  // iterate over the smaller block of input and output
  for (unsigned int y = IUMAX(0, dst->roi().y); y<IUMIN(height, dst->roi().height); ++y)
  {
    for (unsigned int x = IUMAX(0, dst->roi().x); x<IUMIN(width, dst->roi().width); ++x)
    {
      *dst->data(x,y) = matlab_src_buffer[y + x*height];
    }
  }

  return IU_NO_ERROR;
}

//-----------------------------------------------------------------------------
// [device] conversion from matlab to ImageGpu memory layout
template<typename PixelType, class Allocator, IuPixelType _pixel_type>
IuStatus convertMatlabToGpu(double* matlab_src_buffer, unsigned int width, unsigned int height,
                            iu::ImageGpu<PixelType, Allocator, _pixel_type> *dst)
{
  iu::ImageCpu_32f_C1 tmp_cpu(dst->size());
  tmp_cpu.roi() = dst->roi();

  IuStatus status = convertMatlabToCpu(matlab_src_buffer, width, height, &tmp_cpu);
  if(status != IU_SUCCESS)
    return status;

  iuprivate::copy(&tmp_cpu, dst);
  return IU_NO_ERROR;
}

//-----------------------------------------------------------------------------
// [device] conversion from matlab to ImageGpu memory layout
template<typename PixelType, class Allocator, IuPixelType _pixel_type>
IuStatus convertMatlabToGpu(int* matlab_src_buffer, unsigned int width, unsigned int height,
                            iu::ImageGpu<PixelType, Allocator, _pixel_type> *dst)
{
  iu::ImageCpu_32s_C1 tmp_cpu(dst->size());
  tmp_cpu.roi() = dst->roi();

  IuStatus status = convertMatlabToCpu(matlab_src_buffer, width, height, &tmp_cpu);
  if(status != IU_SUCCESS)
    return status;

  iuprivate::copy(&tmp_cpu, dst);
  return IU_NO_ERROR;
}

//-----------------------------------------------------------------------------
// [host] conversion from matlab 3-channel to ImageCpu 4-channel memory layout
IuStatus convertMatlabC3ToCpuC4(double* matlab_src_buffer, unsigned int width, unsigned int height,
                                iu::ImageCpu_32f_C4 *dst)

{
  if(width > dst->width() || height > dst->height())
  {
    std::cerr << "Error in convertMatlabC3ToCpuC4: memory dimensions mismatch!" << std::endl;
    return IU_MEM_COPY_ERROR;
  }

  // iterate over the smaller block of input and output
  for (unsigned int y = IUMAX(0, dst->roi().y); y<IUMIN(height, dst->roi().height); ++y)
  {
    for (unsigned int x = IUMAX(0, dst->roi().x); x<IUMIN(width, dst->roi().width); ++x)
    {
      dst->data(x,y)->x = static_cast<float>(matlab_src_buffer[y + x*height]);
      dst->data(x,y)->y = static_cast<float>(matlab_src_buffer[y + x*height + width*height]);
      dst->data(x,y)->z = static_cast<float>(matlab_src_buffer[y + x*height + 2*width*height]);
      dst->data(x,y)->w = 1.0f;
    }
  }

  return IU_NO_ERROR;
}

//-----------------------------------------------------------------------------
// [host] conversion from matlab 2-channel to ImageCpu 2-channel memory layout
IuStatus convertMatlabC2ToCpuC2(double *matlab_src_buffer, unsigned int width, unsigned int height, iu::ImageCpu_32f_C2 *dst)
{
  if(width > dst->width() || height > dst->height())
  {
    std::cerr << "Error in convertMatlabC2ToCpuC2: memory dimensions mismatch!" << std::endl;
    return IU_MEM_COPY_ERROR;
  }

  // iterate over the smaller block of input and output
  for (unsigned int y = IUMAX(0, dst->roi().y); y<IUMIN(height, dst->roi().height); ++y)
  {
    for (unsigned int x = IUMAX(0, dst->roi().x); x<IUMIN(width, dst->roi().width); ++x)
    {
      dst->data(x,y)->x = static_cast<float>(matlab_src_buffer[y + x*height]);
      dst->data(x,y)->y = static_cast<float>(matlab_src_buffer[y + x*height + width*height]);
    }
  }

  return IU_NO_ERROR;
}

//-----------------------------------------------------------------------------
// [host] conversion from matlab 4-channel to ImageCpu 4-channel memory layout
IuStatus convertMatlabC4ToCpuC4(double *matlab_src_buffer, unsigned int width, unsigned int height, iu::ImageCpu_32f_C4 *dst)
{
  if(width > dst->width() || height > dst->height())
  {
    std::cerr << "Error in convertMatlabC4ToCpuC4: memory dimensions mismatch!" << std::endl;
    return IU_MEM_COPY_ERROR;
  }

  // iterate over the smaller block of input and output
  for (unsigned int y = IUMAX(0, dst->roi().y); y<IUMIN(height, dst->roi().height); ++y)
  {
    for (unsigned int x = IUMAX(0, dst->roi().x); x<IUMIN(width, dst->roi().width); ++x)
    {
      dst->data(x,y)->x = static_cast<float>(matlab_src_buffer[y + x*height]);
      dst->data(x,y)->y = static_cast<float>(matlab_src_buffer[y + x*height + width*height]);
      dst->data(x,y)->z = static_cast<float>(matlab_src_buffer[y + x*height + 2*width*height]);
      dst->data(x,y)->w = static_cast<float>(matlab_src_buffer[y + x*height + 3*width*height]);;
    }
  }

  return IU_NO_ERROR;
}

//-----------------------------------------------------------------------------
// [device] conversion from matlab 2-channel to ImageGpu 2-channel memory layout
IuStatus convertMatlabC2ToGpuC2(double *matlab_src_buffer, unsigned int width, unsigned int height, iu::ImageGpu_32f_C2 *dst)
{
  iu::ImageCpu_32f_C2 tmp_cpu(dst->size());
  tmp_cpu.roi() = dst->roi();

  IuStatus status = iuprivate::convertMatlabC2ToCpuC2(matlab_src_buffer, width, height, &tmp_cpu);
  if(status != IU_SUCCESS)
    return status;

  iuprivate::copy(&tmp_cpu, dst);
  return IU_NO_ERROR;
}

//-----------------------------------------------------------------------------
// [device] conversion from matlab 4-channel to ImageGpu 4-channel memory layout
IuStatus convertMatlabC4ToGpuC4(double *matlab_src_buffer, unsigned int width, unsigned int height, iu::ImageGpu_32f_C4 *dst)
{
  iu::ImageCpu_32f_C4 tmp_cpu(dst->size());
  tmp_cpu.roi() = dst->roi();

  IuStatus status = iuprivate::convertMatlabC4ToCpuC4(matlab_src_buffer, width, height, &tmp_cpu);
  if(status != IU_SUCCESS)
    return status;

  iuprivate::copy(&tmp_cpu, dst);
  return IU_NO_ERROR;
}

//-----------------------------------------------------------------------------
// [device] conversion from matlab 3-channel to ImageGpu 4-channel memory layout
IuStatus convertMatlabC3ToGpuC4(double* matlab_src_buffer, unsigned int width, unsigned int height,
                                iu::ImageGpu_32f_C4 *dst)
{
  iu::ImageCpu_32f_C4 tmp_cpu(dst->size());
  tmp_cpu.roi() = dst->roi();

  IuStatus status = iuprivate::convertMatlabC3ToCpuC4(matlab_src_buffer, width, height, &tmp_cpu);
  if(status != IU_SUCCESS)
    return status;

  iuprivate::copy(&tmp_cpu, dst);
  return IU_NO_ERROR;
}

//-----------------------------------------------------------------------------
// [host] conversion from ImageCpu 4-channel to matlab 3-channel memory layout
IuStatus convertCpuC4ToMatlabC3(iu::ImageCpu_32f_C4 *src, double* matlab_dst_buffer)
{
  unsigned int width = src->roi().width;
  unsigned int height = src->roi().height;

  // iterate over the smaller block of input and output
  for (unsigned int y = src->roi().y; y<height; ++y)
  {
    for (unsigned int x = src->roi().x; x<width; ++x)
    {
      matlab_dst_buffer[y + x*height] = static_cast<double>(src->data(x,y)->x);
      matlab_dst_buffer[y + x*height + width*height] = static_cast<double>(src->data(x,y)->y);
      matlab_dst_buffer[y + x*height + 2*width*height] = static_cast<double>(src->data(x,y)->z);
    }
  }

  return IU_NO_ERROR;
}

//-----------------------------------------------------------------------------
// [host] conversion from ImageCpu 4-channel to matlab 4-channel memory layout
IuStatus convertCpuC4ToMatlabC4(iu::ImageCpu_32f_C4 *src, double* matlab_dst_buffer)
{
  unsigned int width = src->roi().width;
  unsigned int height = src->roi().height;

  // iterate over the smaller block of input and output
  for (unsigned int y = src->roi().y; y<height; ++y)
  {
    for (unsigned int x = src->roi().x; x<width; ++x)
    {
      matlab_dst_buffer[y + x*height] = static_cast<double>(src->data(x,y)->x);
      matlab_dst_buffer[y + x*height + width*height] = static_cast<double>(src->data(x,y)->y);
      matlab_dst_buffer[y + x*height + 2*width*height] = static_cast<double>(src->data(x,y)->z);
      matlab_dst_buffer[y + x*height + 3*width*height] = static_cast<double>(src->data(x,y)->w);
    }
  }

  return IU_NO_ERROR;
}

//-----------------------------------------------------------------------------
// [host] conversion from ImageCpu 2-channel to matlab 2-channel memory layout
IuStatus convertCpuC2ToMatlabC2(iu::ImageCpu_32f_C2 *src, double* matlab_dst_buffer)
{
  unsigned int width = src->roi().width;
  unsigned int height = src->roi().height;

  // iterate over the smaller block of input and output
  for (unsigned int y = src->roi().y; y<height; ++y)
  {
    for (unsigned int x = src->roi().x; x<width; ++x)
    {
      matlab_dst_buffer[y + x*height] = static_cast<double>(src->data(x,y)->x);
      matlab_dst_buffer[y + x*height + width*height] = static_cast<double>(src->data(x,y)->y);
    }
  }

  return IU_NO_ERROR;
}

//-----------------------------------------------------------------------------
// [device] conversion from ImageGpu 4-channel to matlab 3-channel memory layout
IuStatus convertGpuC4ToMatlabC3(iu::ImageGpu_32f_C4 *src, double* matlab_dst_buffer)
{
  iu::ImageCpu_32f_C4 tmp_cpu(src->size());
  tmp_cpu.roi() = src->roi();
  iuprivate::copy(src, &tmp_cpu);

  IuStatus status = iuprivate::convertCpuC4ToMatlabC3(&tmp_cpu, matlab_dst_buffer);
  if(status != IU_SUCCESS)
    return status;

  return IU_NO_ERROR;
}

//-----------------------------------------------------------------------------
// [device] conversion from ImageGpu 4-channel to matlab 4-channel memory layout
IuStatus convertGpuC4ToMatlabC4(iu::ImageGpu_32f_C4 *src, double* matlab_dst_buffer)
{
  iu::ImageCpu_32f_C4 tmp_cpu(src->size());
  tmp_cpu.roi() = src->roi();
  iuprivate::copy(src, &tmp_cpu);

  IuStatus status = iuprivate::convertCpuC4ToMatlabC4(&tmp_cpu, matlab_dst_buffer);
  if(status != IU_SUCCESS)
    return status;

  return IU_NO_ERROR;
}

//-----------------------------------------------------------------------------
// [device] conversion from ImageGpu 2-channel to matlab 2-channel memory layout
IuStatus convertGpuC2ToMatlabC2(iu::ImageGpu_32f_C2 *src, double* matlab_dst_buffer)
{
  iu::ImageCpu_32f_C2 tmp_cpu(src->size());
  tmp_cpu.roi() = src->roi();
  iuprivate::copy(src, &tmp_cpu);

  IuStatus status = iuprivate::convertCpuC2ToMatlabC2(&tmp_cpu, matlab_dst_buffer);
  if(status != IU_SUCCESS)
    return status;

  return IU_NO_ERROR;
}


//-----------------------------------------------------------------------------
// [host] conversion from ImageCpu to matlab memory layout
template<typename PixelType, class Allocator, IuPixelType _pixel_type>
IuStatus convertCpuToMatlab(iu::ImageCpu<PixelType, Allocator, _pixel_type> *src,
                            double* matlab_dst_buffer, unsigned int width, unsigned int height)
{
  if(width > src->width() || height > src->height())
  {
    std::cerr << "Error in convertCpuToMatlab: memory dimensions mismatch!" << std::endl;
    return IU_MEM_COPY_ERROR;
  }

  // iterate over the smaller block of input and output
  for (unsigned int y = IUMAX(0, src->roi().y); y<IUMIN(height, src->roi().height); ++y)
  {
    for (unsigned int x = IUMAX(0, src->roi().x); x<IUMIN(width, src->roi().width); ++x)
    {
      matlab_dst_buffer[y + x*height] = static_cast<double>(*src->data(x,y));
    }
  }

  return IU_NO_ERROR;
}

//-----------------------------------------------------------------------------
// [host] conversion from ImageCpu to matlab memory layout
template<typename PixelType, class Allocator, IuPixelType _pixel_type>
IuStatus convertCpuToMatlab(iu::ImageCpu<PixelType, Allocator, _pixel_type> *src,
                            unsigned char* matlab_dst_buffer, unsigned int width, unsigned int height)
{
  if(width > src->width() || height > src->height())
  {
    std::cerr << "Error in convertCpuToMatlab: memory dimensions mismatch!" << std::endl;
    return IU_MEM_COPY_ERROR;
  }

  // iterate over the smaller block of input and output
  for (unsigned int y = IUMAX(0, src->roi().y); y<IUMIN(height, src->roi().height); ++y)
  {
    for (unsigned int x = IUMAX(0, src->roi().x); x<IUMIN(width, src->roi().width); ++x)
    {
      matlab_dst_buffer[y + x*height] = (*src->data(x,y));
    }
  }

  return IU_NO_ERROR;
}


//-----------------------------------------------------------------------------
// [device] conversion from matlab to ImageGpu memory layout
template<typename PixelType, class Allocator, IuPixelType _pixel_type>
IuStatus convertGpuToMatlab(iu::ImageGpu<PixelType, Allocator, _pixel_type> *src,
                            double* matlab_dst_buffer, unsigned int width, unsigned int height)
{
  // BUG? ... should this be 32f_C1 ???
  // We want a double as output!!
  iu::ImageCpu_32f_C1 tmp_cpu(src->size());
  tmp_cpu.roi() = src->roi();
  iuprivate::copy(src, &tmp_cpu);

  IuStatus status = iuprivate::convertCpuToMatlab(&tmp_cpu, matlab_dst_buffer, width, height);
  if(status != IU_SUCCESS)
    return status;

  return IU_NO_ERROR;
}


//-----------------------------------------------------------------------------
// [device] conversion from matlab to ImageGpu memory layout
template<typename PixelType, class Allocator, IuPixelType _pixel_type>
IuStatus convertGpuToMatlab(iu::ImageGpu<PixelType, Allocator, _pixel_type> *src,
                            unsigned char* matlab_dst_buffer, unsigned int width, unsigned int height)
{
  iu::ImageCpu_8u_C1 tmp_cpu(src->size());
  tmp_cpu.roi() = src->roi();
  iuprivate::copy(src, &tmp_cpu);

  IuStatus status = iuprivate::convertCpuToMatlab(&tmp_cpu, matlab_dst_buffer, width, height);
  if(status != IU_SUCCESS)
    return status;

  return IU_NO_ERROR;
}

} // namespace iuprivate

#endif // IUPRIVATE_IUMATLABCONNECTOR_H
