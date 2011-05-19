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
 * Module      : GUI
 * Class       : qgl
 * Language    : CUDA
 * Description : Implementation of device functions for the qgl.
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */


#ifndef IUPRIVATE_QGL_IMAGE_GPU_WIDGET_CU
#define IUPRIVATE_QGL_IMAGE_GPU_WIDGET_CU

#include "iucutil.h"
#include "iudefs.h"

namespace iuprivate {

texture<unsigned char,  2, cudaReadModeElementType> tex_qgl_image_8u_C1;
texture<uchar4, 2, cudaReadModeElementType> tex_qgl_image_8u_C4;
texture<float,  2, cudaReadModeElementType> tex_qgl_image_32f_C1;
texture<float4, 2, cudaReadModeElementType> tex_qgl_image_32f_C4;


/** Kernel to copy image data into OpenGL PBO. */
__global__ void cuCopyImageToPboKernel_8u_C1(uchar4* dst, int width, int height)
{
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;
  const int c = y*width+x;
  //Add half of a texel to always address exact texel centers
  const float xx = (float)x + 0.5f;
  const float yy = (float)y + 0.5f;

  if(x<width && y<height)
  {
    unsigned char val = tex2D(tex_qgl_image_8u_C1, xx, yy);
    dst[c] = make_uchar4(val, val, val, 1.0f);
  }
}

/** Kernel to copy image data into OpenGL PBO. */
__global__ void cuCopyImageToPboKernel_8u_C4(uchar4* dst, int width, int height)
{
  unsigned long x = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned long y = blockDim.y * blockIdx.y + threadIdx.y;

  const int c = y*width+x;
  //Add half of a texel to always address exact texel centers
  const float xx = (float)x + 0.5f;
  const float yy = (float)y + 0.5f;

  if(x<width && y<height)
  {
    uchar4 val = tex2D(tex_qgl_image_8u_C4, xx, yy);
    dst[c] = val;
  }
}


/** Kernel to copy image data into OpenGL PBO. */
__global__ void cuCopyImageToPboKernel_32f_C1(uchar4* dst, int width, int height,
                                              float min=0.0f, float max=1.0f)
{
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;
  const int c = y*width+x;
  //Add half of a texel to always address exact texel centers
  const float xx = (float)x + 0.5f;
  const float yy = (float)y + 0.5f;

  if(x<width && y<height)
  {
    float val = tex2D(tex_qgl_image_32f_C1, xx, yy);
    //-min/(max-min)
    //val = val * 255.0f;
    val = 255.0f / (max-min) * (val-min);
    dst[c] = make_uchar4(val, val, val, 1.0f);

  }
}

/** Kernel to copy image data into OpenGL PBO. */
__global__ void cuCopyImageToPboKernel_32f_C4(uchar4* dst, int width, int height)
{
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;
  const int c = y*width+x;
  //Add half of a texel to always address exact texel centers
  const float xx = (float)x + 0.5f;
  const float yy = (float)y + 0.5f;

  if(x<width && y<height)
  {
    float4 val = tex2D(tex_qgl_image_32f_C4, xx, yy);
    dst[c] = make_uchar4(val.x*255.0f, val.y*255.0f, val.z*255.0f, val.w*255.0f);
  }
}

IuStatus cuCopyImageToPbo(iu::Image* image, unsigned int num_channels,
                          unsigned int bit_depth, uchar4 *dst,
                          float min=0.0f, float max=1.0f)
{
  // device fragmentation
  const unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(image->width(), dimBlock.x),
               iu::divUp(image->height(), dimBlock.y));

  if(bit_depth == 8)
  {
    if(num_channels == 1)
    {
      iu::ImageGpu_8u_C1* img = reinterpret_cast<iu::ImageGpu_8u_C1*>(image);
      cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<unsigned char>();
      cudaBindTexture2D(0, &tex_qgl_image_8u_C1, img->data(), &channel_desc,
                        img->width(), img->height(), img->pitch());
      cuCopyImageToPboKernel_8u_C1 <<< dimGrid, dimBlock >>> (dst, img->width(), img->height());
      cudaUnbindTexture(tex_qgl_image_8u_C1);
    }
    else
    {
      iu::ImageGpu_8u_C4* img = reinterpret_cast<iu::ImageGpu_8u_C4*>(image);
      cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();
      cudaBindTexture2D(0, &tex_qgl_image_8u_C4, img->data(), &channel_desc,
                        img->width(), img->height(), img->pitch());
      cuCopyImageToPboKernel_8u_C4 <<< dimGrid, dimBlock >>> (dst, img->width(), img->height());
      cudaUnbindTexture(tex_qgl_image_8u_C4);
    }
  }
  else
  {
    if(num_channels == 1)
    {
      iu::ImageGpu_32f_C1* img = reinterpret_cast<iu::ImageGpu_32f_C1*>(image);
      cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
      cudaBindTexture2D(0, &tex_qgl_image_32f_C1, img->data(), &channel_desc,
                        img->width(), img->height(), img->pitch());
      cuCopyImageToPboKernel_32f_C1 <<< dimGrid, dimBlock >>> (dst, img->width(), img->height(), min, max);
      cudaUnbindTexture(tex_qgl_image_32f_C1);
    }
    else
    {
      iu::ImageGpu_32f_C4* img = reinterpret_cast<iu::ImageGpu_32f_C4*>(image);
      cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
      cudaBindTexture2D(0, &tex_qgl_image_32f_C4, img->data(), &channel_desc,
                        img->width(), img->height(), img->pitch());
      cuCopyImageToPboKernel_32f_C4 <<< dimGrid, dimBlock >>> (dst, img->width(), img->height());
      cudaUnbindTexture(tex_qgl_image_32f_C4);
    }
  }

  return iu::checkCudaErrorState();
}




} // namespace iuprivate


#endif // IUPRIVATE_QGL_IMAGE_GPU_WIDGET_CU
