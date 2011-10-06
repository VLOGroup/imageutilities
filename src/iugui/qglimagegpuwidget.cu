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
#include "overlay.h"

namespace iuprivate {

texture<unsigned char,  2, cudaReadModeElementType> tex_qgl_image_8u_C1;
texture<uchar4, 2, cudaReadModeElementType> tex_qgl_image_8u_C4;
texture<float,  2, cudaReadModeElementType> tex_qgl_image_32f_C1;
texture<float4, 2, cudaReadModeElementType> tex_qgl_image_32f_C4;


/** Kernel to copy image data into OpenGL PBO. */
__global__ void cuCopyImageToPboKernel_8u_C1(uchar4* dst, int width, int height,
                                             unsigned char min, unsigned char max)
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
    val = IUMIN(255u, IUMAX(0u, 255u * (val-min) / (max-min) ));
    dst[c] = make_uchar4(val, val, val, 255u);
  }
}

/** Kernel to copy image data into OpenGL PBO. */
__global__ void cuCopyImageToPboKernel_8u_C4(uchar4* dst, int width, int height,
                                             unsigned char min, unsigned char max)
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
    val.x = IUMIN(255u, IUMAX(0u, 255u * (val.x-min) / (max-min)));
    val.y = IUMIN(255u, IUMAX(0u, 255u * (val.y-min) / (max-min)));
    val.z = IUMIN(255u, IUMAX(0u, 255u * (val.z-min) / (max-min)));
    val.w = 255u;
    dst[c] = val;
  }
}


/** Kernel to copy image data into OpenGL PBO. */
__global__ void cuCopyImageToPboKernel_32f_C1(uchar4* dst, int width, int height,
                                              float min, float max)
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
    val = clamp(255.0f * (val-min) / (max-min), 0.0f, 255.0f);
    dst[c] = make_uchar4(val, val, val, 255);

  }
}

/** Kernel to copy image data into OpenGL PBO. */
__global__ void cuCopyImageToPboKernel_32f_C4(uchar4* dst, int width, int height,
                                              float min, float max)
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
    val = clamp(255.0f * (val-min) / (max-min), 0.0f, 255.0f);
    dst[c] = make_uchar4(val.x, val.y, val.z, 255);
  }
}

IuStatus cuCopyImageToPbo(iu::Image* image, unsigned int num_channels,
                          unsigned int bit_depth, uchar4 *dst,
                          float min, float max)
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
      cuCopyImageToPboKernel_8u_C1 <<< dimGrid, dimBlock >>> (dst, img->width(), img->height(), min, max);
      cudaUnbindTexture(tex_qgl_image_8u_C1);
    }
    else
    {
      iu::ImageGpu_8u_C4* img = reinterpret_cast<iu::ImageGpu_8u_C4*>(image);
      cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();
      cudaBindTexture2D(0, &tex_qgl_image_8u_C4, img->data(), &channel_desc,
                        img->width(), img->height(), img->pitch());
      cuCopyImageToPboKernel_8u_C4 <<< dimGrid, dimBlock >>> (dst, img->width(), img->height(), min, max);
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
      cuCopyImageToPboKernel_32f_C4 <<< dimGrid, dimBlock >>> (dst, img->width(), img->height(), min, max);
      cudaUnbindTexture(tex_qgl_image_32f_C4);
    }
  }

  return iu::checkCudaErrorState();
}

/** Kernel to superimpose overlay data onto OpenGL PBO. */
__global__ void cuCopyOverlayToPboKernel_8u_C1(uchar4* dst,
                                               uchar* lut_values, uchar4* lut_colors,
                                               int num_constraints, IuComparisonOperator comp_op,
                                               int width, int height)
{
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;
  const int c = y*width+x;
  //Add half of a texel to always address exact texel centers
  const float xx = (float)x + 0.5f;
  const float yy = (float)y + 0.5f;

  if(x<width && y<height)
  {
    uchar4 cur_color = dst[c];

    // loop over all lookup table values
    for(int i = 0; i<num_constraints; ++i)
    {
      unsigned char val = tex2D(tex_qgl_image_8u_C1, xx, yy);
      unsigned char lut_val = lut_values[i];
      bool mark_pixel = false;

      switch (comp_op)
      {
      case IU_NOT_EQUAL:
        mark_pixel = (val != lut_val);
        break;
      case IU_GREATER:
        mark_pixel = (val > lut_val);
        break;
      case IU_GREATER_EQUAL:
        mark_pixel = (val >= lut_val);
        break;
      case IU_LESS:
        mark_pixel = (val < lut_val);
        break;
      case IU_LESS_EQUAL:
        mark_pixel = (val <= lut_val);
        break;
      case IU_EQUAL:
      default:
        mark_pixel = (val == lut_val);
        break;
      }

      if(mark_pixel)
      {
        uchar4 col = lut_colors[i];
        float alpha = col.w/255.0f;
        cur_color = make_uchar4(alpha*col.x + (1-alpha)*cur_color.x,
                                alpha*col.y + (1-alpha)*cur_color.y,
                                alpha*col.z + (1-alpha)*cur_color.z,
                                255);
      }
    }

    dst[c] = cur_color;
  }
}

/** Kernel to superimpose overlay data onto OpenGL PBO. */
__global__ void cuCopyOverlayToPboKernel_32f_C1(uchar4* dst,
                                                float* lut_values, uchar4* lut_colors,
                                                int num_constraints, IuComparisonOperator comp_op,
                                                int width, int height)
{
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;
  const int c = y*width+x;
  //Add half of a texel to always address exact texel centers
  const float xx = (float)x + 0.5f;
  const float yy = (float)y + 0.5f;

  if(x<width && y<height)
  {
    uchar4 cur_color = dst[c];

    // loop over all lookup table values
    for(int i = 0; i<num_constraints; ++i)
    {
      float val = tex2D(tex_qgl_image_32f_C1, xx, yy);
      float lut_val = lut_values[i];
      bool mark_pixel = false;

      switch (comp_op)
      {
      case IU_NOT_EQUAL:
        mark_pixel = (val != lut_val);
        break;
      case IU_GREATER:
        mark_pixel = (val > lut_val);
        break;
      case IU_GREATER_EQUAL:
        mark_pixel = (val >= lut_val);
        break;
      case IU_LESS:
        mark_pixel = (val < lut_val);
        break;
      case IU_LESS_EQUAL:
        mark_pixel = (val <= lut_val);
        break;
      case IU_EQUAL:
      default:
        mark_pixel = (val == lut_val);
        break;
      }

      if (mark_pixel)
      {
        uchar4 col = lut_colors[i];
        float alpha = col.w/255.0f;
        cur_color = make_uchar4(alpha*col.x + (1-alpha)*cur_color.x,
                                alpha*col.y + (1-alpha)*cur_color.y,
                                alpha*col.z + (1-alpha)*cur_color.z,
                                255);
      }
    }

    dst[c] = cur_color;
  }
}

//-----------------------------------------------------------------------------
IuStatus cuCopyOverlayToPbo(iuprivate::Overlay* overlay, uchar4 *dst, IuSize size)
{
  // device fragmentation
  const unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(size.width, dimBlock.x),
               iu::divUp(size.height, dimBlock.y));

  if(overlay->getConstraintImage()->bitDepth() == 8)
  {
    iu::ImageGpu_8u_C1* constraints = reinterpret_cast<iu::ImageGpu_8u_C1*>(overlay->getConstraintImage());
    iu::LinearDeviceMemory_8u_C1* lut_values = reinterpret_cast<iu::LinearDeviceMemory_8u_C1*>(overlay->getLUTValues());
    if(constraints == NULL || lut_values == NULL)
    {
      fprintf(stderr, "unsuported datatype for constraint image or value LUT.\n");
      return IU_ERROR;
    }
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<unsigned char>();
    cudaBindTexture2D(0, &tex_qgl_image_8u_C1, constraints->data(), &channel_desc,
                      constraints->width(), constraints->height(), constraints->pitch());

    cuCopyOverlayToPboKernel_8u_C1
        <<< dimGrid, dimBlock >>> (dst,
                                   lut_values->data(), overlay->getLUTColors()->data(),
                                   lut_values->length(), overlay->getComparisonOperator(),
                                   constraints->width(), constraints->height());

    cudaUnbindTexture(tex_qgl_image_8u_C1);
  }
  else
  {
    iu::ImageGpu_32f_C1* constraints = reinterpret_cast<iu::ImageGpu_32f_C1*>(overlay->getConstraintImage());
    iu::LinearDeviceMemory_32f_C1* lut_values = reinterpret_cast<iu::LinearDeviceMemory_32f_C1*>(overlay->getLUTValues());
    if(constraints == NULL || lut_values == NULL)
    {
      fprintf(stderr, "unsuported datatype for constraint image.\n");
      return IU_ERROR;
    }
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
    cudaBindTexture2D(0, &tex_qgl_image_32f_C1, constraints->data(), &channel_desc,
                      constraints->width(), constraints->height(), constraints->pitch());
    cuCopyOverlayToPboKernel_32f_C1
        <<< dimGrid, dimBlock >>> (dst, lut_values->data(), overlay->getLUTColors()->data(),
                                   lut_values->length(), overlay->getComparisonOperator(),
                                   constraints->width(), constraints->height());
    cudaUnbindTexture(tex_qgl_image_32f_C1);
  }

  return iu::checkCudaErrorState();
}



} // namespace iuprivate


#endif // IUPRIVATE_QGL_IMAGE_GPU_WIDGET_CU
