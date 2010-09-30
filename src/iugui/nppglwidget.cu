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
 * Class       : NppGLWidget
 * Language    : C++/CUDA
 * Description : Implementation of Cuda wrappers for the DeviceGLWidget
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef NPPGLWIDGET_CU
#define NPPGLWIDGET_CU

#include <float.h>
#include <cutil_math.h>
#include <GL/glew.h>
#include <cuda_gl_interop.h>

#include <iucutil.h>
#include "nppglwidget.cuh"

/* Textures have to be global - do not mess around with those!! */
texture<float1, 2, cudaReadModeElementType> nppglwidget_image_32f_C1__;
texture<float4, 2, cudaReadModeElementType> nppglwidget_image_32f_C4__;

namespace iuprivate {

/******************************************************************************
    CUDA KERNELS
*******************************************************************************/

//-----------------------------------------------------------------------------
/** Converts8-bit integer values out of r/g/b data that can be stored in a picture buffer object
 * @param[in] r first channel of the image
 * @param[in] g second channel of the image
 * @param[in] b third channel of the image
 */
__device__ int rgbToInt(float r, float g, float b)
{
  r = clamp(r*255.0f, 0.0f, 255.0f);
  g = clamp(g*255.0f, 0.0f, 255.0f);
  b = clamp(b*255.0f, 0.0f, 255.0f);
  return (int(b)<<16) | (int(g)<<8) | int(r);
}

//-----------------------------------------------------------------------------
/** Kernel to generates an output image from a grayscale image
 * @param[out] output        output to buffer object
 * @param[in] input         input image on device
 * @param[in] width         width of image
 * @param[in] height        height of image
 * @param[in] stride        stride of image
 */
__global__ void getOutputKernel_32f_C1(int* output, int xoff, int yoff, int width, int height,
                                       float min_val, float max_val)
{
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  float xx = x+xoff+0.5f;
  float yy = y+yoff+0.5f;

  if ((x<width) && (y<height))
  {
    float in = tex2D(nppglwidget_image_32f_C1__, xx, yy).x;
    float val = (in-min_val)/(abs(max_val-min_val));
    if (in == FLT_MIN)
      val = -1.0f;
    output[y*width+x] = rgbToInt(val, val, val);
  }
}

/** Kernel to generates an output image from a RGBA image
 * @param[out] output        output to buffer object
 * @param[in] input         input image on device
 * @param[in] width         width of image
 * @param[in] height        height of image
 * @param[in] stride        stride of image
 */
__global__ void getOutputKernel_32f_C4(int* output, int xoff, int yoff, int width, int height,
                                       float min_val, float max_val)
{
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
  float xx = x+xoff+0.5f;
  float yy = y+yoff+0.5f;
  if ((x<width) && (y<height))
  {
    float4 rgba = tex2D(nppglwidget_image_32f_C4__, xx, yy);

    float r = (rgba.x-min_val)/(abs(max_val-min_val));
    float g = (rgba.y-min_val)/(abs(max_val-min_val));
    float b = (rgba.z-min_val)/(abs(max_val-min_val));

    output[y*width+x] = rgbToInt(rgba.x, rgba.y, rgba.z);
  }
}

//
////-----------------------------------------------------------------------------
///** Kernel to generates an output image from a gray scale image
// * @param[out] output        output to buffer object
// * @param[in] input         input image on device
// * @param[in] width         width of image
// * @param[in] height        height of image
// * @param[in] stride        stride of image
// */
//__global__ void getOutputRGBPlaneKernel( int* output, float* input, int width, int height,
//                                         size_t stride, size_t strideY, float min_val, float max_val)
//{
//  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
//  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
//
//  if ((x<width) && (y<height))
//  {
//    int center = (y*stride)+x;
//    float r = ((float)input[center]-min_val)/(abs(max_val-min_val));
//    float g = ((float)input[center+strideY]-min_val)/(abs(max_val-min_val));
//    float b = ((float)input[center+2*strideY]-min_val)/(abs(max_val-min_val));
//    output[y*width+x] = rgbToInt(r, g, b);
//  }
//}
//
////-----------------------------------------------------------------------------
///** Kernel to convert RGB image data \a input to an int buffer \a output
// * @param[out] output buffer object
// * @param[in] input  input RGB image on device
// * @param[in] width  input image width
// * @param[in] height input image height
// * @param[in] stride input image stride
// */
//__global__ void getOutputRGBKernel(int* output, float3* input,
//                                   int width, int height, size_t stride, float min_val, float max_val)
//{
//  /*
//    ic ... input center
//    oc ... output center
//   */
//  const int x = blockIdx.x*blockDim.x + threadIdx.x;
//  const int y = blockIdx.y*blockDim.y + threadIdx.y;
//  const int ic = y*stride+x;
//  const int oc = y*width+x;
//  if ((x<width) && (y<height))
//  {
//    float r = ((float)input[ic].x-min_val)/(abs(max_val-min_val));
//    float g = ((float)input[ic].y-min_val)/(abs(max_val-min_val));
//    float b = ((float)input[ic].z-min_val)/(abs(max_val-min_val));
//    output[oc] = rgbToInt(r, g, b);
//  }
//}
//
////-----------------------------------------------------------------------------
///** Kernel to convert RGBA image data \a input to an int buffer \a output
// * @param[out] output buffer object
// * @param[in] input  input RGBA image on device
// * @param[in] width  input image width
// * @param[in] height input image height
// * @param[in] stride input image stride
// */
//__global__ void getOutputRGBAKernel(int* output, float4* input,
//                                    int width, int height, size_t stride, float min_val, float max_val)
//{
//  /*
//    ic ... input center
//    oc ... output center
//   */
//  const int x = blockIdx.x*blockDim.x + threadIdx.x;
//  const int y = blockIdx.y*blockDim.y + threadIdx.y;
//  const int ic = y*stride+x;
//  const int oc = y*width+x;
//  if ((x<width) && (y<height))
//  {
////    float a = input[ic].w; // Not used here
//    float r = ((float)input[ic].x-min_val)/(abs(max_val-min_val));
//    float g = ((float)input[ic].y-min_val)/(abs(max_val-min_val));
//    float b = ((float)input[ic].z-min_val)/(abs(max_val-min_val));
//
//    output[oc] = rgbToInt(r, g, b);
//  }
//}
//
////-----------------------------------------------------------------------------
///** Kernel that adds an overlay \a overlay to a pbo int buffer \a data
// * @param[out] data Acts as input and output buffer. Existing data will be overlayed with the new data.
// * @para[in] overlay Overlay mask.
// * @param[in] width width of the overlay mask
// * @param[in] height height of the overlay mask
// * @param[in] pitch pitch of the overlay mask
// * @param[in] r red value [0..255]
// * @param[in] g green value [0..255]
// * @param[in] b blue value [0..255]
// * @param[in] a alpha value [0..255]
// * @param[in] mask_value value for which the mask is shown
// */
//__global__ void createOverlayKernelF( int* data, float* overlay, int width, int height, size_t pitch,
//                                      int _r, int _g, int _b, int _a, float mask_value)
//{
//  /*
//    oc ... overlay center
//    dc ... data center
//   */
//  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
//  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
//  unsigned int oc = y*pitch+x;
//  unsigned int dc = y*width+x;
//
//  if ((x<width) && (y<height))
//  {
//    if (overlay[oc] == mask_value)
//    {
//      int in = data[dc];
//      float r = _r/255.0f;
//      float g = _g/255.0f;
//      float b = _b/255.0f;
//      float a = _a/255.0f;
//
//      float old_r = (in>>16)/255.0f;
//      float old_g = ((in & 0x00FF00)>>8)/255.0f;
//      float old_b = (in & 0x0000FF)/255.0f;
//
//      r = a*r+(1-a)*old_r;
//      g = a*g+(1-a)*old_g;
//      b = a*b+(1-a)*old_b;
//
//      data[dc] = rgbToInt(r, g, b);
//    }
//  }
//}
//
////-----------------------------------------------------------------------------
//__global__ void createOverlayKernelUC( int* data, unsigned char* overlay, int width, int height, size_t pitch,
//                                       int _r, int _g, int _b, int _a, unsigned char mask_value)
//{
//  /*
//    oc ... overlay center
//    dc ... data center
//   */
//  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
//  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
//  unsigned int oc = y*pitch+x;
//  unsigned int dc = y*width+x;
//
//  if ((x<width) && (y<height))
//  {
//    if (overlay[oc] == mask_value)
//    {
//      int in = data[dc];
//      float r = _r/255.0f;
//      float g = _g/255.0f;
//      float b = _b/255.0f;
//      float a = _a/255.0f;
//
//      float old_r = (in>>16)/255.0f;
//      float old_g = ((in & 0x00FF00)>>8)/255.0f;
//      float old_b = (in & 0x0000FF)/255.0f;
//
//      r = a*r+(1-a)*old_r;
//      g = a*g+(1-a)*old_g;
//      b = a*b+(1-a)*old_b;
//
//      data[dc] = rgbToInt(r, g, b);
//    }
//  }
//}

/******************************************************************************
  CUDA INTERFACES
*******************************************************************************/

//-----------------------------------------------------------------------------
IuStatus cuInitTextures()
{
  // general float texture
  nppglwidget_image_32f_C1__.filterMode = cudaFilterModePoint;
  nppglwidget_image_32f_C1__.addressMode[0] = cudaAddressModeClamp;
  nppglwidget_image_32f_C1__.addressMode[1] = cudaAddressModeClamp;
  nppglwidget_image_32f_C1__.normalized = false;

  // general float texture
  nppglwidget_image_32f_C4__.filterMode = cudaFilterModePoint;
  nppglwidget_image_32f_C4__.addressMode[0] = cudaAddressModeClamp;
  nppglwidget_image_32f_C4__.addressMode[1] = cudaAddressModeClamp;
  nppglwidget_image_32f_C4__.normalized = false;

  IU_CHECK_CUDA_ERRORS();
  return IU_SUCCESS;
}

//-----------------------------------------------------------------------------
/** Register a buffer object with CUDA
 * @param pbo index of picture buffer object
 */
IuStatus cuPboRegister(GLuint pbo, bool& registered)
{
  if(!registered)
  {
    cudaGLRegisterBufferObject(pbo);
    registered = true;
  }
  IU_CHECK_CUDA_ERRORS();
  return IU_SUCCESS;
}

//-----------------------------------------------------------------------------
/** Unregister a buffer object with CUDA
 * @param pbo index of picture buffer object
 */
IuStatus cuPboUnregister(GLuint pbo, bool& registered)
{
  if(registered)
  {
    cudaGLUnregisterBufferObject(pbo);
    registered = false;
  }
  IU_CHECK_CUDA_ERRORS();
  return IU_SUCCESS;
}

//-----------------------------------------------------------------------------
/** Wrapper to copy the corresponding image data to the pbo
 * @param[out] pbo_out index of picture buffer object for output image
 * @param[in] dImage device memory image
 */
IuStatus cuGetOutput(int pbo_out, iu::Image* image,
                      float min_val, float max_val, IuRect roi)
{
  // prepare fragmentation for processing
  const unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(roi.width - roi.x, dimBlock.x),
               iu::divUp(roi.height - roi.y, dimBlock.y));

  int* out_data = NULL;
  cudaGLMapBufferObject((void**)&out_data, pbo_out);

  //  distinguish the different kernels
  if((image->bitDepth() == 32) && (image->nChannels() == 1))
  {
    // 32-bit; 1-channel
    iu::ImageNpp_32f_C1 *src = reinterpret_cast<iu::ImageNpp_32f_C1*>(image);
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float1>();
    cudaBindTexture2D(0, &nppglwidget_image_32f_C1__, src->data(), &channel_desc,
                      src->width(), src->height(), src->pitch());
    getOutputKernel_32f_C1 <<< dimGrid, dimBlock >>> (
        out_data, roi.x, roi.y, roi.width, roi.height, min_val, max_val);
  }
  else if((image->bitDepth() == 32) && (image->nChannels() == 4))
  {
    // 32-bit; 4-channel
    iu::ImageNpp_32f_C4 *src = reinterpret_cast<iu::ImageNpp_32f_C4*>(image);
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
    cudaBindTexture2D(0, &nppglwidget_image_32f_C4__, (float4*)src->data(), &channel_desc,
                      src->width(), src->height(), src->pitch());
    getOutputKernel_32f_C4 <<< dimGrid, dimBlock >>> (
        out_data, roi.x, roi.y, roi.width, roi.height, min_val, max_val);
  }
  else
  {
    fprintf(stderr, "Image format not supported by NppWidget. Not able to paint image into widget.\n");
    cudaGLUnmapBufferObject(pbo_out);
    return NPP_INVALID_INPUT;
  }

  cudaGLUnmapBufferObject(pbo_out);
  IU_CHECK_CUDA_ERRORS();
  return IU_SUCCESS;
}
//
////-----------------------------------------------------------------------------
///** Wrapper to copy the corresponding image data to the pbo
// * @param[out] pbo_out index of picture buffer object for output image
// * @param[in] dImage device memory image
// */
//bool getOutput( int pbo_out, Cuda::DeviceMemory<float, 3>* dImage,
//                                    float min_val, float max_val, Cuda::Size<2> size)
//{
//  int* out_data = NULL;
//  cudaGLMapBufferObject((void**)&out_data, pbo_out);
//
//  // prepare fragmentation for processing
//  unsigned int nb_x = size[0]/BLOCK_SIZE;
//  unsigned int nb_y = size[1]/BLOCK_SIZE;
//  if (nb_x * BLOCK_SIZE < size[0]) nb_x++;
//  if (nb_y * BLOCK_SIZE < size[1]) nb_y++;
//  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
//  dim3 dimGrid(nb_x, nb_y);
//
//  getOutputRGBPlaneKernel <<< dimGrid, dimBlock >>> (
//      out_data, dImage->getBuffer(), size[0], size[1],
//      dImage->stride[0], dImage->stride[1], min_val, max_val);
//
//  cudaGLUnmapBufferObject(pbo_out);
//  VMLIB_CHECK_CUDA_ERROR()
//      return true;
//}
//
////-----------------------------------------------------------------------------
///** Wrapper to copy the corresponding RGB image data to the pbo
// * @param pbo_out index of picture buffer object for output image
// * @param dImage RGB image in device memory
// */
//bool getOutput( int pbo_out, Cuda::DeviceMemory<float3, 2>* dImage,
//                                    float min_val, float max_val, Cuda::Size<2> size)
//{
//  int* out_data = NULL;
//  cudaGLMapBufferObject((void**)&out_data, pbo_out);
//
//  // prepare fragmentation for processing
//  unsigned int nb_x = size[0]/BLOCK_SIZE;
//  unsigned int nb_y = size[1]/BLOCK_SIZE;
//  if (nb_x * BLOCK_SIZE < size[0]) nb_x++;
//  if (nb_y * BLOCK_SIZE < size[1]) nb_y++;
//  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
//  dim3 dimGrid(nb_x, nb_y);
//
//  getOutputRGBKernel <<< dimGrid, dimBlock >>> (
//      out_data, dImage->getBuffer(), size[0], size[1], dImage->stride[0], min_val, max_val);
//
//  cudaGLUnmapBufferObject(pbo_out);
//  VMLIB_CHECK_CUDA_ERROR()
//      return true;
//}
//
////-----------------------------------------------------------------------------
///** Wrapper to copy the corresponding RGBA image data to the pbo
// * @param pbo_out index of picture buffer object for output image
// */
//bool getOutput( int pbo_out, Cuda::DeviceMemory<float4, 2>* dImage,
//                                    float min_val, float max_val, Cuda::Size<2> size)
//{
//  int* out_data = NULL;
//  cudaGLMapBufferObject((void**)&out_data, pbo_out);
//
//  // prepare fragmentation for processing
//  unsigned int nb_x = size[0]/BLOCK_SIZE;
//  unsigned int nb_y = size[1]/BLOCK_SIZE;
//  if (nb_x * BLOCK_SIZE < size[0]) nb_x++;
//  if (nb_y * BLOCK_SIZE < size[1]) nb_y++;
//  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
//  dim3 dimGrid(nb_x, nb_y);
//
//  getOutputRGBAKernel <<< dimGrid, dimBlock >>> (
//      out_data, dImage->getBuffer(), size[0], size[1], dImage->stride[0], min_val, max_val);
//
//  cudaGLUnmapBufferObject(pbo_out);
//  VMLIB_CHECK_CUDA_ERROR()
//      return true;
//}
//
////-----------------------------------------------------------------------------
///** Creates a colored overlay
// * @param pbo            buffer object for output image
// * @param[in] overlay mask that defines overlay
// * @param[in] r red value [0..255]
// * @param[in] g green value [0..255]
// * @param[in] b blue value [0..255]
// * @param[in] a alpha value [0..255]
// */
//bool createOverlayF( int pbo, Cuda::DeviceMemory<float, 2>* overlay,
//                                         int r, int g, int b, int a, float mask_value,
//                                         Cuda::Size<2> size)
//{
//  int* out_data = NULL;
//  cudaGLMapBufferObject((void**)&out_data, pbo);
//  // prepare fragmentation for processing
//  unsigned int nb_x = size[0]/BLOCK_SIZE;
//  unsigned int nb_y = size[1]/BLOCK_SIZE;
//  if (nb_x * BLOCK_SIZE < size[0]) nb_x++;
//  if (nb_y * BLOCK_SIZE < size[1]) nb_y++;
//  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
//  dim3 dimGrid(nb_x, nb_y);
//
//  createOverlayKernelF<<< dimGrid, dimBlock >>>( out_data, overlay->getBuffer(),
//                                                 size[0], size[1], overlay->stride[0],
//                                                 r, g, b, a, mask_value);
//
//  cudaGLUnmapBufferObject(pbo);
//  VMLIB_CHECK_CUDA_ERROR()
//      return true;
//}
//
////-----------------------------------------------------------------------------
//bool createOverlayUC( int pbo, Cuda::DeviceMemory<unsigned char, 2>* overlay,
//                                          int r, int g, int b, int a, unsigned char mask_value,
//                                          Cuda::Size<2> size)
//{
//  int* out_data = NULL;
//  cudaGLMapBufferObject((void**)&out_data, pbo);
//  // prepare fragmentation for processing
//  unsigned int nb_x = size[0]/BLOCK_SIZE;
//  unsigned int nb_y = size[1]/BLOCK_SIZE;
//  if (nb_x * BLOCK_SIZE < size[0]) nb_x++;
//  if (nb_y * BLOCK_SIZE < size[1]) nb_y++;
//  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
//  dim3 dimGrid(nb_x, nb_y);
//
//  createOverlayKernelUC<<< dimGrid, dimBlock >>>( out_data, overlay->getBuffer(),
//                                                  size[0], size[1], overlay->stride[0],
//                                                  r, g, b, a, mask_value);
//
//  cudaGLUnmapBufferObject(pbo);
//  VMLIB_CHECK_CUDA_ERROR()
//      return true;
//}

} // namespace iu

#endif // NPPGLWIDGET_CU
