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
 * Project     : vmgpu
 * Module      : Tools
 * Class       : NppGLWidget
 * Language    : C++/CUDA
 * Description : Definition of Cuda wrappers for the DeviceGLWidget
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef NPPGLWIDGET_CUH
#define NPPGLWIDGET_CUH

// includes, cuda
#include "iucore/coredefs.h"
#include "iucore/memorydefs.h"


namespace iuprivate
{
  IuStatus cuInitTextures();
  IuStatus cuPboRegister(GLuint pbo, bool& registered);
  IuStatus cuPboUnregister(GLuint pbo, bool& registered);
  // bool pboConnect(GLuint pbo, int* buffer, bool& registered);
  // bool pboDisconnect();

  /** Wrapper to copy the corresponding RGB image data to the pbo
   * @param pbo_out index of picture buffer object for output image.
   * @param dImage Grayscale image in device memory
   * @param min_val minimum value (will be mapped to 0)
   * @param max_val maximum value (will be mapped to 1)
   * @param size size of area to be drawn
   */
  IuStatus cuGetOutput( int pbo_dest, iu::Image* image,
                         float min_val, float max_val, IuRect size);

  //  /** Wrapper to copy the corresponding RGB image data to the pbo
  //   * @param pbo_out index of picture buffer object for output image.
  //   * @param dImage Color image in device memory
  //   * @param min_val minimum value (will be mapped to 0)
  //   * @param max_val maximum value (will be mapped to 1)
  //   * @param size size of area to be drawn
//   */
//  bool getOutput( int pbo_dest, Cuda::DeviceMemory<float, 3>* dImage, float min_val,
//                  float max_val, Cuda::Size<2> size);
//
//  /** Wrapper to copy the corresponding RGB image data to the pbo
//   * @param pbo_out index of picture buffer object for output image
//   * @param dImage RGB image in device memory (val range = [0..1])
//   * @param size size of area to be drawn
//   */
//  bool getOutput(int pbo_dest, Cuda::DeviceMemory<float3, 2>* dImage, float min_val,
//                 float max_val, Cuda::Size<2> size);
//
//  /** Wrapper to copy the corresponding RGB image data to the pbo
//   * @param pbo_out index of picture buffer object for output image
//   * @param dImage RGBA image in device memory (val range = [0..1])
//   * @param size size of area to be drawn
//   */
//  bool getOutput(int pbo_dest, Cuda::DeviceMemory<float4, 2>* dImage, float min_val,
//                 float max_val, Cuda::Size<2> size);

//  /** Creates a colored overlay
//   * @param pbo buffer object for output image
//   * @param[in] overlay mask that defines overlay
//   * @param[in] r red value [0..255]
//   * @param[in] g green value [0..255]
//   * @param[in] b blue value [0..255]
//   * @param[in] a alpha value [0..255]
//   * @param size size of area to be drawn
//   */
//  bool createOverlayF( int pbo, Cuda::DeviceMemory<float, 2>* overlay,
//                       int r, int g, int b, int a, float mask_value, Cuda::Size<2> size);
//  bool createOverlayUC( int pbo, Cuda::DeviceMemory<unsigned char, 2>* overlay,
//                        int r, int g, int b, int a, unsigned char mask_value, Cuda::Size<2> size);
}

#endif // NPPGLWIDGET_CUH
