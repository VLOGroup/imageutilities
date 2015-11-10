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
 * Module      : ImageUtilities
 * Class       : none
 * Language    : C/CUDA
 * Description : Helper functions to bind textures correctly
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#include <cuda_runtime.h>
#include "iudefs.h"

#ifndef IU_BIND_TEXTURES_CUH
#define IU_BIND_TEXTURES_CUH

namespace iu {
#ifdef __CUDACC__ // only include this in cuda files (seen by nvcc)

const cudaChannelFormatDesc chd_float  = cudaCreateChannelDesc<float>();
const cudaChannelFormatDesc chd_float2 = cudaCreateChannelDesc<float2>();
const cudaChannelFormatDesc chd_float4 = cudaCreateChannelDesc<float4>();

//---------------------------------------------------------------------------
// 2D
template<typename DataType>
static inline void unbindTexture(texture<DataType, cudaTextureType2D>& tex)
{
  cudaUnbindTexture(tex);
}

//---------------------------------------------------------------------------
// 2D; 32f_C1
static inline void bindTexture(texture<float, cudaTextureType2D>& tex, const iu::ImageGpu_32f_C1* mem,
                        cudaTextureFilterMode filter_mode=cudaFilterModeLinear)
{
  tex.addressMode[0] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.addressMode[1] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.filterMode = filter_mode;
  tex.normalized = false;
   cudaBindTexture2D(0, tex, mem->data(), chd_float,
                                    mem->width(), mem->height(), mem->pitch());
}

//---------------------------------------------------------------------------
// 2D; 32f_C2
static inline void bindTexture(texture<float2, cudaTextureType2D>& tex, const iu::ImageGpu_32f_C2* mem,
                        cudaTextureFilterMode filter_mode=cudaFilterModeLinear)
{
  tex.addressMode[0] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.addressMode[1] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.filterMode = filter_mode;
  tex.normalized = false;
   cudaBindTexture2D(0, tex, mem->data(), chd_float2,
                                    mem->width(), mem->height(), mem->pitch());
}

//---------------------------------------------------------------------------
// 2D; 32f_C4
static inline void bindTexture(texture<float4, cudaTextureType2D>& tex, const iu::ImageGpu_32f_C4* mem,
                        cudaTextureFilterMode filter_mode=cudaFilterModeLinear)
{
  tex.addressMode[0] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.addressMode[1] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.filterMode = filter_mode;
  tex.normalized = false;
   cudaBindTexture2D(0, tex, mem->data(), chd_float4,
                                    mem->width(), mem->height(), mem->pitch());
}

//---------------------------------------------------------------------------
// 3D-slice; 32f_C1
static inline void bindTexture(texture<float, cudaTextureType2D>& tex, const iu::VolumeGpu_32f_C1* mem, int slice,
                        cudaTextureFilterMode filter_mode=cudaFilterModeLinear)
{
  tex.addressMode[0] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.addressMode[1] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.filterMode = filter_mode;
  tex.normalized = false;
   cudaBindTexture2D(0, tex, mem->data(0,0,slice), chd_float,
                                    mem->width(), mem->height(), mem->pitch());
}

//---------------------------------------------------------------------------
// 3D-slice; 32f_C2
static inline void bindTexture(texture<float2, cudaTextureType2D>& tex, const iu::VolumeGpu_32f_C2* mem, int slice,
                        cudaTextureFilterMode filter_mode=cudaFilterModeLinear)
{
  tex.addressMode[0] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.addressMode[1] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.filterMode = filter_mode;
  tex.normalized = false;
   cudaBindTexture2D(0, tex, mem->data(0,0,slice), chd_float2,
                                    mem->width(), mem->height(), mem->pitch());
}

//---------------------------------------------------------------------------
// 3D-slice; 32f_C4
static inline void bindTexture(texture<float4, cudaTextureType2D>& tex, const iu::VolumeGpu_32f_C4* mem, int slice,
                        cudaTextureFilterMode filter_mode=cudaFilterModeLinear)
{
  tex.addressMode[0] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.addressMode[1] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.filterMode = filter_mode;
  tex.normalized = false;
   cudaBindTexture2D(0, tex, mem->data(0,0,slice), chd_float4,
                                    mem->width(), mem->height(), mem->pitch());
}


#endif // __CUDACC__
} // namespace iu


#endif // IU_BIND_TEXTURES_CUH
