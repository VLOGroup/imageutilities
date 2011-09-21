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
#include <cutil_inline.h>
#include "iudefs.h"

#ifndef IU_BIND_TEXTURES_CUH
#define IU_BIND_TEXTURES_CUH

namespace iu {
#ifdef __CUDACC__ // only include this in cuda files (seen by nvcc)

static const cudaChannelFormatDesc chd_float  = cudaCreateChannelDesc<float>();
static const cudaChannelFormatDesc chd_float2 = cudaCreateChannelDesc<float2>();
static const cudaChannelFormatDesc chd_float4 = cudaCreateChannelDesc<float4>();

//---------------------------------------------------------------------------
// 2D; 32f_C1
inline void bindTexture(texture<float, 2>& tex, iu::ImageGpu_32f_C1* mem,
                        cudaTextureFilterMode filter_mode=cudaFilterModeLinear)
{
  tex.addressMode[0] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.addressMode[1] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.filterMode = filter_mode;
  tex.normalized = false;
  CUDA_SAFE_CALL( cudaBindTexture2D(0, tex, mem->data(), chd_float,
                                    mem->width(), mem->height(), mem->pitch()));
}

//---------------------------------------------------------------------------
// 2D; 32f_C2
inline void bindTexture(texture<float2, 2>& tex, iu::ImageGpu_32f_C2* mem,
                        cudaTextureFilterMode filter_mode=cudaFilterModeLinear)
{
  tex.addressMode[0] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.addressMode[1] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.filterMode = filter_mode;
  tex.normalized = false;
  CUDA_SAFE_CALL( cudaBindTexture2D(0, tex, mem->data(), chd_float2,
                                    mem->width(), mem->height(), mem->pitch()));
}

//---------------------------------------------------------------------------
// 2D; 32f_C4
inline void bindTexture(texture<float4, 2>& tex, iu::ImageGpu_32f_C4* mem,
                        cudaTextureFilterMode filter_mode=cudaFilterModeLinear)
{
  tex.addressMode[0] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.addressMode[1] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.filterMode = filter_mode;
  tex.normalized = false;
  CUDA_SAFE_CALL( cudaBindTexture2D(0, tex, mem->data(), chd_float4,
                                    mem->width(), mem->height(), mem->pitch()));
}

//---------------------------------------------------------------------------
// 3D-slice; 32f_C1
inline void bindTexture(texture<float, 2>& tex, iu::VolumeGpu_32f_C1* mem, int slice,
                        cudaTextureFilterMode filter_mode=cudaFilterModeLinear)
{
  tex.addressMode[0] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.addressMode[1] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.filterMode = filter_mode;
  tex.normalized = false;
  CUDA_SAFE_CALL( cudaBindTexture2D(0, tex, mem->data(0,0,slice), chd_float,
                                    mem->width(), mem->height(), mem->pitch()));
}

//---------------------------------------------------------------------------
// 3D-slice; 32f_C2
inline void bindTexture(texture<float2, 2>& tex, iu::VolumeGpu_32f_C2* mem, int slice,
                        cudaTextureFilterMode filter_mode=cudaFilterModeLinear)
{
  tex.addressMode[0] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.addressMode[1] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.filterMode = filter_mode;
  tex.normalized = false;
  CUDA_SAFE_CALL( cudaBindTexture2D(0, tex, mem->data(0,0,slice), chd_float2,
                                    mem->width(), mem->height(), mem->pitch()));
}

//---------------------------------------------------------------------------
// 3D-slice; 32f_C4
inline void bindTexture(texture<float4, 2>& tex, iu::VolumeGpu_32f_C4* mem, int slice,
                        cudaTextureFilterMode filter_mode=cudaFilterModeLinear)
{
  tex.addressMode[0] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.addressMode[1] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.filterMode = filter_mode;
  tex.normalized = false;
  CUDA_SAFE_CALL( cudaBindTexture2D(0, tex, mem->data(0,0,slice), chd_float4,
                                    mem->width(), mem->height(), mem->pitch()));
}


#endif // __CUDACC__
} // namespace iu


#endif // IU_BIND_TEXTURES_CUH
