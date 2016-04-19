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
 * Class       :
 * Language    : C++
 * Description : Typedefs for simpler image usage
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUCORE_MEMORYDEFS_H
#define IUCORE_MEMORYDEFS_H

// template includes
#include "coredefs.h"
#include "linearhostmemory.h"
#include "lineardevicememory.h"
#include "image_allocator_cpu.h"
#include "image_cpu.h"
#include "image_allocator_gpu.h"
#include "image_gpu.h"
#include "volume_allocator_cpu.h"
#include "volume_cpu.h"
#include "volume_allocator_gpu.h"
#include "volume_gpu.h"

/* ***************************************************************************
 *  explicit type definitions for template classes
 * ***************************************************************************/

namespace iu {

/* ****************************************************************************
 *  Linear Memory
 * ****************************************************************************/

/*
  Host
*/
// 8-bit
typedef LinearHostMemory<unsigned char> LinearHostMemory_8u_C1;
typedef LinearHostMemory<uchar2> LinearHostMemory_8u_C2;
typedef LinearHostMemory<uchar3> LinearHostMemory_8u_C3;
typedef LinearHostMemory<uchar4> LinearHostMemory_8u_C4;
// 16-bit
typedef LinearHostMemory<unsigned short> LinearHostMemory_16u_C1;
typedef LinearHostMemory<ushort2> LinearHostMemory_16u_C2;
typedef LinearHostMemory<ushort3> LinearHostMemory_16u_C3;
typedef LinearHostMemory<ushort4> LinearHostMemory_16u_C4;
// 32-bit
typedef LinearHostMemory<float> LinearHostMemory_32f_C1;
typedef LinearHostMemory<float2> LinearHostMemory_32f_C2;
typedef LinearHostMemory<float3> LinearHostMemory_32f_C3;
typedef LinearHostMemory<float4> LinearHostMemory_32f_C4;

typedef LinearHostMemory<int> LinearHostMemory_32s_C1;
typedef LinearHostMemory<int2> LinearHostMemory_32s_C2;
typedef LinearHostMemory<int3> LinearHostMemory_32s_C3;
typedef LinearHostMemory<int4> LinearHostMemory_32s_C4;

/*
  Device
*/
// 8-bit
typedef LinearDeviceMemory<unsigned char> LinearDeviceMemory_8u_C1;
typedef LinearDeviceMemory<uchar2> LinearDeviceMemory_8u_C2;
typedef LinearDeviceMemory<uchar3> LinearDeviceMemory_8u_C3;
typedef LinearDeviceMemory<uchar4> LinearDeviceMemory_8u_C4;
// 16-bit
typedef LinearDeviceMemory<unsigned short> LinearDeviceMemory_16u_C1;
typedef LinearDeviceMemory<ushort2> LinearDeviceMemory_16u_C2;
typedef LinearDeviceMemory<ushort3> LinearDeviceMemory_16u_C3;
typedef LinearDeviceMemory<ushort4> LinearDeviceMemory_16u_C4;
// 32-bit
typedef LinearDeviceMemory<float> LinearDeviceMemory_32f_C1;
typedef LinearDeviceMemory<float2> LinearDeviceMemory_32f_C2;
typedef LinearDeviceMemory<float3> LinearDeviceMemory_32f_C3;
typedef LinearDeviceMemory<float4> LinearDeviceMemory_32f_C4;

typedef LinearDeviceMemory<int> LinearDeviceMemory_32s_C1;
typedef LinearDeviceMemory<int2> LinearDeviceMemory_32s_C2;
typedef LinearDeviceMemory<int3> LinearDeviceMemory_32s_C3;
typedef LinearDeviceMemory<int4> LinearDeviceMemory_32s_C4;


/* ****************************************************************************
 *  2d Image Memory
 * ****************************************************************************/

/*
  Host
*/
// Cpu Images; 8u
typedef ImageCpu<unsigned char, iuprivate::ImageAllocatorCpu<unsigned char>, IU_8U_C1> ImageCpu_8u_C1;
typedef ImageCpu<uchar2, iuprivate::ImageAllocatorCpu<uchar2>, IU_8U_C2> ImageCpu_8u_C2;
typedef ImageCpu<uchar3, iuprivate::ImageAllocatorCpu<uchar3>, IU_8U_C3> ImageCpu_8u_C3;
typedef ImageCpu<uchar4, iuprivate::ImageAllocatorCpu<uchar4>, IU_8U_C4> ImageCpu_8u_C4;

// Cpu Images; 16u
typedef ImageCpu<unsigned short, iuprivate::ImageAllocatorCpu<unsigned short>, IU_16U_C1> ImageCpu_16u_C1;
typedef ImageCpu<ushort2, iuprivate::ImageAllocatorCpu<ushort2>, IU_16U_C2> ImageCpu_16u_C2;
typedef ImageCpu<ushort3, iuprivate::ImageAllocatorCpu<ushort3>, IU_16U_C3> ImageCpu_16u_C3;
typedef ImageCpu<ushort4, iuprivate::ImageAllocatorCpu<ushort4>, IU_16U_C4> ImageCpu_16u_C4;

// Cpu Images; 32s
typedef ImageCpu<int, iuprivate::ImageAllocatorCpu<int>, IU_32S_C1> ImageCpu_32s_C1;
typedef ImageCpu<int2, iuprivate::ImageAllocatorCpu<int2>, IU_32S_C2> ImageCpu_32s_C2;
typedef ImageCpu<int3, iuprivate::ImageAllocatorCpu<int3>, IU_32S_C3> ImageCpu_32s_C3;
typedef ImageCpu<int4, iuprivate::ImageAllocatorCpu<int4>, IU_32S_C4> ImageCpu_32s_C4;

// Cpu Images; 32u
typedef ImageCpu<unsigned int, iuprivate::ImageAllocatorCpu<unsigned int>, IU_32U_C1> ImageCpu_32u_C1;
typedef ImageCpu<uint2, iuprivate::ImageAllocatorCpu<uint2>, IU_32U_C2> ImageCpu_32u_C2;
typedef ImageCpu<uint4, iuprivate::ImageAllocatorCpu<uint4>, IU_32U_C4> ImageCpu_32u_C4;


// Cpu Images; 32f
typedef ImageCpu<float, iuprivate::ImageAllocatorCpu<float>, IU_32F_C1> ImageCpu_32f_C1;
typedef ImageCpu<float2, iuprivate::ImageAllocatorCpu<float2>, IU_32F_C2> ImageCpu_32f_C2;
typedef ImageCpu<float3, iuprivate::ImageAllocatorCpu<float3>, IU_32F_C3> ImageCpu_32f_C3;
typedef ImageCpu<float4, iuprivate::ImageAllocatorCpu<float4>, IU_32F_C4> ImageCpu_32f_C4;

/*
  Device
*/
// Gpu Images; 8u
typedef ImageGpu<unsigned char, iuprivate::ImageAllocatorGpu<unsigned char>, IU_8U_C1> ImageGpu_8u_C1;
typedef ImageGpu<uchar2, iuprivate::ImageAllocatorGpu<uchar2>, IU_8U_C2> ImageGpu_8u_C2;
typedef ImageGpu<uchar3, iuprivate::ImageAllocatorGpu<uchar3>, IU_8U_C3> ImageGpu_8u_C3;
typedef ImageGpu<uchar4, iuprivate::ImageAllocatorGpu<uchar4>, IU_8U_C4> ImageGpu_8u_C4;

// Gpu Images; 16u
typedef ImageGpu<unsigned short, iuprivate::ImageAllocatorGpu<unsigned short>, IU_16U_C1> ImageGpu_16u_C1;
typedef ImageGpu<ushort2, iuprivate::ImageAllocatorGpu<ushort2>, IU_16U_C2> ImageGpu_16u_C2;
typedef ImageGpu<ushort3, iuprivate::ImageAllocatorGpu<ushort3>, IU_16U_C3> ImageGpu_16u_C3;
typedef ImageGpu<ushort4, iuprivate::ImageAllocatorGpu<ushort4>, IU_16U_C4> ImageGpu_16u_C4;

// Gpu Images; 32s
typedef ImageGpu<int, iuprivate::ImageAllocatorGpu<int>, IU_32S_C1> ImageGpu_32s_C1;
typedef ImageGpu<int2, iuprivate::ImageAllocatorGpu<int2>, IU_32S_C2> ImageGpu_32s_C2;
typedef ImageGpu<int3, iuprivate::ImageAllocatorGpu<int3>, IU_32S_C3> ImageGpu_32s_C3;
typedef ImageGpu<int4, iuprivate::ImageAllocatorGpu<int4>, IU_32S_C4> ImageGpu_32s_C4;

// Gpu Images; 32u
typedef ImageGpu<unsigned int, iuprivate::ImageAllocatorGpu<unsigned int>, IU_32U_C1> ImageGpu_32u_C1;
typedef ImageGpu<uint2, iuprivate::ImageAllocatorGpu<uint2>, IU_32U_C2> ImageGpu_32u_C2;
typedef ImageGpu<uint4, iuprivate::ImageAllocatorGpu<uint4>, IU_32U_C4> ImageGpu_32u_C4;


// Gpu Images; 32f
typedef ImageGpu<float, iuprivate::ImageAllocatorGpu<float>, IU_32F_C1> ImageGpu_32f_C1;
typedef ImageGpu<float2, iuprivate::ImageAllocatorGpu<float2>, IU_32F_C2> ImageGpu_32f_C2;
typedef ImageGpu<float3, iuprivate::ImageAllocatorGpu<float3>, IU_32F_C3> ImageGpu_32f_C3;
typedef ImageGpu<float4, iuprivate::ImageAllocatorGpu<float4>, IU_32F_C4> ImageGpu_32f_C4;


/* ****************************************************************************
 *  3d Volume Memory
 * ****************************************************************************/

/*
  Host
*/
// Cpu Volumes; 8u
typedef VolumeCpu<unsigned char, iuprivate::VolumeAllocatorCpu<unsigned char>, IU_8U_C1> VolumeCpu_8u_C1;
typedef VolumeCpu<uchar2, iuprivate::VolumeAllocatorCpu<uchar2>, IU_8U_C2> VolumeCpu_8u_C2;
typedef VolumeCpu<uchar3, iuprivate::VolumeAllocatorCpu<uchar3>, IU_8U_C3> VolumeCpu_8u_C3;
typedef VolumeCpu<uchar4, iuprivate::VolumeAllocatorCpu<uchar4>, IU_8U_C4> VolumeCpu_8u_C4;

// Cpu Volumes; 16u
typedef VolumeCpu<unsigned short, iuprivate::VolumeAllocatorCpu<unsigned short>, IU_16U_C1> VolumeCpu_16u_C1;

// Cpu Volumes; 32u
typedef VolumeCpu<unsigned int, iuprivate::VolumeAllocatorCpu<unsigned int>, IU_32U_C1> VolumeCpu_32u_C1;
typedef VolumeCpu<uint2, iuprivate::VolumeAllocatorCpu<uint2>, IU_32U_C2> VolumeCpu_32u_C2;
typedef VolumeCpu<uint4, iuprivate::VolumeAllocatorCpu<uint4>, IU_32U_C4> VolumeCpu_32u_C4;

// Cpu Volumes; 32s
typedef VolumeCpu<int, iuprivate::VolumeAllocatorCpu<int>, IU_32S_C1> VolumeCpu_32s_C1;
typedef VolumeCpu<int2, iuprivate::VolumeAllocatorCpu<int2>, IU_32S_C2> VolumeCpu_32s_C2;
typedef VolumeCpu<int4, iuprivate::VolumeAllocatorCpu<int4>, IU_32S_C4> VolumeCpu_32s_C4;


// Cpu Volumes; 32f
typedef VolumeCpu<float, iuprivate::VolumeAllocatorCpu<float>, IU_32F_C1> VolumeCpu_32f_C1;
typedef VolumeCpu<float2, iuprivate::VolumeAllocatorCpu<float2>, IU_32F_C2> VolumeCpu_32f_C2;
typedef VolumeCpu<float3, iuprivate::VolumeAllocatorCpu<float3>, IU_32F_C3> VolumeCpu_32f_C3;
typedef VolumeCpu<float4, iuprivate::VolumeAllocatorCpu<float4>, IU_32F_C4> VolumeCpu_32f_C4;

/*
  Device
*/
// Gpu Volumes; 8u
typedef VolumeGpu<unsigned char, iuprivate::VolumeAllocatorGpu<unsigned char>, IU_8U_C1> VolumeGpu_8u_C1;
typedef VolumeGpu<uchar2, iuprivate::VolumeAllocatorGpu<uchar2>, IU_8U_C2> VolumeGpu_8u_C2;
typedef VolumeGpu<uchar3, iuprivate::VolumeAllocatorGpu<uchar3>, IU_8U_C3> VolumeGpu_8u_C3;
typedef VolumeGpu<uchar4, iuprivate::VolumeAllocatorGpu<uchar4>, IU_8U_C4> VolumeGpu_8u_C4;

// Gpu Volumes; 16u
typedef VolumeGpu<unsigned short, iuprivate::VolumeAllocatorGpu<unsigned short>, IU_16U_C1> VolumeGpu_16u_C1;

// Gpu Volumes; 32u
typedef VolumeGpu<unsigned int, iuprivate::VolumeAllocatorGpu<unsigned int>, IU_32U_C1> VolumeGpu_32u_C1;
typedef VolumeGpu<uint2, iuprivate::VolumeAllocatorGpu<uint2>, IU_32U_C2> VolumeGpu_32u_C2;
typedef VolumeGpu<uint4, iuprivate::VolumeAllocatorGpu<uint4>, IU_32U_C4> VolumeGpu_32u_C4;

// Gpu Volumes; 32s
typedef VolumeGpu<int, iuprivate::VolumeAllocatorGpu<int>, IU_32S_C1> VolumeGpu_32s_C1;
typedef VolumeGpu<int2, iuprivate::VolumeAllocatorGpu<int2>, IU_32S_C2> VolumeGpu_32s_C2;
typedef VolumeGpu<int4, iuprivate::VolumeAllocatorGpu<int4>, IU_32S_C4> VolumeGpu_32s_C4;


// Gpu Volumes; 32f
typedef VolumeGpu<float, iuprivate::VolumeAllocatorGpu<float>, IU_32F_C1> VolumeGpu_32f_C1;
typedef VolumeGpu<float2, iuprivate::VolumeAllocatorGpu<float2>, IU_32F_C2> VolumeGpu_32f_C2;
typedef VolumeGpu<float3, iuprivate::VolumeAllocatorGpu<float3>, IU_32F_C3> VolumeGpu_32f_C3;
typedef VolumeGpu<float4, iuprivate::VolumeAllocatorGpu<float4>, IU_32F_C4> VolumeGpu_32f_C4;

} // namespace iu

#endif // IUCORE_MEMORYDEFS_H
