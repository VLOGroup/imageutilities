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
#include "imagepyramid.h"

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


/* ****************************************************************************
 *  2d Image Memory
 * ****************************************************************************/

/*
  Host
*/
// Cpu Images; 8-bit
typedef ImageCpu<unsigned char, iuprivate::ImageAllocatorCpu<unsigned char> > ImageCpu_8u_C1;
typedef ImageCpu<uchar2, iuprivate::ImageAllocatorCpu<uchar2> > ImageCpu_8u_C2;
typedef ImageCpu<uchar3, iuprivate::ImageAllocatorCpu<uchar3> > ImageCpu_8u_C3;
typedef ImageCpu<uchar4, iuprivate::ImageAllocatorCpu<uchar4> > ImageCpu_8u_C4;
// Cpu Images; 16-bit
typedef ImageCpu<unsigned short, iuprivate::ImageAllocatorCpu<unsigned short> > ImageCpu_16u_C1;
typedef ImageCpu<ushort2, iuprivate::ImageAllocatorCpu<ushort2> > ImageCpu_16u_C2;
typedef ImageCpu<ushort3, iuprivate::ImageAllocatorCpu<ushort3> > ImageCpu_16u_C3;
typedef ImageCpu<ushort4, iuprivate::ImageAllocatorCpu<ushort4> > ImageCpu_16u_C4;
// Cpu Images; 32-bit
typedef ImageCpu<float, iuprivate::ImageAllocatorCpu<float> > ImageCpu_32f_C1;
typedef ImageCpu<float2, iuprivate::ImageAllocatorCpu<float2> > ImageCpu_32f_C2;
typedef ImageCpu<float3, iuprivate::ImageAllocatorCpu<float3> > ImageCpu_32f_C3;
typedef ImageCpu<float4, iuprivate::ImageAllocatorCpu<float4> > ImageCpu_32f_C4;
typedef ImageCpu<int, iuprivate::ImageAllocatorCpu<int> > ImageCpu_32s_C1;
typedef ImageCpu<int2, iuprivate::ImageAllocatorCpu<int2> > ImageCpu_32s_C2;
typedef ImageCpu<int3, iuprivate::ImageAllocatorCpu<int3> > ImageCpu_32s_C3;
typedef ImageCpu<int4, iuprivate::ImageAllocatorCpu<int4> > ImageCpu_32s_C4;

/*
  Device
*/
// Gpu Images; 8-bit
typedef ImageGpu<unsigned char, iuprivate::ImageAllocatorGpu<unsigned char> > ImageGpu_8u_C1;
typedef ImageGpu<uchar2, iuprivate::ImageAllocatorGpu<uchar2> > ImageGpu_8u_C2;
typedef ImageGpu<uchar3, iuprivate::ImageAllocatorGpu<uchar3> > ImageGpu_8u_C3;
typedef ImageGpu<uchar4, iuprivate::ImageAllocatorGpu<uchar4> > ImageGpu_8u_C4;

// Gpu Images; 16-bit
typedef ImageGpu<unsigned short, iuprivate::ImageAllocatorGpu<unsigned short> > ImageGpu_16u_C1;
typedef ImageGpu<ushort2, iuprivate::ImageAllocatorGpu<ushort2> > ImageGpu_16u_C2;
typedef ImageGpu<ushort3, iuprivate::ImageAllocatorGpu<ushort3> > ImageGpu_16u_C3;
typedef ImageGpu<ushort4, iuprivate::ImageAllocatorGpu<ushort4> > ImageGpu_16u_C4;

// Gpu Images; 32-bit
typedef ImageGpu<float, iuprivate::ImageAllocatorGpu<float> > ImageGpu_32f_C1;
typedef ImageGpu<float2, iuprivate::ImageAllocatorGpu<float2> > ImageGpu_32f_C2;
typedef ImageGpu<float3, iuprivate::ImageAllocatorGpu<float3> > ImageGpu_32f_C3;
typedef ImageGpu<float4, iuprivate::ImageAllocatorGpu<float4> > ImageGpu_32f_C4;
typedef ImageGpu<int, iuprivate::ImageAllocatorGpu<int> > ImageGpu_32s_C1;
typedef ImageGpu<int2, iuprivate::ImageAllocatorGpu<int2> > ImageGpu_32s_C2;
typedef ImageGpu<int3, iuprivate::ImageAllocatorGpu<int3> > ImageGpu_32s_C3;
typedef ImageGpu<int4, iuprivate::ImageAllocatorGpu<int4> > ImageGpu_32s_C4;


/* ****************************************************************************
 *  3d Volume Memory
 * ****************************************************************************/

/*
  Host
*/
// Cpu Volumes; 8-bit
typedef VolumeCpu<unsigned char, iuprivate::VolumeAllocatorCpu<unsigned char> > VolumeCpu_8u_C1;
typedef VolumeCpu<uchar2, iuprivate::VolumeAllocatorCpu<uchar2> > VolumeCpu_8u_C2;
typedef VolumeCpu<uchar4, iuprivate::VolumeAllocatorCpu<uchar4> > VolumeCpu_8u_C4;
// Cpu Volumes; 32-bit
typedef VolumeCpu<float, iuprivate::VolumeAllocatorCpu<float> > VolumeCpu_32f_C1;
typedef VolumeCpu<float2, iuprivate::VolumeAllocatorCpu<float2> > VolumeCpu_32f_C2;
typedef VolumeCpu<float4, iuprivate::VolumeAllocatorCpu<float4> > VolumeCpu_32f_C4;

/*
  Device
*/
// Gpu Volumes; 8-bit
typedef VolumeGpu<unsigned char, iuprivate::VolumeAllocatorGpu<unsigned char> > VolumeGpu_8u_C1;
typedef VolumeGpu<uchar2, iuprivate::VolumeAllocatorGpu<uchar2> > VolumeGpu_8u_C2;
typedef VolumeGpu<uchar4, iuprivate::VolumeAllocatorGpu<uchar4> > VolumeGpu_8u_C4;

// Gpu Volumes; 32-bit
typedef VolumeGpu<float, iuprivate::VolumeAllocatorGpu<float> > VolumeGpu_32f_C1;
typedef VolumeGpu<float2, iuprivate::VolumeAllocatorGpu<float2> > VolumeGpu_32f_C2;
typedef VolumeGpu<float4, iuprivate::VolumeAllocatorGpu<float4> > VolumeGpu_32f_C4;


/* ****************************************************************************
 *  Image Pyramid
 * ****************************************************************************/

/*
  Device
*/
// Gpu Pyramid; 8-bit
typedef ImagePyramid<iu::ImageGpu_8u_C1>  ImagePyramid_8u_C1;
typedef ImagePyramid<iu::ImageGpu_8u_C2>  ImagePyramid_8u_C2;
typedef ImagePyramid<iu::ImageGpu_8u_C4>  ImagePyramid_8u_C4;

// Gpu Pyramid; 32-bit
typedef ImagePyramid<iu::ImageGpu_32f_C1>  ImagePyramid_32f_C1;
typedef ImagePyramid<iu::ImageGpu_32f_C2>  ImagePyramid_32f_C2;
typedef ImagePyramid<iu::ImageGpu_32f_C4>  ImagePyramid_32f_C4;


} // namespace iu

#endif // IUCORE_MEMORYDEFS_H
