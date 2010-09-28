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
#include "image_allocator_npp.h"
#include "image_npp.h"
#include "volume_allocator_cpu.h"
#include "volume_cpu.h"
#include "volume_allocator_gpu.h"
#include "volume_gpu.h"

/* ***************************************************************************
   explicit type definitions for template classes
 * ***************************************************************************/
namespace iu {

  // Linear Host memory; 8-bit
  typedef LinearHostMemory<Npp8u> LinearHostMemory_8u;
  typedef LinearHostMemory<unsigned char> LinearHostMemory_8u_C1;
  typedef LinearHostMemory<uchar2> LinearHostMemory_8u_C2;
  typedef LinearHostMemory<uchar3> LinearHostMemory_8u_C3;
  typedef LinearHostMemory<uchar4> LinearHostMemory_8u_C4;

  // Linear Host memory; 32-bit
  typedef LinearHostMemory<Npp32f> LinearHostMemory_32f;
  typedef LinearHostMemory<float> LinearHostMemory_32f_C1;
  typedef LinearHostMemory<float2> LinearHostMemory_32f_C2;
  typedef LinearHostMemory<float3> LinearHostMemory_32f_C3;
  typedef LinearHostMemory<float4> LinearHostMemory_32f_C4;

  // Linear device memory; 8-bit
  typedef LinearDeviceMemory<Npp8u> LinearDeviceMemory_8u;
  typedef LinearDeviceMemory<unsigned char> LinearDeviceMemory_8u_C1;
  typedef LinearDeviceMemory<uchar2> LinearDeviceMemory_8u_C2;
  typedef LinearDeviceMemory<uchar3> LinearDeviceMemory_8u_C3;
  typedef LinearDeviceMemory<uchar4> LinearDeviceMemory_8u_C4;

  // Linear device memory; 32-bit
  typedef LinearDeviceMemory<Npp32f> LinearDeviceMemory_32f;
  typedef LinearDeviceMemory<float> LinearDeviceMemory_32f_C1;
  typedef LinearDeviceMemory<float2> LinearDeviceMemory_32f_C2;
  typedef LinearDeviceMemory<float3> LinearDeviceMemory_32f_C3;
  typedef LinearDeviceMemory<float4> LinearDeviceMemory_32f_C4;

  // Cpu Images; 8-bit
  typedef ImageCpu<Npp8u, 1, iuprivate::ImageAllocatorCpu<Npp8u, 1> > ImageCpu_8u_C1;
  typedef ImageCpu<Npp8u, 2, iuprivate::ImageAllocatorCpu<Npp8u, 2> > ImageCpu_8u_C2;
  typedef ImageCpu<Npp8u, 3, iuprivate::ImageAllocatorCpu<Npp8u, 3> > ImageCpu_8u_C3;
  typedef ImageCpu<Npp8u, 4, iuprivate::ImageAllocatorCpu<Npp8u, 4> > ImageCpu_8u_C4;

  // Cpu Images; 16-bit
  typedef ImageCpu<Npp16u, 1, iuprivate::ImageAllocatorCpu<Npp16u, 1> > ImageCpu_16u_C1;
  typedef ImageCpu<Npp16u, 2, iuprivate::ImageAllocatorCpu<Npp16u, 2> > ImageCpu_16u_C2;
  typedef ImageCpu<Npp16u, 3, iuprivate::ImageAllocatorCpu<Npp16u, 3> > ImageCpu_16u_C3;
  typedef ImageCpu<Npp16u, 4, iuprivate::ImageAllocatorCpu<Npp16u, 4> > ImageCpu_16u_C4;

  // Cpu Images; 32-bit
  typedef ImageCpu<Npp32f, 1, iuprivate::ImageAllocatorCpu<Npp32f, 1> > ImageCpu_32f_C1;
  typedef ImageCpu<Npp32f, 2, iuprivate::ImageAllocatorCpu<Npp32f, 2> > ImageCpu_32f_C2;
  typedef ImageCpu<Npp32f, 3, iuprivate::ImageAllocatorCpu<Npp32f, 3> > ImageCpu_32f_C3;
  typedef ImageCpu<Npp32f, 4, iuprivate::ImageAllocatorCpu<Npp32f, 4> > ImageCpu_32f_C4;

  // Npp Images; 8-bit
  typedef ImageNpp<Npp8u, 1, iuprivate::ImageAllocatorNpp<Npp8u, 1> > ImageNpp_8u_C1;
  typedef ImageNpp<Npp8u, 2, iuprivate::ImageAllocatorNpp<Npp8u, 2> > ImageNpp_8u_C2;
  typedef ImageNpp<Npp8u, 3, iuprivate::ImageAllocatorNpp<Npp8u, 3> > ImageNpp_8u_C3;
  typedef ImageNpp<Npp8u, 4, iuprivate::ImageAllocatorNpp<Npp8u, 4> > ImageNpp_8u_C4;

  // Npp Images; 16-bit
  typedef ImageNpp<Npp16u, 1, iuprivate::ImageAllocatorNpp<Npp16u, 1> > ImageNpp_16u_C1;
  typedef ImageNpp<Npp16u, 2, iuprivate::ImageAllocatorNpp<Npp16u, 2> > ImageNpp_16u_C2;
  typedef ImageNpp<Npp16u, 3, iuprivate::ImageAllocatorNpp<Npp16u, 3> > ImageNpp_16u_C3;
  typedef ImageNpp<Npp16u, 4, iuprivate::ImageAllocatorNpp<Npp16u, 4> > ImageNpp_16u_C4;

  // Npp Images; 32-bit
  typedef ImageNpp<Npp32f, 1, iuprivate::ImageAllocatorNpp<Npp32f, 1> > ImageNpp_32f_C1;
  typedef ImageNpp<Npp32f, 2, iuprivate::ImageAllocatorNpp<Npp32f, 2> > ImageNpp_32f_C2;
  typedef ImageNpp<Npp32f, 3, iuprivate::ImageAllocatorNpp<Npp32f, 3> > ImageNpp_32f_C3;
  typedef ImageNpp<Npp32f, 4, iuprivate::ImageAllocatorNpp<Npp32f, 4> > ImageNpp_32f_C4;

  // Cpu Volumes; 8-bit
  typedef VolumeCpu<unsigned char, iuprivate::VolumeAllocatorCpu<unsigned char> > VolumeCpu_8u_C1;
  typedef VolumeCpu<uchar2, iuprivate::VolumeAllocatorCpu<uchar2> > VolumeCpu_8u_C2;
  typedef VolumeCpu<uchar4, iuprivate::VolumeAllocatorCpu<uchar4> > VolumeCpu_8u_C4;

  // Cpu Volumes; 32-bit
  typedef VolumeCpu<float, iuprivate::VolumeAllocatorCpu<float> > VolumeCpu_32f_C1;
  typedef VolumeCpu<float2, iuprivate::VolumeAllocatorCpu<float2> > VolumeCpu_32f_C2;
  typedef VolumeCpu<float4, iuprivate::VolumeAllocatorCpu<float4> > VolumeCpu_32f_C4;

  // Gpu Volumes; 8-bit
  typedef VolumeGpu<unsigned char, iuprivate::VolumeAllocatorGpu<unsigned char> > VolumeGpu_8u_C1;
  typedef VolumeGpu<uchar2, iuprivate::VolumeAllocatorGpu<uchar2> > VolumeGpu_8u_C2;
  typedef VolumeGpu<uchar4, iuprivate::VolumeAllocatorGpu<uchar4> > VolumeGpu_8u_C4;

  // Gpu Volumes; 32-bit
  typedef VolumeGpu<float, iuprivate::VolumeAllocatorGpu<float> > VolumeGpu_32f_C1;
  typedef VolumeGpu<float2, iuprivate::VolumeAllocatorGpu<float2> > VolumeGpu_32f_C2;
  typedef VolumeGpu<float4, iuprivate::VolumeAllocatorGpu<float4> > VolumeGpu_32f_C4;

} // namespace iu

#endif // IUCORE_MEMORYDEFS_H
