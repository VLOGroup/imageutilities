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

#include "tensor_cpu.h"
#include "tensor_gpu.h"

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

typedef LinearHostMemory<unsigned int> LinearHostMemory_32u_C1;
typedef LinearHostMemory<uint2> LinearHostMemory_32u_C2;
typedef LinearHostMemory<uint4> LinearHostMemory_32u_C4;


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

typedef LinearDeviceMemory<unsigned int> LinearDeviceMemory_32u_C1;
typedef LinearDeviceMemory<uint2> LinearDeviceMemory_32u_C2;
typedef LinearDeviceMemory<uint4> LinearDeviceMemory_32u_C4;



/* ****************************************************************************
 *  2d Image Memory
 * ****************************************************************************/

/*
  Host
*/
// Cpu Images; 8u
typedef ImageCpu<unsigned char, ImageAllocatorCpu<unsigned char> > ImageCpu_8u_C1;
typedef ImageCpu<uchar2, ImageAllocatorCpu<uchar2> > ImageCpu_8u_C2;
typedef ImageCpu<uchar3, ImageAllocatorCpu<uchar3> > ImageCpu_8u_C3;
typedef ImageCpu<uchar4, ImageAllocatorCpu<uchar4> > ImageCpu_8u_C4;

// Cpu Images; 16u
typedef ImageCpu<unsigned short, ImageAllocatorCpu<unsigned short> > ImageCpu_16u_C1;
typedef ImageCpu<ushort2, ImageAllocatorCpu<ushort2> > ImageCpu_16u_C2;
typedef ImageCpu<ushort3, ImageAllocatorCpu<ushort3> > ImageCpu_16u_C3;
typedef ImageCpu<ushort4, ImageAllocatorCpu<ushort4> > ImageCpu_16u_C4;

// Cpu Images; 32s
typedef ImageCpu<int, ImageAllocatorCpu<int> > ImageCpu_32s_C1;
typedef ImageCpu<int2, ImageAllocatorCpu<int2> > ImageCpu_32s_C2;
typedef ImageCpu<int3, ImageAllocatorCpu<int3> > ImageCpu_32s_C3;
typedef ImageCpu<int4, ImageAllocatorCpu<int4> > ImageCpu_32s_C4;

// Cpu Images; 32u
typedef ImageCpu<unsigned int, ImageAllocatorCpu<unsigned int> > ImageCpu_32u_C1;
typedef ImageCpu<uint2, ImageAllocatorCpu<uint2> > ImageCpu_32u_C2;
typedef ImageCpu<uint4, ImageAllocatorCpu<uint4> > ImageCpu_32u_C4;


// Cpu Images; 32f
typedef ImageCpu<float, ImageAllocatorCpu<float> > ImageCpu_32f_C1;
typedef ImageCpu<float2, ImageAllocatorCpu<float2> > ImageCpu_32f_C2;
typedef ImageCpu<float3, ImageAllocatorCpu<float3> > ImageCpu_32f_C3;
typedef ImageCpu<float4, ImageAllocatorCpu<float4> > ImageCpu_32f_C4;

/*
  Device
*/
// Gpu Images; 8u
typedef ImageGpu<unsigned char, ImageAllocatorGpu<unsigned char> > ImageGpu_8u_C1;
typedef ImageGpu<uchar2, ImageAllocatorGpu<uchar2> > ImageGpu_8u_C2;
typedef ImageGpu<uchar3, ImageAllocatorGpu<uchar3> > ImageGpu_8u_C3;
typedef ImageGpu<uchar4, ImageAllocatorGpu<uchar4> > ImageGpu_8u_C4;

// Gpu Images; 16u
typedef ImageGpu<unsigned short, ImageAllocatorGpu<unsigned short> > ImageGpu_16u_C1;
typedef ImageGpu<ushort2, ImageAllocatorGpu<ushort2> > ImageGpu_16u_C2;
typedef ImageGpu<ushort3, ImageAllocatorGpu<ushort3> > ImageGpu_16u_C3;
typedef ImageGpu<ushort4, ImageAllocatorGpu<ushort4> > ImageGpu_16u_C4;

// Gpu Images; 32s
typedef ImageGpu<int, ImageAllocatorGpu<int> > ImageGpu_32s_C1;
typedef ImageGpu<int2, ImageAllocatorGpu<int2> > ImageGpu_32s_C2;
typedef ImageGpu<int3, ImageAllocatorGpu<int3> > ImageGpu_32s_C3;
typedef ImageGpu<int4, ImageAllocatorGpu<int4> > ImageGpu_32s_C4;

// Gpu Images; 32u
typedef ImageGpu<unsigned int, ImageAllocatorGpu<unsigned int> > ImageGpu_32u_C1;
typedef ImageGpu<uint2, ImageAllocatorGpu<uint2> > ImageGpu_32u_C2;
typedef ImageGpu<uint4, ImageAllocatorGpu<uint4> > ImageGpu_32u_C4;


// Gpu Images; 32f
typedef ImageGpu<float, ImageAllocatorGpu<float> > ImageGpu_32f_C1;
typedef ImageGpu<float2, ImageAllocatorGpu<float2> > ImageGpu_32f_C2;
typedef ImageGpu<float3, ImageAllocatorGpu<float3> > ImageGpu_32f_C3;
typedef ImageGpu<float4, ImageAllocatorGpu<float4> > ImageGpu_32f_C4;


/* ****************************************************************************
 *  3d Volume Memory
 * ****************************************************************************/

/*
  Host
*/
// Cpu Volumes; 8u
typedef VolumeCpu<unsigned char, VolumeAllocatorCpu<unsigned char> > VolumeCpu_8u_C1;
typedef VolumeCpu<uchar2, VolumeAllocatorCpu<uchar2> > VolumeCpu_8u_C2;
typedef VolumeCpu<uchar3, VolumeAllocatorCpu<uchar3> > VolumeCpu_8u_C3;
typedef VolumeCpu<uchar4, VolumeAllocatorCpu<uchar4> > VolumeCpu_8u_C4;

// Cpu Volumes; 16u
typedef VolumeCpu<unsigned short, VolumeAllocatorCpu<unsigned short> > VolumeCpu_16u_C1;

// Cpu Volumes; 32u
typedef VolumeCpu<unsigned int, VolumeAllocatorCpu<unsigned int> > VolumeCpu_32u_C1;
typedef VolumeCpu<uint2, VolumeAllocatorCpu<uint2> > VolumeCpu_32u_C2;
typedef VolumeCpu<uint4, VolumeAllocatorCpu<uint4> > VolumeCpu_32u_C4;

// Cpu Volumes; 32s
typedef VolumeCpu<int, VolumeAllocatorCpu<int> > VolumeCpu_32s_C1;
typedef VolumeCpu<int2, VolumeAllocatorCpu<int2> > VolumeCpu_32s_C2;
typedef VolumeCpu<int4, VolumeAllocatorCpu<int4> > VolumeCpu_32s_C4;


// Cpu Volumes; 32f
typedef VolumeCpu<float, VolumeAllocatorCpu<float> > VolumeCpu_32f_C1;
typedef VolumeCpu<float2, VolumeAllocatorCpu<float2> > VolumeCpu_32f_C2;
typedef VolumeCpu<float3, VolumeAllocatorCpu<float3> > VolumeCpu_32f_C3;
typedef VolumeCpu<float4, VolumeAllocatorCpu<float4> > VolumeCpu_32f_C4;

/*
  Device
*/
// Gpu Volumes; 8u
typedef VolumeGpu<unsigned char, VolumeAllocatorGpu<unsigned char> > VolumeGpu_8u_C1;
typedef VolumeGpu<uchar2, VolumeAllocatorGpu<uchar2> > VolumeGpu_8u_C2;
typedef VolumeGpu<uchar3, VolumeAllocatorGpu<uchar3> > VolumeGpu_8u_C3;
typedef VolumeGpu<uchar4, VolumeAllocatorGpu<uchar4> > VolumeGpu_8u_C4;

// Gpu Volumes; 16u
typedef VolumeGpu<unsigned short, VolumeAllocatorGpu<unsigned short> > VolumeGpu_16u_C1;

// Gpu Volumes; 32u
typedef VolumeGpu<unsigned int, VolumeAllocatorGpu<unsigned int> > VolumeGpu_32u_C1;
typedef VolumeGpu<uint2, VolumeAllocatorGpu<uint2> > VolumeGpu_32u_C2;
typedef VolumeGpu<uint4, VolumeAllocatorGpu<uint4> > VolumeGpu_32u_C4;

// Gpu Volumes; 32s
typedef VolumeGpu<int, VolumeAllocatorGpu<int> > VolumeGpu_32s_C1;
typedef VolumeGpu<int2, VolumeAllocatorGpu<int2> > VolumeGpu_32s_C2;
typedef VolumeGpu<int4, VolumeAllocatorGpu<int4> > VolumeGpu_32s_C4;


// Gpu Volumes; 32f
typedef VolumeGpu<float, VolumeAllocatorGpu<float> > VolumeGpu_32f_C1;
typedef VolumeGpu<float2, VolumeAllocatorGpu<float2> > VolumeGpu_32f_C2;
typedef VolumeGpu<float3, VolumeAllocatorGpu<float3> > VolumeGpu_32f_C3;
typedef VolumeGpu<float4, VolumeAllocatorGpu<float4> > VolumeGpu_32f_C4;


/* ****************************************************************************
 *  4D Tensor
 * ****************************************************************************/

/*
  Host
*/
typedef TensorCpu<unsigned char> TensorCpu_8u;
typedef TensorCpu<unsigned short> TensorCpu_16u;
typedef TensorCpu<float> TensorCpu_32f;
typedef TensorCpu<int> TensorCpu_32s;
typedef TensorCpu<double> TensorCpu_64f;

/*
  Device
*/
// 8-bit
typedef TensorGpu<unsigned char> TensorGpu_8u;
typedef TensorGpu<unsigned short> TensorGpu_16u;
typedef TensorGpu<float> TensorGpu_32f;
typedef TensorGpu<unsigned int> TensorGpu_32u;
typedef TensorGpu<int> TensorGpu_32s;
typedef TensorGpu<double> TensorGpu_64f;


/* ****************************************************************************
 *  Define size checks
 * ****************************************************************************/
static inline void checkSize(const iu::Volume *volume1, const iu::Volume *volume2,
               const char* file, const char* function, const int line)
{
  if (volume1->size() != volume2->size())
  {
    std::stringstream msg;
    msg << "Size mismatch! Size of first Volume is " << volume1->size() << ". ";
    msg << "Size of second Volume is " << volume2->size();
    throw IuException(msg.str(), file, function, line);
  }
}

static inline void checkSize(const iu::Image *image1, const iu::Image *image2,
               const char* file, const char* function, const int line)
{
  if (image1->size() != image2->size())
  {
    std::stringstream msg;
    msg << "Size mismatch! Size of first Image is " << image1->size() << ". ";
    msg << "Size of second Image is " << image2->size();
    throw IuException(msg.str(), file, function, line);
  }
}

static inline void checkSize(const iu::LinearMemory *linmem1, const iu::LinearMemory *linmem2,
               const char* file, const char* function, const int line)
{
  if (linmem1->length() != linmem2->length())
  {
    std::stringstream msg;
    msg << "Size mismatch! Length of first LinearMemory is " << linmem1->length() << ". ";
    msg << "Length of second LinearMemory is " << linmem2->length();
    throw IuException(msg.str(), file, function, line);
  }
}

static inline void checkSize(const iu::Image *image, const iu::LinearMemory *linmem,
               const char* file, const char* function, const int line)
{
  if (image->size().width*image->size().height != linmem->length())
  {
    std::stringstream msg;
    msg << "Size mismatch! Number of elements in Image is " << image->size().width*image->size().height << ". ";
    msg << "Length of LinearMemory is " << linmem->length();
    throw IuException(msg.str(), file, function, line);
  }
}

#define IU_SIZE_CHECK(variable1, variable2)  checkSize(variable1, variable2, __FILE__, __FUNCTION__, __LINE__)
} // namespace iu

#endif // IUCORE_MEMORYDEFS_H
