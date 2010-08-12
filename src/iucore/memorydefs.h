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

/* ***************************************************************************
   explicit type definitions for template classes
 * ***************************************************************************/
namespace iu {
  // Linear host memory
  typedef LinearHostMemory<Npp8u> LinearHostMemory_8u;
  typedef LinearHostMemory<Npp32f> LinearHostMemory_32f;
  // Linear device memory
  typedef LinearDeviceMemory<Npp8u> LinearDeviceMemory_8u;
  typedef LinearDeviceMemory<Npp32f> LinearDeviceMemory_32f;
  // Cpu Images; 8-bit
  typedef ImageCpu<Npp8u, 1, iuprivate::ImageAllocatorCpu<Npp8u, 1> > ImageCpu_8u_C1;
  typedef ImageCpu<Npp8u, 2, iuprivate::ImageAllocatorCpu<Npp8u, 2> > ImageCpu_8u_C2;
  typedef ImageCpu<Npp8u, 3, iuprivate::ImageAllocatorCpu<Npp8u, 3> > ImageCpu_8u_C3;
  typedef ImageCpu<Npp8u, 4, iuprivate::ImageAllocatorCpu<Npp8u, 4> > ImageCpu_8u_C4;
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
  // Npp Images; 32-bit
  typedef ImageNpp<Npp32f, 1, iuprivate::ImageAllocatorNpp<Npp32f, 1> > ImageNpp_32f_C1;
  typedef ImageNpp<Npp32f, 2, iuprivate::ImageAllocatorNpp<Npp32f, 2> > ImageNpp_32f_C2;
  typedef ImageNpp<Npp32f, 3, iuprivate::ImageAllocatorNpp<Npp32f, 3> > ImageNpp_32f_C3;
  typedef ImageNpp<Npp32f, 4, iuprivate::ImageAllocatorNpp<Npp32f, 4> > ImageNpp_32f_C4;
} // namespace iu

#endif // IUCORE_MEMORYDEFS_H
