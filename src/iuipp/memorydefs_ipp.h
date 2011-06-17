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
 * Module      : IPP-Connector
 * Class       : none
 * Language    : C
 * Description : Typedefs for simpler image usage of the IPP-connector
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

// template includes
#include "image_allocator_ipp.h"
#include "image_ipp.h"

#ifndef MEMORYDEFS_IPP_H
#define MEMORYDEFS_IPP_H

namespace iu {

// typedefs of ipp images here so these files can stand on their on and are only included if used from an ipp aware application
// Ipp Images; 8-bit
typedef ImageIpp<Ipp8u, 1, iuprivate::ImageAllocatorIpp<Ipp8u, 1>, IU_8U_C1> ImageIpp_8u_C1;
typedef ImageIpp<Ipp8u, 2, iuprivate::ImageAllocatorIpp<Ipp8u, 2>, IU_8U_C2> ImageIpp_8u_C2;
typedef ImageIpp<Ipp8u, 3, iuprivate::ImageAllocatorIpp<Ipp8u, 3>, IU_8U_C3> ImageIpp_8u_C3;
typedef ImageIpp<Ipp8u, 4, iuprivate::ImageAllocatorIpp<Ipp8u, 4>, IU_8U_C4> ImageIpp_8u_C4;
// Ipp Images; 32-bit
typedef ImageIpp<Ipp32f, 1, iuprivate::ImageAllocatorIpp<Ipp32f, 1>, IU_32F_C1> ImageIpp_32f_C1;
typedef ImageIpp<Ipp32f, 2, iuprivate::ImageAllocatorIpp<Ipp32f, 2>, IU_32F_C2> ImageIpp_32f_C2;
typedef ImageIpp<Ipp32f, 3, iuprivate::ImageAllocatorIpp<Ipp32f, 3>, IU_32F_C3> ImageIpp_32f_C3;
typedef ImageIpp<Ipp32f, 4, iuprivate::ImageAllocatorIpp<Ipp32f, 4>, IU_32F_C4> ImageIpp_32f_C4;

}

#endif // MEMORYDEFS_IPP_H
