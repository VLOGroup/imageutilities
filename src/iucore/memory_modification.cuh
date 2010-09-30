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
 * Class       : none
 * Language    : C
 * Description : Definition of CUDA wrappers for memory modifications
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUCORE_MEMORY_MODIFICATION_CUH
#define IUCORE_MEMORY_MODIFICATION_CUH

#include "coredefs.h"
#include "memorydefs.h"

namespace iuprivate {

/** Sets values of 1D linear gpu memory.
 * \param value The pixel value to be set.
 * \param buffer Pointer to the buffer
 */
IuStatus cuSetValue(const unsigned char& value, iu::LinearDeviceMemory_8u_C1* dst);
IuStatus cuSetValue(const float& value, iu::LinearDeviceMemory_32f_C1* dst);

/** Sets values of 2D gpu memory.
 * \param value The pixel value to be set.
 * \param dst Destination image
 * \param roi Region of interest of processed pixels
 */
IuStatus cuSetValue(const unsigned char& value, iu::ImageGpu_8u_C1 *dst, const IuRect &roi);
IuStatus cuSetValue(const uchar2& value, iu::ImageGpu_8u_C2 *dst, const IuRect &roi);
IuStatus cuSetValue(const uchar3& value, iu::ImageGpu_8u_C3 *dst, const IuRect &roi);
IuStatus cuSetValue(const uchar4& value, iu::ImageGpu_8u_C4 *dst, const IuRect &roi);
IuStatus cuSetValue(const float& value, iu::ImageGpu_32f_C1 *dst, const IuRect &roi);
IuStatus cuSetValue(const float2& value, iu::ImageGpu_32f_C2 *dst, const IuRect &roi);
IuStatus cuSetValue(const float3& value, iu::ImageGpu_32f_C3 *dst, const IuRect &roi);
IuStatus cuSetValue(const float4& value, iu::ImageGpu_32f_C4 *dst, const IuRect &roi);


/** Sets values of 3D gpu memory.
 * \param value The pixel value to be set.
 * \param dst Destination image
 * \param roi Region of interest of processed pixels
 */
IuStatus cuSetValue(const unsigned char& value, iu::VolumeGpu_8u_C1 *dst, const IuCube &roi);
IuStatus cuSetValue(const uchar2& value, iu::VolumeGpu_8u_C2 *dst, const IuCube &roi);
IuStatus cuSetValue(const uchar4& value, iu::VolumeGpu_8u_C4 *dst, const IuCube &roi);
IuStatus cuSetValue(const float& value, iu::VolumeGpu_32f_C1 *dst, const IuCube &roi);
IuStatus cuSetValue(const float2& value, iu::VolumeGpu_32f_C2 *dst, const IuCube &roi);
IuStatus cuSetValue(const float4& value, iu::VolumeGpu_32f_C4 *dst, const IuCube &roi);

/** Clamps all values of srcdst to the interval min/max.
 * \param min Minimum value for the clamping.
 * \param max Maximum value for the clamping.
 * \param srcdst Image which pixels are clamped.
 * \param roi Region of interest of processed pixels.
 */
IuStatus cuClamp(const float& min, const float& max,
                  iu::ImageGpu_32f_C1 *srcdst, const IuRect &roi);

/** Converts an 32-bit 3-channel image to a 32-bit 4-channel image (adds alpha channel with value 1.0 everywhere)
 *
 */
IuStatus cuConvert(const iu::ImageGpu_32f_C3* src, const IuRect& src_roi, iu::ImageGpu_32f_C4* dst, const IuRect& dst_roi);

/** Converts an 32-bit 4-channel image to a 32-bit 3-channel image (the alpha channel is simply neglected).
 *
 */
IuStatus cuConvert(const iu::ImageGpu_32f_C4* src, const IuRect& src_roi, iu::ImageGpu_32f_C3* dst, const IuRect& dst_roi);

} // namespace iuprivate

#endif // IUCORE_MEMORY_MODIFICATION_CUH
