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
 * Description : Definition of convert operations on host and device memory
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */


#ifndef IUPRIVATE_IUCORE_CONVERT_H
#define IUPRIVATE_IUCORE_CONVERT_H

//
//  W A R N I N G
//  -------------
//
// This file is not part of the IU API.  It exists purely as an
// implementation detail.  This header file may change from version to
// version without notice, or even be removed.
//

#include "coredefs.h"
#include "memorydefs.h"

namespace iuprivate {

// [host] 2D bit depth conversion; 32f_C1 -> 8u_C1;
void convert_32f8u_C1(const iu::ImageCpu_32f_C1* src, iu::ImageCpu_8u_C1 *dst,
                      float mul_constant=255.0f, float add_constant=0.0f);

// [host] 2D bit depth conversion; 16u_C1 -> 32f_C1;
void convert_16u32f_C1(const iu::ImageCpu_16u_C1* src, iu::ImageCpu_32f_C1 *dst,
                       float mul_constant, float add_constant);

 // [device] 2D bit depth conversion; 32f_C1 -> 8u_C1;
void convert_32f8u_C1(const iu::ImageGpu_32f_C1* src, const IuRect& src_roi, iu::ImageGpu_8u_C1 *dst, const IuRect& dst_roi,
                      float mul_constant=255.0f, unsigned char add_constant=0);

// [device] 2D bit depth conversion; 32f_C4 -> 8u_C4;
void convert_32f8u_C4(const iu::ImageGpu_32f_C4* src, const IuRect& src_roi, iu::ImageGpu_8u_C4 *dst, const IuRect& dst_roi,
                     float mul_constant=255.0f, unsigned char add_constant=0);

// [device] 2D bit depth conversion; 8u_C1 -> 32f_C1;
void convert_8u32f_C1(const iu::ImageGpu_8u_C1* src, const IuRect& src_roi, iu::ImageGpu_32f_C1 *dst, const IuRect& dst_roi,
                       float mul_constant=1/255.0f, float add_constant=0.0f);


// 2D conversion; device; 32-bit 3-channel -> 32-bit 4-channel
void convert(const iu::ImageGpu_32f_C3* src, const IuRect& src_roi, iu::ImageGpu_32f_C4* dst, const IuRect& dst_roi);

// 2D conversion; device; 32-bit 4-channel -> 32-bit 3-channel
void convert(const iu::ImageGpu_32f_C4* src, const IuRect& src_roi, iu::ImageGpu_32f_C3* dst, const IuRect& dst_roi);

// [device] 2D Color conversion from RGB to HSV (32-bit 4-channel)
void convertRgbHsv(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst, bool normalize);

// [device] 2D Color conversion from HSV to RGB (32-bit 4-channel)
void convertHsvRgb(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst, bool denormalize);

} // namespace iuprivate

#endif // IUPRIVATE_IUCORE_CONVERT_H
