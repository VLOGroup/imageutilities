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
 * Module      : Math
 * Class       : none
 * Language    : CUDA
 * Description : Definition of Cuda wrappers for arithmetic functions
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUMATH_ARITHMETIC_CUH
#define IUMATH_ARITHMETIC_CUH

#include <iucore/coredefs.h>
#include <iucore/memorydefs.h>

namespace iuprivate {

// cuda wrapper: weighted add; Not-in-place; 32-bit;
IuStatus cuAddWeighted(const iu::ImageGpu_32f_C1* src1, const float& weight1,
                        const iu::ImageGpu_32f_C1* src2, const float& weight2,
                        iu::ImageGpu_32f_C1* dst, const IuRect& roi);


/** Cuda wrappers for not-in-place multiplication of every pixel with a constant factor.
 * \param src Source image.
 * \param factor Multiplication factor applied to each pixel.
 * \param dst Destination image.
 * \param roi Region of interest in the source and destination image
 *
 * \note supported gpu: 8u_C1, 8u_C4, 32f_C1, 32f_C4
 * \note 3-channel stuff not supported due to texture usage!
 */
IuStatus cuMulC(const iu::ImageGpu_8u_C1* src, const unsigned char& factor, iu::ImageGpu_8u_C1* dst, const IuRect& roi);
IuStatus cuMulC(const iu::ImageGpu_8u_C4* src, const uchar4& factor, iu::ImageGpu_8u_C4* dst, const IuRect& roi);
IuStatus cuMulC(const iu::ImageGpu_32f_C1* src, const float& factor, iu::ImageGpu_32f_C1* dst, const IuRect& roi);
IuStatus cuMulC(const iu::ImageGpu_32f_C2* src, const float2& factor, iu::ImageGpu_32f_C2* dst, const IuRect& roi);
IuStatus cuMulC(const iu::ImageGpu_32f_C4* src, const float4& factor, iu::ImageGpu_32f_C4* dst, const IuRect& roi);

/** Cuda wrappers for volumetric not-in-place multiplication of every pixel with a constant factor.
 * \param src Source volume.
 * \param factor Multiplication factor applied to each pixel.
 * \param dst Destination volume.
 *
 * \note supported gpu: 32f_C1,
 * \note 3-channel stuff not supported due to texture usage!
 */
IuStatus cuMulC(const iu::VolumeGpu_32f_C1* src, const float& factor, iu::VolumeGpu_32f_C1* dst);

/** Cuda wrappers for not-in-place addition to every pixel of a constant factor.
 * \param src Source image.
 * \param val value that will be added
 * \param dst Destination image.
 * \param roi Region of interest in the source and destination image
 *
 * \note supported gpu: 8u_C1, 8u_C4, 32f_C1, 32f_C4
 * \note 3-channel stuff not supported due to texture usage!
 */
IuStatus cuAddC(const iu::ImageGpu_8u_C1* src, const unsigned char& val, iu::ImageGpu_8u_C1* dst, const IuRect& roi);
IuStatus cuAddC(const iu::ImageGpu_8u_C4* src, const uchar4& val, iu::ImageGpu_8u_C4* dst, const IuRect& roi);
IuStatus cuAddC(const iu::ImageGpu_32f_C1* src, const float& val, iu::ImageGpu_32f_C1* dst, const IuRect& roi);
IuStatus cuAddC(const iu::ImageGpu_32f_C2* src, const float2& val, iu::ImageGpu_32f_C2* dst, const IuRect& roi);
IuStatus cuAddC(const iu::ImageGpu_32f_C4* src, const float4& val, iu::ImageGpu_32f_C4* dst, const IuRect& roi);


} // namespace iuprivate

#endif // IUMATH_ARITHMETIC_CUH
