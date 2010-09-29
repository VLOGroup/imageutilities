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
NppStatus cuAddWeighted(const iu::ImageGpu_32f_C1* src1, const Npp32f& weight1,
                        const iu::ImageGpu_32f_C1* src2, const Npp32f& weight2,
                        iu::ImageGpu_32f_C1* dst, const IuRect& roi);


/** Cuda wrappers for not-in-place multiplication of every pixel with a constant factor.
 * \param src Source image.
 * \param factor Multiplication factor applied to each pixel.
 * \param dst Destination image.
 * \param roi Region of interest in the source and destination image
 *
 * \note supported npp: 8u_C1, 8u_C4, 32f_C1, 32f_C4
 * \note 3-channel stuff not supported due to texture usage!
 */
NppStatus cuMulC(const iu::ImageGpu_8u_C1* src, const Npp8u& factor, iu::ImageGpu_8u_C1* dst, const IuRect& roi);
NppStatus cuMulC(const iu::ImageGpu_8u_C4* src, const Npp8u factor[4], iu::ImageGpu_8u_C4* dst, const IuRect& roi);
NppStatus cuMulC(const iu::ImageGpu_32f_C1* src, const Npp32f& factor, iu::ImageGpu_32f_C1* dst, const IuRect& roi);
NppStatus cuMulC(const iu::ImageGpu_32f_C4* src, const Npp32f factor[4], iu::ImageGpu_32f_C4* dst, const IuRect& roi);


} // namespace iuprivate

#endif // IUMATH_ARITHMETIC_CUH
