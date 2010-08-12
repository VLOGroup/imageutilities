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
 * Language    : C++
 * Description : Definition of arithmetic functions
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUMATH_ARITHMETIC_H
#define IUMATH_ARITHMETIC_H

#include <iucore/coredefs.h>
#include <iucore/memorydefs.h>

namespace iuprivate {

/** Adding an image with an additional weighting factor to another.
 * \param src1 Source image 1.
 * \param src2 Source image 2.
 * \param weight Weighting of image 2 before its added to image 1.
 * \param dst Result image dst=weight1*src1 + weight1*src2.
 * \param roi Region of interest in the source and destination image
 *
 * \note supported npp: 32f_C1
 */
// [device] weighted add; Not-in-place; 32-bit;
void addWeighted(const iu::ImageNpp_32f_C1* src1, const Npp32f& weight1,
                 const iu::ImageNpp_32f_C1* src2, const Npp32f& weight2,
                 iu::ImageNpp_32f_C1* dst, const IuRect& roi);

/** Not-in-place multiplication of every pixel with a constant factor.
 * \param src Source image.
 * \param factor Multiplication factor applied to each pixel.
 * \param dst Destination image.
 * \param roi Region of interest in the source and destination image
 *
 * \note supported npp: 8u_C1, 8u_C4, 32f_C1, 32f_C4,
 */
// [device] multiplication with factor; Not-in-place; 8-bit;
void mulC(const iu::ImageNpp_8u_C1* src, const Npp8u& factor, iu::ImageNpp_8u_C1* dst, const IuRect& roi);
void mulC(const iu::ImageNpp_8u_C4* src, const Npp8u factor[4], iu::ImageNpp_8u_C4* dst, const IuRect& roi);

// [device] multiplication with factor; Not-in-place; 32-bit;
void mulC(const iu::ImageNpp_32f_C1* src, const Npp32f& factor, iu::ImageNpp_32f_C1* dst, const IuRect& roi);
void mulC(const iu::ImageNpp_32f_C4* src, const Npp32f factor[4], iu::ImageNpp_32f_C4* dst, const IuRect& roi);

/** In-place multiplication of every pixel with a constant factor.
 * \param factor Multiplication factor applied to each pixel.
 * \param srcdst Source and destination
 * \param roi Region of interest in the source/destination image
 *
 * \note supported npp: 8u_C1, 8u_C4, 32f_C1, 32f_C4,
 */
// multiplication with factor; host; 8-bit




} // namespace iuprivate

#endif // IUMATH_ARITHMETIC_H
