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
 * Module      : Filter
 * Class       : none
 * Language    : CUDA
 * Description : Definition of CUDA wrappers for reduce functions on Gpu images
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUPRIVATE_REDUCE_CUH
#define IUPRIVATE_REDUCE_CUH

#include <iucore/coredefs.h>
#include <iucore/memorydefs.h>

namespace iuprivate {

IuStatus cuCubicBSplinePrefilter_32f_C1I(iu::ImageGpu_32f_C1 *input);
IuStatus cuReduce(iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst,
                  IuInterpolationType interpolation);



} // namespace iuprivate

#endif // IUPRIVATE_REDUCE_CUH
