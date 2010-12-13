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
 * Module      : Geometric Transform
 * Class       : none
 * Language    : C++
 * Description : Implementation of prolongate transformations
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

//#include <iostream>
#include <math.h>
#include <iucore/copy.h>
#include <iufilter/filter.h>
#include "prolongate.h"

namespace iuprivate {

/* ***************************************************************************
 *  Declaration of CUDA WRAPPERS
 * ***************************************************************************/
extern IuStatus cuProlongate(iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst,
                             IuInterpolationType interpolation);
extern IuStatus cuProlongate(iu::ImageGpu_32f_C2* src, iu::ImageGpu_32f_C2* dst,
                             IuInterpolationType interpolation);
extern IuStatus cuProlongate(iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst,
                             IuInterpolationType interpolation);

/* ***************************************************************************/


/* ***************************************************************************
 *  FUNCTION IMPLEMENTATIONS
 * ***************************************************************************/

// device; 32-bit; 1-channel
IuStatus prolongate(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst,
                    IuInterpolationType interpolation)
{
  return cuProlongate(const_cast<iu::ImageGpu_32f_C1*>(src), dst, interpolation);
}

// device; 32-bit; 2-channel
IuStatus prolongate(const iu::ImageGpu_32f_C2* src, iu::ImageGpu_32f_C2* dst,
                    IuInterpolationType interpolation)
{
  return cuProlongate(const_cast<iu::ImageGpu_32f_C2*>(src), dst, interpolation);
}

// device; 32-bit; 4-channel
IuStatus prolongate(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst,
                    IuInterpolationType interpolation)
{
  return cuProlongate(const_cast<iu::ImageGpu_32f_C4*>(src), dst, interpolation);
}

}
