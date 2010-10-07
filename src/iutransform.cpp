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
 * Module      : Geometric Transformation
 * Class       : Wrapper
 * Language    : C
 * Description : Implementation of public interfaces for the geometric transformation module
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */


#include "iutransform.h"
#include "iutransform/reduce.h"

namespace iu {

/* ***************************************************************************
     Geometric Transformation
 * ***************************************************************************/
void reduce(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst,
            IuInterpolationType interpolation,
            bool gauss_prefilter, bool bicubic_bspline_prefilter)
{iuprivate::reduce(src, dst, interpolation, gauss_prefilter, bicubic_bspline_prefilter);}

} // namespace iu
