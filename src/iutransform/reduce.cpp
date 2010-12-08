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
 * Description : Implementation of reduction transformations
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#include <math.h>
#include <iucore/copy.h>
#include <iufilter/filter.h>
#include "reduce.h"
#include <iostream>

namespace iuprivate {

/* ***************************************************************************
 *  Declaration of CUDA WRAPPERS
 * ***************************************************************************/
extern IuStatus cuReduce(iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst,
                         IuInterpolationType interpolation);

/* ***************************************************************************/


/* ***************************************************************************
 *  FUNCTION IMPLEMENTATIONS
 * ***************************************************************************/

// device; 32-bit; 1-channel
IuStatus reduce(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst,
                IuInterpolationType interpolation,
                bool gauss_prefilter, bool bicubic_bspline_prefilter)
{
  IuStatus status;

  // temporary variable if there is some pre-filtering
  iu::ImageGpu_32f_C1* filtered = const_cast<iu::ImageGpu_32f_C1*>(src);

  // gauss pre-filter
  if(gauss_prefilter)
  {
    filtered = new iu::ImageGpu_32f_C1(src->size());


    // x_/y_factor < 0
    float x_factor = (float)dst->width() / (float)src->width();
    float y_factor = (float)dst->height() / (float)src->height();

    float sigma =/*0.5774f*/0.35f * sqrt(0.5f*(x_factor+y_factor));
    unsigned int kernel_size = 5;

    iuprivate::filterGauss(src, filtered, src->roi(), sigma, kernel_size);
  }

  // convert the input image into cubic bspline coefficients
  // (only useful for cubic interpolation!)
  if(bicubic_bspline_prefilter && interpolation==IU_INTERPOLATE_CUBIC)
  {
    filtered = new iu::ImageGpu_32f_C1(src->size());
    iuprivate::copy(src, filtered);
    iuprivate::cubicBSplinePrefilter(filtered);
  }

  status = cuReduce(filtered, dst, interpolation);

  // cleanup
  if(gauss_prefilter || bicubic_bspline_prefilter)
  {
    delete(filtered);
  }

  return status;
}

}
