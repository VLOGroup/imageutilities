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

#include <iucore/copy.h>
#include <iufilter/filter.h>
#include "reduce.h"

#include <math.h>
#include <memory>

namespace iuprivate {

/* ***************************************************************************
 *  Declaration of CUDA WRAPPERS
 * ***************************************************************************/
extern void cuReduce(iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst,
                         IuInterpolationType interpolation);

/* ***************************************************************************/


/* ***************************************************************************
 *  FUNCTION IMPLEMENTATIONS
 * ***************************************************************************/

// device; 32-bit; 1-channel
void reduce(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst,
                IuInterpolationType interpolation,
                bool gauss_prefilter, bool bicubic_bspline_prefilter)
{
  // temporary variable if there is some pre-filtering
  iu::ImageGpu_32f_C1* filtered = const_cast<iu::ImageGpu_32f_C1*>(src);

  std::auto_ptr<iu::ImageGpu_32f_C1> filtered_auto;

  // gauss pre-filter
  if(gauss_prefilter)
  {
    filtered = new iu::ImageGpu_32f_C1(src->size());
	filtered_auto.reset( filtered );

    // x_/y_factor < 0
    float x_factor = (float)dst->width() / (float)src->width();
    float y_factor = (float)dst->height() / (float)src->height();

    float factor = 0.5f*(x_factor+y_factor);
        //  //    float t=1.0f;
        //  //float sigma = 1.0f/sqrt(6.2832*t) * exp(-(factor*factor)/(2.0f*t)); // lindeberg
//    float sigma =/*0.5774f*/0.3f * sqrtf(factor); //hmmm
        //  //float sigma = 0.1f * sqrtf(0.5f*(x_factor+y_factor));

    float sigma = 1/(3*factor) ;  // empirical!
//    unsigned int kernel_size = 5;
    unsigned int kernel_size = ceil(6*sigma);
    if (kernel_size % 2 == 0)
      kernel_size++;

    //std::cout << "factor=" << factor << " sigma=" << sigma <<  " kernel_sz=" << kernel_size << std::endl;
    iuprivate::filterGauss(src, filtered, src->roi(), sigma, kernel_size);

  }

  // convert the input image into cubic bspline coefficients
  // (only useful for cubic interpolation!)
  if(bicubic_bspline_prefilter && interpolation==IU_INTERPOLATE_CUBIC)
  {
    if(!filtered_auto.get())
    {
      // if no gaussian prefilter applied alloc mem, etc.
      filtered = new iu::ImageGpu_32f_C1(src->size());
	  filtered_auto.reset( filtered );
      iuprivate::copy(src, filtered);
    }

    iuprivate::cubicBSplinePrefilter(filtered);
  }

  cuReduce(filtered, dst, interpolation);

}

}
