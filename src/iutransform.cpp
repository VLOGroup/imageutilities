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
#include "iutransform/prolongate.h"
#include "iutransform/remap.h"

namespace iu {

/* ***************************************************************************
     Geometric Transformation
 * ***************************************************************************/

/*
  image reduction
 */
void reduce(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst,
            IuInterpolationType interpolation,
            bool gauss_prefilter, bool bicubic_bspline_prefilter)
{iuprivate::reduce(src, dst, interpolation, gauss_prefilter, bicubic_bspline_prefilter);}


/*
  image prolongation
 */
void prolongate(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst,
                IuInterpolationType interpolation)
{iuprivate::prolongate(src, dst, interpolation);}

void prolongate(const iu::ImageGpu_32f_C2* src, iu::ImageGpu_32f_C2* dst,
                IuInterpolationType interpolation)
{iuprivate::prolongate(src, dst, interpolation);}

void prolongate(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst,
                IuInterpolationType interpolation)
{iuprivate::prolongate(src, dst, interpolation);}

/*
  image remapping (warping)
 */
// 8u_C1
void remap(iu::ImageGpu_8u_C1* src,
           iu::ImageGpu_32f_C1* dx_map, iu::ImageGpu_32f_C1* dy_map,
           iu::ImageGpu_8u_C1* dst, IuInterpolationType interpolation)
{iuprivate::remap(src, dx_map, dy_map, dst, interpolation);}

// 32f_C1
void remap(iu::ImageGpu_32f_C1* src,
           iu::ImageGpu_32f_C1* dx_map, iu::ImageGpu_32f_C1* dy_map,
           iu::ImageGpu_32f_C1* dst, IuInterpolationType interpolation)
{iuprivate::remap(src, dx_map, dy_map, dst, interpolation);}


//IuStatus remap(iu::ImageGpu_32f_C2* src,
//           iu::ImageGpu_32f_C1* dx_map, iu::ImageGpu_32f_C1* dy_map,
//           iu::ImageGpu_32f_C2* dst, IuInterpolationType interpolation)
//{return iuprivate::remap(src, dx_map, dy_map, dst, interpolation);}

//IuStatus remap(iu::ImageGpu_32f_C4* src,
//           iu::ImageGpu_32f_C1* dx_map, iu::ImageGpu_32f_C1* dy_map,
//           iu::ImageGpu_32f_C4* dst, IuInterpolationType interpolation)
//{return iuprivate::remap(src, dx_map, dy_map, dst, interpolation);}

} // namespace iu
