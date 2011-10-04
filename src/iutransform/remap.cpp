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
 * Description : Implementation of remap transformations (with dense disparities)
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#include "remap.h"

namespace iuprivate {

/* ***************************************************************************
 *  Declaration of CUDA WRAPPERS
 * ***************************************************************************/
extern void cuRemap(iu::ImageGpu_8u_C1* src,
                        iu::ImageGpu_32f_C1* dx_map, iu::ImageGpu_32f_C1* dy_map,
                        iu::ImageGpu_8u_C1* dst, IuInterpolationType interpolation);
extern void cuRemap(iu::ImageGpu_32f_C1* src,
                        iu::ImageGpu_32f_C1* dx_map, iu::ImageGpu_32f_C1* dy_map,
                        iu::ImageGpu_32f_C1* dst, IuInterpolationType interpolation);
//extern IuStatus cuRemap(iu::ImageGpu_32f_C2* src,
//                        iu::ImageGpu_32f_C1* dx_map, iu::ImageGpu_32f_C1* dy_map,
//                        iu::ImageGpu_32f_C2* dst, IuInterpolationType interpolation);
//extern IuStatus cuRemap(iu::ImageGpu_32f_C4* src,
//                        iu::ImageGpu_32f_C1* dx_map, iu::ImageGpu_32f_C1* dy_map,
//                        iu::ImageGpu_32f_C4* dst, IuInterpolationType interpolation);

/* ***************************************************************************/


/* ***************************************************************************
 *  FUNCTION IMPLEMENTATIONS
 * ***************************************************************************/

// device; 8-bit; 1-channel
void remap(iu::ImageGpu_8u_C1* src,
               iu::ImageGpu_32f_C1* dx_map, iu::ImageGpu_32f_C1* dy_map,
               iu::ImageGpu_8u_C1* dst, IuInterpolationType interpolation)
{
  cuRemap(src, dx_map, dy_map, dst, interpolation);
}

// device; 32-bit; 1-channel
void remap(iu::ImageGpu_32f_C1* src,
               iu::ImageGpu_32f_C1* dx_map, iu::ImageGpu_32f_C1* dy_map,
               iu::ImageGpu_32f_C1* dst, IuInterpolationType interpolation)
{
  cuRemap(src, dx_map, dy_map, dst, interpolation);
}

//// device; 32-bit; 2-channel
//IuStatus remap(iu::ImageGpu_32f_C2* src,
//               iu::ImageGpu_32f_C1* dx_map, iu::ImageGpu_32f_C1* dy_map,
//               iu::ImageGpu_32f_C2* dst, IuInterpolationType interpolation)
//{
//  return cuRemap(src, dx_map, dy_map, dst, interpolation);
//}

//// device; 32-bit; 4-channel
//IuStatus remap(iu::ImageGpu_32f_C4* src,
//               iu::ImageGpu_32f_C1* dx_map, iu::ImageGpu_32f_C1* dy_map,
//               iu::ImageGpu_32f_C4* dst, IuInterpolationType interpolation)
//{
//  return cuRemap(src, dx_map, dy_map, dst, interpolation);
//}

}
