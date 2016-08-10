#pragma once

#include <iucore/coredefs.h>
#include <iucore/memorydefs.h>

namespace iuprivate {

void remap(iu::ImageGpu_8u_C1* src,
           iu::ImageGpu_32f_C1* dx_map, iu::ImageGpu_32f_C1* dy_map,
           iu::ImageGpu_8u_C1* dst, IuInterpolationType interpolation);
void remap(iu::ImageGpu_32f_C1* src,
           iu::ImageGpu_32f_C1* dx_map, iu::ImageGpu_32f_C1* dy_map,
           iu::ImageGpu_32f_C1* dst, IuInterpolationType interpolation);
//void remap(iu::ImageGpu_32f_C2* src,
//           iu::ImageGpu_32f_C1* dx_map, iu::ImageGpu_32f_C1* dy_map,
//           iu::ImageGpu_32f_C2* dst, IuInterpolationType interpolation);
void remap(iu::ImageGpu_32f_C4* src,
           iu::ImageGpu_32f_C1* dx_map, iu::ImageGpu_32f_C1* dy_map,
           iu::ImageGpu_32f_C4* dst, IuInterpolationType interpolation);

void remapAffine(iu::ImageGpu_32f_C1* src,
                 float a1, float a2, float a3, float a4,
                 float b1, float b2,
                 iu::ImageGpu_32f_C1* dst);

}  // namespace iuprivate


