#pragma once

#include <iucore/coredefs.h>
#include <iucore/memorydefs.h>

namespace iuprivate {

void prolongate(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst,
                    IuInterpolationType interpolation = IU_INTERPOLATE_NEAREST);
void prolongate(const iu::ImageGpu_32f_C2* src, iu::ImageGpu_32f_C2* dst,
                    IuInterpolationType interpolation = IU_INTERPOLATE_NEAREST);
void prolongate(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst,
                    IuInterpolationType interpolation = IU_INTERPOLATE_NEAREST);



}  // namespace iuprivate


