#pragma once

#include <iucore/coredefs.h>
#include <iucore/memorydefs.h>

namespace iuprivate {

// device; 32-bit; 1-channel
void reduce(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst,
            IuInterpolationType interpolation = IU_INTERPOLATE_LINEAR,
            bool gauss_prefilter = true);

void reduce(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst,
            iu::ImageGpu_32f_C1* temp, iu::ImageGpu_32f_C1* temp_filter, cudaStream_t stream,
            IuInterpolationType interpolation = IU_INTERPOLATE_LINEAR,
            bool gauss_prefilter = true);

} // namespace iuprivate

