
//#include <iostream>
#include <math.h>
#include <iucore/copy.h>
#include "filter.h"
#include "prolongate.h"

namespace iuprivate {

/* ***************************************************************************
 *  Declaration of CUDA WRAPPERS
 * ***************************************************************************/
extern void cuProlongate(iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst,
                             IuInterpolationType interpolation);
extern void cuProlongate(iu::ImageGpu_32f_C2* src, iu::ImageGpu_32f_C2* dst,
                             IuInterpolationType interpolation);
extern void cuProlongate(iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst,
                             IuInterpolationType interpolation);

/* ***************************************************************************/


/* ***************************************************************************
 *  FUNCTION IMPLEMENTATIONS
 * ***************************************************************************/

// device; 32-bit; 1-channel
void prolongate(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_32f_C1* dst,
                    IuInterpolationType interpolation)
{
  return cuProlongate(const_cast<iu::ImageGpu_32f_C1*>(src), dst, interpolation);
}

// device; 32-bit; 2-channel
void prolongate(const iu::ImageGpu_32f_C2* src, iu::ImageGpu_32f_C2* dst,
                    IuInterpolationType interpolation)
{
  return cuProlongate(const_cast<iu::ImageGpu_32f_C2*>(src), dst, interpolation);
}

// device; 32-bit; 4-channel
void prolongate(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst,
                    IuInterpolationType interpolation)
{
  return cuProlongate(const_cast<iu::ImageGpu_32f_C4*>(src), dst, interpolation);
}

}
