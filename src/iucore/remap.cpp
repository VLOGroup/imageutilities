
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
extern void cuRemap(iu::ImageGpu_32f_C2* src,
                    iu::ImageGpu_32f_C1* dx_map, iu::ImageGpu_32f_C1* dy_map,
                    iu::ImageGpu_32f_C2* dst, IuInterpolationType interpolation);
extern void cuRemap(iu::ImageGpu_32f_C4* src,
                    iu::ImageGpu_32f_C1* dx_map, iu::ImageGpu_32f_C1* dy_map,
                    iu::ImageGpu_32f_C4* dst, IuInterpolationType interpolation);

extern void cuRemapAffine(iu::ImageGpu_32f_C1* src,
                          float a1, float a2, float a3, float a4,
                          float b1, float b2,
                          iu::ImageGpu_32f_C1* dst);

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
//void remap(iu::ImageGpu_32f_C2* src,
//               iu::ImageGpu_32f_C1* dx_map, iu::ImageGpu_32f_C1* dy_map,
//               iu::ImageGpu_32f_C2* dst, IuInterpolationType interpolation)
//{
//  return cuRemap(src, dx_map, dy_map, dst, interpolation);
//}

// device; 32-bit; 4-channel
void remap(iu::ImageGpu_32f_C4* src,
               iu::ImageGpu_32f_C1* dx_map, iu::ImageGpu_32f_C1* dy_map,
               iu::ImageGpu_32f_C4* dst, IuInterpolationType interpolation)
{
  return cuRemap(src, dx_map, dy_map, dst, interpolation);
}

void remapAffine(iu::ImageGpu_32f_C1* src,
                 float a1, float a2, float a3, float a4,
                 float b1, float b2,
                 iu::ImageGpu_32f_C1* dst)
{
  return cuRemapAffine(src, a1, a2, a3, a4, b1, b2, dst);
}

}
