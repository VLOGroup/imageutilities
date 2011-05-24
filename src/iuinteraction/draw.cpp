#include "draw.h"

namespace iuprivate {

extern IuStatus CUDAdrawLine(iu::ImageGpu_8u_C1 *image, int x_start, int y_start,
                             int x_end, int y_end, int line_width, unsigned char value);

//-----------------------------------------------------------------------------
IuStatus drawLine(iu::ImageGpu_8u_C1 *image, int x_start, int y_start,
                  int x_end, int y_end, int line_width, unsigned char value)
{
  return CUDAdrawLine(image, x_start, y_start, x_end, y_end, line_width, value);
}

}
