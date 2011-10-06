#include "draw.h"

namespace iuprivate {

extern
void CUDAdrawLine(iu::Image *image, int x_start, int y_start,
                  int x_end, int y_end, int line_width, float value);

//extern void CUDAdrawLine(iu::Image *image, int x_start, int y_start,
//                             int x_end, int y_end, int line_width, unsigned char value);

//-----------------------------------------------------------------------------
void drawLine(iu::Image *image, int x_start, int y_start,
              int x_end, int y_end, int line_width, float value)
{
  if (image->onDevice())
    CUDAdrawLine(image, x_start, y_start, x_end, y_end, line_width, value);
  else
    throw IuException("Only GPU images supported.", __FILE__, __FUNCTION__, __LINE__);


}

}
