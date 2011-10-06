#include "iuinteraction.h"
#include <iuinteraction/draw.h>

namespace iu {

//-------------------------------------------------------------------------------------
void drawLine(iu::Image *image, int x_start, int y_start,
              int x_end, int y_end, int line_width, float value)
{
  iuprivate::drawLine(image, x_start, y_start, x_end, y_end, line_width, value);
}

} // namespace iu
