#include "iuinteraction.h"
#include <iuinteraction/draw.h>

namespace iu {

IuStatus drawLine(iu::ImageGpu_8u_C1 *image, int x_start, int y_start,
		  int x_end, int y_end, int line_width, unsigned char value)
{ return iuprivate::drawLine(image, x_start, y_start, x_end, y_end, line_width, value); }

} // namespace iu
