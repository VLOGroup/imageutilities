#ifndef IUINTERACTION_H
#define IUINTERACTION_H

#include "iudefs.h"


namespace iu {

/** Draw a line to input image
//! @param image       The image to which will be drawn
//! @param x_start     First coordinate x
//! @param y_start     First coordinate y
//! @param x_end       Second coordinate x
//! @param y_end       Second coordinate y
//! @param line_width  Width of the line
//! @param value       Intensity to draw with
 */
IU_DLLAPI IuStatus drawLine(iu::ImageGpu_8u_C1 *image, int x_start, int y_start,
                            int x_end, int y_end, int line_width, unsigned char value);


} // namespace iu

#endif // IUINTERACTION_H
