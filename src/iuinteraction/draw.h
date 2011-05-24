
#ifndef DRAW_H
#define DRAW_H

//
//  W A R N I N G
//  -------------
//
// This file is not part of the IU API.  It exists purely as an
// implementation detail.  This header file may change from version to
// version without notice, or even be removed.
//

#include <iucore/coredefs.h>
#include <iucore/memorydefs.h>

namespace iuprivate {

IuStatus drawLine(iu::ImageGpu_8u_C1 *image, int x_start, int y_start,
                  int x_end, int y_end, int line_width, unsigned char value);


} // namespace iuprivate

#endif // DRAW_H
