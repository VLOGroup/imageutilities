#ifndef IU_CONTAINERS_H
#define IU_CONTAINERS_H

#include <vector>
#include <deque>

#include "iucore/memorydefs.h"
#include "iucore/imagepyramid.h"


namespace iu {

// Image and Pyramid 'lists' implemented as double-ended queue
typedef std::vector<iu::Image*> ImageVector;
typedef std::deque<iu::Image*> ImageDeque;
typedef std::deque<iu::ImagePyramid*> ImagePyramidDeque;

} // namespace iu

#endif // IU_CONTAINERS_H
