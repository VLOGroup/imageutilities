#ifndef IU_CONTAINERS_H
#define IU_CONTAINERS_H

#include <vector>
#include <deque>
#include <map>

#include "iucore/memorydefs.h"
#include "iucore/imagepyramid.h"


namespace iu {

// Image and Pyramid 'lists' implemented as double-ended queue
typedef std::vector<iu::Image*> ImageVector;
typedef std::deque<iu::Image*> ImageDeque;
typedef std::deque<iu::ImageGpu_32f_C1*> ImageGpuDeque_32f_C1;
typedef std::deque<iu::ImageGpu_32f_C2*> ImageGpuDeque_32f_C2;
typedef std::deque<iu::ImagePyramid*> ImagePyramidDeque;
typedef std::map<float,iu::Image*> TimeImageMap;
typedef std::deque<iu::ImagePyramid*> ImagePyramidDeque;
typedef std::map<float,iu::ImagePyramid*> TimeImagePyramidMap;

} // namespace iu

#endif // IU_CONTAINERS_H
