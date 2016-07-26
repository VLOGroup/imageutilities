#pragma once 

#include "../../src/iucore.h"

namespace cuda
{

void test(iu::LinearDeviceMemory<float, 1>& img);
void test(iu::LinearDeviceMemory<float, 2>& img);


}
