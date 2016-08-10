#pragma once 

#include "../../src/iucore.h"

namespace cuda
{

void test(iu::LinearDeviceMemory<float, 1>& img);
void test(iu::LinearDeviceMemory<float, 2>& img);

void test(iu::LinearDeviceMemory<float, 4>& img);
void test2(iu::LinearDeviceMemory<float, 4>& img);

}
