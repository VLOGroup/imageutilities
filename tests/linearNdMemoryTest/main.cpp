#include <iostream>
#include "../config.h"
#include "../../src/iucore.h"
#include "../../src/iuio.h"
#include "../../src/iumath.h"
#include "../../src/iucore/iuvector.h"
#include "../../src/iucore/linearhostmemory.h"
#include "test.cuh"

int main()
{

  iu::Size<1> size;
  std::cout << size << std::endl;

  size.fill(5);
  size[0] = 200;
  std::cout << size << std::endl;

  size = size * 10;
  std::cout << size << std::endl;
  //std::cout << size.width << " " << size.height << " " << size.depth << std::endl;

  size /= 5;
  std::cout << size << std::endl;

  //std::cout << size.width << " " << size.height << " " << size.depth << std::endl;

  iu::LinearDeviceMemory<float, 1> test(size);
  iu::math::fill(test, 5.f);
  std::cout << test << std::endl;

  cuda::test(test);

  iu::LinearHostMemory<float, 1> test2(size);
  std::cout << test2 << std::endl;

  iu::copy(&test, &test2);
  std::cout << test2.data()[size[0]-1] << std::endl;
  std::cout << "finished!" << std::endl;
}
