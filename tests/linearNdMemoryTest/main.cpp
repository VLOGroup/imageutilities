#include <iostream>

#include "../config.h"

#include "../../src/iucore.h"
#include "../../src/iuio.h"
#include "../../src/iumath.h"

#include "test.cuh"

void testSize3d()
{
  std::cout << "testSize() started." << std::endl;
  const unsigned int Ndim = 3;
  iu::Size<Ndim> size(2);
  std::cout << size << std::endl;
  size.fill(5);
  std::cout << size << std::endl;
  size.width = 200;
  std::cout << size << std::endl;
  size = size * 10;
  std::cout << size << std::endl;
  std::cout << size.width << " " << size.height << " " << size.depth
      << std::endl;
  std::cout << "testSize() finished." << std::endl;
}

void testLinear1dMemory()
{
  std::cout << "testLinear1dMemory() started." << std::endl;
  iu::Size<1> size(10);
  iu::LinearDeviceMemory<float, 1> mem_d(size);
  iu::math::fill(mem_d, 5.f);
  std::cout << mem_d << std::endl;
  cuda::test(mem_d);
  iu::LinearHostMemory<float, 1> mem_h(size);
  iu::copy(&mem_d, &mem_h);
  std::cout << mem_h.getPixel(mem_d.numel()-1) << std::endl;
  std::cout << mem_h.data(mem_d.numel()-1)[0] << std::endl;
  std::cout << "testLinear1dMemory() finished." << std::endl;
}

void testLinear2dMemory()
{
  std::cout << "testLinear2dMemory() started." << std::endl;
  iu::Size<2> size;
  size[0] = 10;
  size[1] = 100;
  iu::LinearDeviceMemory<float, 2> mem_d(size);
  iu::math::fill(mem_d, 5.f);
  std::cout << mem_d << std::endl;
  cuda::test(mem_d);
  iu::LinearHostMemory<float, 2> mem_h(size);
  iu::copy(&mem_d, &mem_h);
  std::cout << mem_h.getPixel(mem_d.numel()-1) << std::endl;
  std::cout << "testLinear2dMemory() finished." << std::endl;
}

void testLinear4dMemory()
{
  std::cout << "testLinear4dMemory() started." << std::endl;
  iu::Size<4> size(20);
  iu::LinearDeviceMemory<float, 4> mem_d(size);
  iu::math::fill(mem_d, 0.f);
  std::cout << mem_d << std::endl;
  cuda::test(mem_d);
  iu::LinearHostMemory<float, 4> mem_h(size);
  iu::copy(&mem_d, &mem_h);
  std::cout << mem_h.getPixel(size[0]-1, size[1]-1, size[2]-1, size[3]-1) << std::endl;

  iu::math::fill(mem_d, 0.f);
  cuda::test2(mem_d);
  iu::copy(&mem_d, &mem_h);
  std::cout << mem_h.getPixel(size[0]-1, size[1]-1, size[2]-1, size[3]-1) << std::endl;
  std::cout << "testLinear4dMemory() finished." << std::endl;
}

int main()
{
  testSize3d();
  testLinear1dMemory();
  testLinear2dMemory();
  testLinear4dMemory();

  std::cout << "main() finished!" << std::endl;
}
