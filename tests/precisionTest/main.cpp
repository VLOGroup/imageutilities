#include <iostream>
#include "../config.h"
#include "../../src/iucore.h"
#include "../../src/iumath.h"

int main()
{

  typedef double PixelType;

  const unsigned int numel = 100;
  iu::LinearDeviceMemory<PixelType> d_linmem(numel);
  iu::LinearHostMemory<PixelType> h_linmem(numel);
  iu::math::fill(d_linmem, static_cast<PixelType>(10));

  iu::LinearDeviceMemory<PixelType> d_linmem2(numel);
  iu::math::fill(d_linmem2, static_cast<PixelType>(2));

  iu::math::addC(d_linmem, static_cast<PixelType>(1.5), d_linmem);
  iu::math::addWeighted(d_linmem, static_cast<PixelType>(0.5),
                       d_linmem2, static_cast<PixelType>(0.5),
                       d_linmem);

  iu::math::fill(h_linmem, static_cast<PixelType>(0));
  iu::copy(&d_linmem, &h_linmem);

  return EXIT_SUCCESS;
}
