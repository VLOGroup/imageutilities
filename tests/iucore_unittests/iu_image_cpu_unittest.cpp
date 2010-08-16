/*
 * Copyright (c) ICG. All rights reserved.
 *
 * Institute for Computer Graphics and Vision
 * Graz University of Technology / Austria
 *
 *
 * This software is distributed WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 * PURPOSE.  See the above copyright notices for more information.
 *
 *
 * Project     : ImageUtilities
 * Module      : Unit Tests
 * Class       : none
 * Language    : C++
 * Description : Unit tests for Cpu images
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

// system includes
#include <iostream>
#include <cuda_runtime.h>
#include <iucore.h>

using namespace iu;

/* compares two float values.
   taken from [1]
   [1] http://www.cygnus-software.com/papers/comparingfloats/Comparing%20floating%20point%20numbers.htm
*/
bool almostEquals(float A, float B, int maxUlps = 1)
{
  // Make sure maxUlps is non-negative and small enough that the
  // default NAN won't compare as equal to anything.
  assert(maxUlps > 0 && maxUlps < 4 * 1024 * 1024);
  int aInt = *(int*)&A;
  // Make aInt lexicographically ordered as a twos-complement int
  if (aInt < 0)
    aInt = 0x80000000 - aInt;
  // Make bInt lexicographically ordered as a twos-complement int
  int bInt = *(int*)&B;
  if (bInt < 0)
    bInt = 0x80000000 - bInt;
  int intDiff = abs(aInt - bInt);
  if (intDiff <= maxUlps)
    return true;
  return false;
}

int main(int argc, char** argv)
{
  std::cout << "Starting iu_image_cpu_unittest ..." << std::endl;

  // test image size
  IuSize sz(79,63);

  iu::ImageCpu_8u_C1 im_cpu_8u_C1(sz);
  iu::ImageCpu_8u_C2 im_cpu_8u_C2(sz);
  iu::ImageCpu_8u_C3 im_cpu_8u_C3(sz);
  iu::ImageCpu_8u_C4 im_cpu_8u_C4(sz);
  iu::ImageCpu_32f_C1 im_cpu_32f_C1(sz);
  iu::ImageCpu_32f_C2 im_cpu_32f_C2(sz);
  iu::ImageCpu_32f_C3 im_cpu_32f_C3(sz);
  iu::ImageCpu_32f_C4 im_cpu_32f_C4(sz);

  // set values
  Npp8u set_value_8u = 1;
  Npp32f set_value_32f = 1.1f;

  iu::setValue(set_value_8u*1, &im_cpu_8u_C1, im_cpu_8u_C1.roi());
  iu::setValue(set_value_8u*2, &im_cpu_8u_C2, im_cpu_8u_C2.roi());
  iu::setValue(set_value_8u*3, &im_cpu_8u_C3, im_cpu_8u_C3.roi());
  iu::setValue(set_value_8u*4, &im_cpu_8u_C4, im_cpu_8u_C4.roi());
  iu::setValue(set_value_32f*1, &im_cpu_32f_C1, im_cpu_32f_C1.roi());
  iu::setValue(set_value_32f*2, &im_cpu_32f_C2, im_cpu_32f_C2.roi());
  iu::setValue(set_value_32f*3, &im_cpu_32f_C3, im_cpu_32f_C3.roi());
  iu::setValue(set_value_32f*4, &im_cpu_32f_C4, im_cpu_32f_C4.roi());
  {
    // check if set values are correct
    for (unsigned int y = 0; y<sz.height; ++y)
    {
      for (unsigned int x = 0; x<sz.width; ++x)
      {
        if( im_cpu_8u_C1.data(x,y)[0] != set_value_8u*1)
          return EXIT_FAILURE;
        if((im_cpu_8u_C2.data(x,y)[0] != set_value_8u*2) &&
           (im_cpu_8u_C2.data(x,y)[1] != set_value_8u*2))
          return EXIT_FAILURE;
        if((im_cpu_8u_C3.data(x,y)[0] != set_value_8u*3) &&
           (im_cpu_8u_C3.data(x,y)[1] != set_value_8u*3) &&
           (im_cpu_8u_C3.data(x,y)[2] != set_value_8u*3))
          return EXIT_FAILURE;
        if((im_cpu_8u_C4.data(x,y)[0] != set_value_8u*4) &&
           (im_cpu_8u_C4.data(x,y)[1] != set_value_8u*4) &&
           (im_cpu_8u_C4.data(x,y)[2] != set_value_8u*4) &&
           (im_cpu_8u_C4.data(x,y)[3] != set_value_8u*4))
          return EXIT_FAILURE;

        if( !almostEquals( im_cpu_32f_C1.data(x,y)[0], set_value_32f*1))
          return EXIT_FAILURE;
        if( !almostEquals(im_cpu_32f_C2.data(x,y)[0], set_value_32f*2) &&
            !almostEquals(im_cpu_32f_C2.data(x,y)[1], set_value_32f*2) )
          return EXIT_FAILURE;
        if( !almostEquals(im_cpu_32f_C3.data(x,y)[0], set_value_32f*3) &&
            !almostEquals(im_cpu_32f_C3.data(x,y)[1], set_value_32f*3) &&
            !almostEquals(im_cpu_32f_C3.data(x,y)[2], set_value_32f*3) )
          return EXIT_FAILURE;
        if( !almostEquals(im_cpu_32f_C4.data(x,y)[0], set_value_32f*4) &&
            !almostEquals(im_cpu_32f_C4.data(x,y)[1], set_value_32f*4) &&
            !almostEquals(im_cpu_32f_C4.data(x,y)[2], set_value_32f*4) &&
            !almostEquals(im_cpu_32f_C4.data(x,y)[3], set_value_32f*4) )
          return EXIT_FAILURE;
      }
    }
  }
  // copy test
  {
    iu::ImageCpu_8u_C1 cp_cpu_8u_C1(sz);
    iu::ImageCpu_8u_C2 cp_cpu_8u_C2(sz);
    iu::ImageCpu_8u_C3 cp_cpu_8u_C3(sz);
    iu::ImageCpu_8u_C4 cp_cpu_8u_C4(sz);
    iu::ImageCpu_32f_C1 cp_cpu_32f_C1(sz);
    iu::ImageCpu_32f_C2 cp_cpu_32f_C2(sz);
    iu::ImageCpu_32f_C3 cp_cpu_32f_C3(sz);
    iu::ImageCpu_32f_C4 cp_cpu_32f_C4(sz);

    iu::copy(&im_cpu_8u_C1, &cp_cpu_8u_C1);
    iu::copy(&im_cpu_8u_C2, &cp_cpu_8u_C2);
    iu::copy(&im_cpu_8u_C3, &cp_cpu_8u_C3);
    iu::copy(&im_cpu_8u_C4, &cp_cpu_8u_C4);
    iu::copy(&im_cpu_32f_C1, &cp_cpu_32f_C1);
    iu::copy(&im_cpu_32f_C2, &cp_cpu_32f_C2);
    iu::copy(&im_cpu_32f_C3, &cp_cpu_32f_C3);
    iu::copy(&im_cpu_32f_C4, &cp_cpu_32f_C4);

    // check if set values are correct
    for (unsigned int y = 0; y<sz.height; ++y)
    {
      for (unsigned int x = 0; x<sz.width; ++x)
      {
        if( cp_cpu_8u_C1.data(x,y)[0] != set_value_8u*1)
          return EXIT_FAILURE;
        if((cp_cpu_8u_C2.data(x,y)[0] != set_value_8u*2) &&
           (cp_cpu_8u_C2.data(x,y)[1] != set_value_8u*2))
          return EXIT_FAILURE;
        if((cp_cpu_8u_C3.data(x,y)[0] != set_value_8u*3) &&
           (cp_cpu_8u_C3.data(x,y)[1] != set_value_8u*3) &&
           (cp_cpu_8u_C3.data(x,y)[2] != set_value_8u*3))
          return EXIT_FAILURE;
        if((cp_cpu_8u_C4.data(x,y)[0] != set_value_8u*4) &&
           (cp_cpu_8u_C4.data(x,y)[1] != set_value_8u*4) &&
           (cp_cpu_8u_C4.data(x,y)[2] != set_value_8u*4) &&
           (cp_cpu_8u_C4.data(x,y)[3] != set_value_8u*4))
          return EXIT_FAILURE;

        if( !almostEquals( cp_cpu_32f_C1.data(x,y)[0], set_value_32f*1))
          return EXIT_FAILURE;
        if( !almostEquals(cp_cpu_32f_C2.data(x,y)[0], set_value_32f*2) &&
            !almostEquals(cp_cpu_32f_C2.data(x,y)[1], set_value_32f*2) )
          return EXIT_FAILURE;
        if( !almostEquals(cp_cpu_32f_C3.data(x,y)[0], set_value_32f*3) &&
            !almostEquals(cp_cpu_32f_C3.data(x,y)[1], set_value_32f*3) &&
            !almostEquals(cp_cpu_32f_C3.data(x,y)[2], set_value_32f*3) )
          return EXIT_FAILURE;
        if( !almostEquals(cp_cpu_32f_C4.data(x,y)[0], set_value_32f*4) &&
            !almostEquals(cp_cpu_32f_C4.data(x,y)[1], set_value_32f*4) &&
            !almostEquals(cp_cpu_32f_C4.data(x,y)[2], set_value_32f*4) &&
            !almostEquals(cp_cpu_32f_C4.data(x,y)[3], set_value_32f*4) )
          return EXIT_FAILURE;
      }
    }
  }

  std::cout << std::endl;
  std::cout << "**************************************************************************" << std::endl;
  std::cout << "*  Everything seem to be ok. -- All assertions passed.                   *" << std::endl;
  std::cout << "*  Look at the images and close the windows to derminate the unittests.  *" << std::endl;
  std::cout << "**************************************************************************" << std::endl;
  std::cout << std::endl;

  return EXIT_SUCCESS;
}
