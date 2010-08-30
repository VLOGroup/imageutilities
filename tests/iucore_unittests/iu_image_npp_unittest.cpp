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
 * Description : Unit tests for npp images
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
  std::cout << "Starting iu_image_npp_unittest ..." << std::endl;

  // test image size
  IuSize sz(79,63);
  IuSize sz2(120,1);
  bool check_size = sz == sz2;
  if(check_size)
    std::cout << "sz and sz2 are equal" << std::endl;
  else
    std::cout << "sz and sz2 are different" << std::endl;
  sz2 = sz;
  check_size = sz == sz2;
  if(check_size)
    std::cout << "sz and sz2 are equal" << std::endl;
  else
    std::cout << "sz and sz2 are different" << std::endl;

  iu::ImageNpp_8u_C1 im_npp_8u_C1(sz);
  iu::ImageNpp_8u_C2 im_npp_8u_C2(sz);
  iu::ImageNpp_8u_C3 im_npp_8u_C3(sz);
  iu::ImageNpp_8u_C4 im_npp_8u_C4(sz);
  iu::ImageNpp_32f_C1 im_npp_32f_C1(sz);
  iu::ImageNpp_32f_C2 im_npp_32f_C2(sz);
//  iu::ImageNpp_32f_C3 im_npp_32f_C3(sz);
  iu::ImageNpp_32f_C4 im_npp_32f_C4(sz);

  // set values
  {
    Npp8u set_value_8u = 2;
    Npp32f set_value_32f = 1.331f;

    iu::setValue(set_value_8u, &im_npp_8u_C1, im_npp_8u_C1.roi());
    iu::setValue(set_value_8u, &im_npp_8u_C2, im_npp_8u_C2.roi());
    iu::setValue(set_value_8u, &im_npp_8u_C4, im_npp_8u_C4.roi());
    iu::setValue(set_value_32f, &im_npp_32f_C1, im_npp_32f_C1.roi());
    iu::setValue(set_value_32f, &im_npp_32f_C2, im_npp_32f_C2.roi());
    iu::setValue(set_value_32f, &im_npp_32f_C4, im_npp_32f_C4.roi());

    // copy test (device -> host) and check if the copied values
    // are the same then the set values on the device side
    {
      iu::ImageCpu_8u_C1 cp_cpu_8u_C1(sz);
      iu::ImageCpu_8u_C2 cp_cpu_8u_C2(sz);
      iu::ImageCpu_8u_C4 cp_cpu_8u_C4(sz);
      iu::ImageCpu_32f_C1 cp_cpu_32f_C1(sz);
      iu::ImageCpu_32f_C2 cp_cpu_32f_C2(sz);
      iu::ImageCpu_32f_C4 cp_cpu_32f_C4(sz);

      iu::copy(&im_npp_8u_C1, &cp_cpu_8u_C1);
      iu::copy(&im_npp_8u_C2, &cp_cpu_8u_C2);
      iu::copy(&im_npp_8u_C4, &cp_cpu_8u_C4);
      iu::copy(&im_npp_32f_C1, &cp_cpu_32f_C1);
      iu::copy(&im_npp_32f_C2, &cp_cpu_32f_C2);
      iu::copy(&im_npp_32f_C4, &cp_cpu_32f_C4);

      // check if set values are correct
      for (unsigned int y = 0; y<sz.height; ++y)
      {
        for (unsigned int x = 0; x<sz.width; ++x)
        {
          if( cp_cpu_8u_C1.data(x,y)[0] != set_value_8u)
            return EXIT_FAILURE;
          if((cp_cpu_8u_C2.data(x,y)[0] != set_value_8u) &&
             (cp_cpu_8u_C2.data(x,y)[1] != set_value_8u))
            return EXIT_FAILURE;
          if((cp_cpu_8u_C4.data(x,y)[0] != set_value_8u) &&
             (cp_cpu_8u_C4.data(x,y)[1] != set_value_8u) &&
             (cp_cpu_8u_C4.data(x,y)[2] != set_value_8u) &&
             (cp_cpu_8u_C4.data(x,y)[3] != set_value_8u))
            return EXIT_FAILURE;

          if( !almostEquals( cp_cpu_32f_C1.data(x,y)[0], set_value_32f))
            return EXIT_FAILURE;
          if( !almostEquals(cp_cpu_32f_C2.data(x,y)[0], set_value_32f) &&
              !almostEquals(cp_cpu_32f_C2.data(x,y)[1], set_value_32f) )
            return EXIT_FAILURE;
          if( !almostEquals(cp_cpu_32f_C4.data(x,y)[0], set_value_32f) &&
              !almostEquals(cp_cpu_32f_C4.data(x,y)[1], set_value_32f) &&
              !almostEquals(cp_cpu_32f_C4.data(x,y)[2], set_value_32f) &&
              !almostEquals(cp_cpu_32f_C4.data(x,y)[3], set_value_32f) )
            return EXIT_FAILURE;
        }
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
