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
#include <QApplication>
#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <cuda_runtime.h>
#include <iucore.h>
#include <iumath.h>

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

  iu::ImageGpu_8u_C1 im_npp_8u_C1(sz);
  iu::ImageGpu_8u_C4 im_npp_8u_C4(sz);
  iu::ImageGpu_32f_C1 im_npp_32f_C1(sz);
  iu::ImageGpu_32f_C2 im_npp_32f_C2(sz);
  iu::ImageGpu_32f_C4 im_npp_32f_C4(sz);


  Npp8u set_value_8u = 2;
  Npp32f set_value_32f = 1.331f;

  iu::setValue(set_value_8u, &im_npp_8u_C1, im_npp_8u_C1.roi());
  iu::setValue(set_value_8u, &im_npp_8u_C4, im_npp_8u_C4.roi());
  iu::setValue(set_value_32f, &im_npp_32f_C1, im_npp_32f_C1.roi());
  iu::setValue(set_value_32f, &im_npp_32f_C2, im_npp_32f_C2.roi());
  iu::setValue(set_value_32f, &im_npp_32f_C4, im_npp_32f_C4.roi());

  // test minMax
  {
    Npp8u min_8u_C1, max_8u_C1;
    Npp8u min_8u_C4[4], max_8u_C4[4];
    Npp32f min_32f_C1, max_32f_C1;
    Npp32f min_32f_C4[4], max_32f_C4[4];

    iu::minMax(&im_npp_8u_C1, im_npp_8u_C1.roi(), min_8u_C1, max_8u_C1);
    if(min_8u_C1 != set_value_8u || max_8u_C1 != set_value_8u)
      return EXIT_FAILURE;
    iu::minMax(&im_npp_8u_C4, im_npp_8u_C4.roi(), min_8u_C4, max_8u_C4);
    if((min_8u_C4[0] != set_value_8u) || (max_8u_C4[0] != set_value_8u) ||
       (min_8u_C4[1] != set_value_8u) || (max_8u_C4[1] != set_value_8u) ||
       (min_8u_C4[2] != set_value_8u) || (max_8u_C4[2] != set_value_8u) ||
       (min_8u_C4[3] != set_value_8u) || (max_8u_C4[3] != set_value_8u) )
      return EXIT_FAILURE;

    iu::minMax(&im_npp_32f_C1, im_npp_32f_C1.roi(), min_32f_C1, max_32f_C1);
    if( !almostEquals(min_32f_C1, set_value_32f) ||
        !almostEquals(max_32f_C1, set_value_32f) )
      return EXIT_FAILURE;
    iu::minMax(&im_npp_32f_C4, im_npp_32f_C4.roi(), min_32f_C4, max_32f_C4);
    if(!almostEquals(min_32f_C4[0], set_value_32f) ||
       !almostEquals(max_32f_C4[0], set_value_32f) ||
       !almostEquals(min_32f_C4[1], set_value_32f) ||
       !almostEquals(max_32f_C4[1], set_value_32f) ||
       !almostEquals(min_32f_C4[2], set_value_32f) ||
       !almostEquals(max_32f_C4[2], set_value_32f) ||
       !almostEquals(min_32f_C4[3], set_value_32f) ||
       !almostEquals(max_32f_C4[3], set_value_32f) )
      return EXIT_FAILURE;
  } // end test minMax

  std::cout << "Testing minMax ok ..." << std::endl;

  // test summation
  {
    Npp64s sum_64s_C1;
    Npp64f sum_64f_C1;

    iu::summation(&im_npp_8u_C1, im_npp_8u_C1.roi(), sum_64s_C1);
    Npp64s desired_result_64s = sz.width*sz.height*set_value_8u;
    Npp64f desired_result_64f = sz.width*sz.height*set_value_32f;
    if(sum_64s_C1 != desired_result_64s)
      return EXIT_FAILURE;

    iu::summation(&im_npp_32f_C1, im_npp_32f_C1.roi(), sum_64f_C1);
    if( !almostEquals(sum_64f_C1, desired_result_64f, 10) )
      return EXIT_FAILURE;
  } // end test summation

  std::cout << "Testing summation ok ..." << std::endl;


  std::cout << std::endl;
  std::cout << "**************************************************************************" << std::endl;
  std::cout << "*  Everything seem to be ok. -- All assertions passed.                   *" << std::endl;
  std::cout << "*  Look at the images and close the windows to derminate the unittests.  *" << std::endl;
  std::cout << "**************************************************************************" << std::endl;
  std::cout << std::endl;

  return EXIT_SUCCESS;
}
