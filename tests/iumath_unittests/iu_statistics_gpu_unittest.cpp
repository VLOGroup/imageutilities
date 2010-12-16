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
//#include <cutil_math.h>
#include <iucore.h>
#include <iumath.h>
#include <iucutil.h>

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
  std::cout << "Starting iu_image_gpu_unittest ..." << std::endl;

  // test image size
  IuSize sz(922,307);

  iu::ImageGpu_8u_C1 im_gpu_8u_C1(sz);
  iu::ImageGpu_8u_C4 im_gpu_8u_C4(sz);
  iu::ImageGpu_32f_C1 im_gpu_32f_C1(sz);
  iu::ImageGpu_32f_C2 im_gpu_32f_C2(sz);
  iu::ImageGpu_32f_C4 im_gpu_32f_C4(sz);

  unsigned char min_value_8u_C1 = 1;
  unsigned char max_value_8u_C1 = 0xff;
  uchar4 set_value_8u_C4 = make_uchar4(4);
  float set_value_32f_C1 = 1.331f;
  float2 set_value_32f_C2 = make_float2(2.331f);
  float4 set_value_32f_C4 = make_float4(4.331f);

  iu::setValue(min_value_8u_C1, &im_gpu_8u_C1, im_gpu_8u_C1.roi());
  iu::setValue(max_value_8u_C1, &im_gpu_8u_C1,
               IuRect(sz.width-10, sz.height-10, sz.width-8, sz.height-8));
  iu::setValue(set_value_8u_C4, &im_gpu_8u_C4, im_gpu_8u_C4.roi());
  iu::setValue(set_value_32f_C1, &im_gpu_32f_C1, im_gpu_32f_C1.roi());
  iu::setValue(set_value_32f_C2, &im_gpu_32f_C2, im_gpu_32f_C2.roi());
  iu::setValue(set_value_32f_C4, &im_gpu_32f_C4, im_gpu_32f_C4.roi());

  // test minMax
  {
    unsigned char min_8u_C1, max_8u_C1;
    uchar4 min_8u_C4, max_8u_C4;
    float min_32f_C1, max_32f_C1;
    float2 min_32f_C2, max_32f_C2;
    float4 min_32f_C4, max_32f_C4;

    iu::minMax(&im_gpu_8u_C1, im_gpu_8u_C1.roi(), min_8u_C1, max_8u_C1);
    printf("min, max 8u_C1: %d/%d (desired:%d/%d)\n",
           min_8u_C1, max_8u_C1, min_value_8u_C1, max_value_8u_C1);
    if(min_8u_C1 != min_value_8u_C1 || max_8u_C1 != max_value_8u_C1)
      return EXIT_FAILURE;

    printf("min, max 8u_C4:\n");
    iu::minMax(&im_gpu_8u_C4, im_gpu_8u_C4.roi(), min_8u_C4, max_8u_C4);
    if(min_8u_C4 != set_value_8u_C4 || max_8u_C4 != set_value_8u_C4)
      return EXIT_FAILURE;

    printf("min, max 32f_C1:\n");
    iu::minMax(&im_gpu_32f_C1, im_gpu_32f_C1.roi(), min_32f_C1, max_32f_C1);
    if( !almostEquals(min_32f_C1, set_value_32f_C1) ||
        !almostEquals(max_32f_C1, set_value_32f_C1) )
      return EXIT_FAILURE;

    printf("min, max 32f_C2:\n");
    iu::minMax(&im_gpu_32f_C2, im_gpu_32f_C2.roi(), min_32f_C2, max_32f_C2);
    if(min_32f_C2 != set_value_32f_C2 || max_32f_C2 != set_value_32f_C2)
      return EXIT_FAILURE;

    printf("min, max 32f_C4:\n");
    iu::minMax(&im_gpu_32f_C4, im_gpu_32f_C4.roi(), min_32f_C4, max_32f_C4);
    if(min_32f_C4 != set_value_32f_C4 || max_32f_C4 != set_value_32f_C4)
      return EXIT_FAILURE;
  } // end test minMax

  std::cout << "Testing minMax ok ..." << std::endl;

  // test summation
  {
    long sum_64s_C1;
    double sum_64f_C1;

    iu::summation(&im_gpu_8u_C1, im_gpu_8u_C1.roi(), sum_64s_C1);
    long desired_result_64s = sz.width*sz.height*max_value_8u_C1;
    double desired_result_64f = sz.width*sz.height*set_value_32f_C1;
    if(sum_64s_C1 != desired_result_64s)
      return EXIT_FAILURE;

    iu::summation(&im_gpu_32f_C1, im_gpu_32f_C1.roi(), sum_64f_C1);
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
