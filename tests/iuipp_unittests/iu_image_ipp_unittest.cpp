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
#include <cutil_math.h>

#include <iu/iuipp.h>

int main(int argc, char** argv)
{
  std::cout << "Starting iu_image_ipp_unittest ..." << std::endl;

  // test image size
  IuSize sz(79,63);

  iu::ImageIpp_8u_C1 im_ipp_8u_C1(sz);
  iu::ImageIpp_8u_C2 im_ipp_8u_C2(sz);
  iu::ImageIpp_8u_C3 im_ipp_8u_C3(sz);
  iu::ImageIpp_8u_C4 im_ipp_8u_C4(sz);
  iu::ImageIpp_32f_C1 im_ipp_32f_C1(sz);
  iu::ImageIpp_32f_C2 im_ipp_32f_C2(sz);
  iu::ImageIpp_32f_C3 im_ipp_32f_C3(sz);
  iu::ImageIpp_32f_C4 im_ipp_32f_C4(sz);

//  unsigned char set_value_8u_C1 = 1;
//  uchar2 set_value_8u_C2 = make_uchar2(2,2);
//  uchar3 set_value_8u_C3 = make_uchar3(3,3,3);
//  uchar4 set_value_8u_C4 = make_uchar4(4,4,4,4);

//  float set_value_32f_C1 = 1.1f;
//  float2 set_value_32f_C2 = make_float2(2.2f);
//  float3 set_value_32f_C3 = make_float3(3.3f);
//  float4 set_value_32f_C4 = make_float4(4.4f);

//  // set values test
//  {
//    std::cout << "testing setValue on cpu ..." << std::endl;

//    iu::setValue(set_value_8u_C1, &im_ipp_8u_C1, im_ipp_8u_C1.roi());
//    iu::setValue(set_value_8u_C2, &im_ipp_8u_C2, im_ipp_8u_C2.roi());
//    iu::setValue(set_value_8u_C3, &im_ipp_8u_C3, im_ipp_8u_C3.roi());
//    iu::setValue(set_value_8u_C4, &im_ipp_8u_C4, im_ipp_8u_C4.roi());
//    iu::setValue(set_value_32f_C1, &im_ipp_32f_C1, im_ipp_32f_C1.roi());
//    iu::setValue(set_value_32f_C2, &im_ipp_32f_C2, im_ipp_32f_C2.roi());
//    iu::setValue(set_value_32f_C3, &im_ipp_32f_C3, im_ipp_32f_C3.roi());
//    iu::setValue(set_value_32f_C4, &im_ipp_32f_C4, im_ipp_32f_C4.roi());

//    // check if set values are correct
//    for (unsigned int y = 0; y<sz.height; ++y)
//    {
//      for (unsigned int x = 0; x<sz.width; ++x)
//      {
//        // 8-bit
//        if( *im_ipp_8u_C1.data(x,y) != set_value_8u_C1)
//          return EXIT_FAILURE;
//        if( *im_ipp_8u_C2.data(x,y) != set_value_8u_C2)
//          return EXIT_FAILURE;
//        if( *im_ipp_8u_C3.data(x,y) != set_value_8u_C3)
//          return EXIT_FAILURE;
//        if( *im_ipp_8u_C4.data(x,y) != set_value_8u_C4)
//          return EXIT_FAILURE;

//        // 32-bit
//        if( *im_ipp_32f_C1.data(x,y) != set_value_32f_C1)
//          return EXIT_FAILURE;
//        if( *im_ipp_32f_C2.data(x,y) != set_value_32f_C2)
//          return EXIT_FAILURE;
//        if( *im_ipp_32f_C3.data(x,y) != set_value_32f_C3)
//          return EXIT_FAILURE;
//        if( *im_ipp_32f_C4.data(x,y) != set_value_32f_C4)
//          return EXIT_FAILURE;
//      }
//    }
//  }

//  // copy test
//  {
//    std::cout << "testing copy cpu -> cpu ..." << std::endl;

//    iu::ImageIpp_8u_C1 cp_ipp_8u_C1(sz);
//    iu::ImageIpp_8u_C2 cp_ipp_8u_C2(sz);
//    iu::ImageIpp_8u_C3 cp_ipp_8u_C3(sz);
//    iu::ImageIpp_8u_C4 cp_ipp_8u_C4(sz);
//    iu::ImageIpp_32f_C1 cp_ipp_32f_C1(sz);
//    iu::ImageIpp_32f_C2 cp_ipp_32f_C2(sz);
//    iu::ImageIpp_32f_C3 cp_ipp_32f_C3(sz);
//    iu::ImageIpp_32f_C4 cp_ipp_32f_C4(sz);

//    iu::copy(&im_ipp_8u_C1, &cp_ipp_8u_C1);
//    iu::copy(&im_ipp_8u_C2, &cp_ipp_8u_C2);
//    iu::copy(&im_ipp_8u_C3, &cp_ipp_8u_C3);
//    iu::copy(&im_ipp_8u_C4, &cp_ipp_8u_C4);
//    iu::copy(&im_ipp_32f_C1, &cp_ipp_32f_C1);
//    iu::copy(&im_ipp_32f_C2, &cp_ipp_32f_C2);
//    iu::copy(&im_ipp_32f_C3, &cp_ipp_32f_C3);
//    iu::copy(&im_ipp_32f_C4, &cp_ipp_32f_C4);

//    // check if set values are correct
//    for (unsigned int y = 0; y<sz.height; ++y)
//    {
//      for (unsigned int x = 0; x<sz.width; ++x)
//      {
//        if( *cp_ipp_8u_C1.data(x,y) != set_value_8u_C1)
//          return EXIT_FAILURE;
//        if( *cp_ipp_8u_C2.data(x,y) != set_value_8u_C2)
//          return EXIT_FAILURE;
//        if( *cp_ipp_8u_C3.data(x,y) != set_value_8u_C3)
//          return EXIT_FAILURE;
//        if( *cp_ipp_8u_C4.data(x,y) != set_value_8u_C4)
//          return EXIT_FAILURE;

//        if( *cp_ipp_32f_C1.data(x,y) != set_value_32f_C1)
//          return EXIT_FAILURE;
//        if( *cp_ipp_32f_C2.data(x,y) != set_value_32f_C2)
//          return EXIT_FAILURE;
//        if( *cp_ipp_32f_C3.data(x,y) != set_value_32f_C3)
//          return EXIT_FAILURE;
//        if( *cp_ipp_32f_C4.data(x,y) != set_value_32f_C4)
//          return EXIT_FAILURE;
//      }
//    }
//  }

  std::cout << std::endl;
  std::cout << "**************************************************************************" << std::endl;
  std::cout << "*  Everything seem to be ok. -- All assertions passed.                   *" << std::endl;
  std::cout << "*  Look at the images and close the windows to derminate the unittests.  *" << std::endl;
  std::cout << "**************************************************************************" << std::endl;
  std::cout << std::endl;

  return EXIT_SUCCESS;
}
