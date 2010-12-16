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
 * Description : Unit tests for gpu images
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
#include <iu/iucutil.h>

int main(int argc, char** argv)
{
  std::cout << "Starting iu_image_gpu_unittest ..." << std::endl;

  // test image size
  IuSize sz(79,63);

  iu::ImageGpu_8u_C1 im_gpu_8u_C1(sz);
  iu::ImageGpu_8u_C2 im_gpu_8u_C2(sz);
  iu::ImageGpu_8u_C3 im_gpu_8u_C3(sz);
  iu::ImageGpu_8u_C4 im_gpu_8u_C4(sz);
  iu::ImageGpu_32f_C1 im_gpu_32f_C1(sz);
  iu::ImageGpu_32f_C2 im_gpu_32f_C2(sz);
  iu::ImageGpu_32f_C3 im_gpu_32f_C3(sz);
  iu::ImageGpu_32f_C4 im_gpu_32f_C4(sz);

  unsigned char set_value_8u_C1 = 1;
  uchar2 set_value_8u_C2 = make_uchar2(2,2);
  uchar3 set_value_8u_C3 = make_uchar3(3,3,3);
  uchar4 set_value_8u_C4 = make_uchar4(4,4,4,4);
  float set_value_32f_C1 = 1.1f;
  float2 set_value_32f_C2 = make_float2(2.2f);
  float3 set_value_32f_C3 = make_float3(3.3f);
  float4 set_value_32f_C4 = make_float4(4.4f);

  // copy values back to cpu to compare the set values
  iu::ImageCpu_8u_C1 im_cpu_8u_C1(sz);
  iu::ImageCpu_8u_C2 im_cpu_8u_C2(sz);
  iu::ImageCpu_8u_C3 im_cpu_8u_C3(sz);
  iu::ImageCpu_8u_C4 im_cpu_8u_C4(sz);
  iu::ImageCpu_32f_C1 im_cpu_32f_C1(sz);
  iu::ImageCpu_32f_C2 im_cpu_32f_C2(sz);
  iu::ImageCpu_32f_C3 im_cpu_32f_C3(sz);
  iu::ImageCpu_32f_C4 im_cpu_32f_C4(sz);


  // set values on cpu and copy to gpu and back again
  {
    std::cout << "Testing copy. setValue on cpu (should work because of previous test) and copy forth and back" << std::endl;

    iu::setValue(set_value_8u_C1, &im_cpu_8u_C1, im_cpu_8u_C1.roi());
    iu::setValue(set_value_8u_C2, &im_cpu_8u_C2, im_cpu_8u_C2.roi());
    iu::setValue(set_value_8u_C3, &im_cpu_8u_C3, im_cpu_8u_C3.roi());
    iu::setValue(set_value_8u_C4, &im_cpu_8u_C4, im_cpu_8u_C4.roi());
    iu::setValue(set_value_32f_C1, &im_cpu_32f_C1, im_cpu_32f_C1.roi());
    iu::setValue(set_value_32f_C2, &im_cpu_32f_C2, im_cpu_32f_C2.roi());
    iu::setValue(set_value_32f_C3, &im_cpu_32f_C3, im_cpu_32f_C3.roi());
    iu::setValue(set_value_32f_C4, &im_cpu_32f_C4, im_cpu_32f_C4.roi());

    std::cout << "  copy cpu -> gpu ..." << std::endl;
    iu::copy(&im_cpu_8u_C1, &im_gpu_8u_C1);
    iu::copy(&im_cpu_8u_C2, &im_gpu_8u_C2);
    iu::copy(&im_cpu_8u_C3, &im_gpu_8u_C3);
    iu::copy(&im_cpu_8u_C4, &im_gpu_8u_C4);
    iu::copy(&im_cpu_32f_C1, &im_gpu_32f_C1);
    iu::copy(&im_cpu_32f_C2, &im_gpu_32f_C2);
    iu::copy(&im_cpu_32f_C3, &im_gpu_32f_C3);
    iu::copy(&im_cpu_32f_C4, &im_gpu_32f_C4);
    std::cout << "  copy gpu -> cpu ..." << std::endl;
    iu::copy(&im_gpu_8u_C1, &im_cpu_8u_C1);
    iu::copy(&im_gpu_8u_C2, &im_cpu_8u_C2);
    iu::copy(&im_gpu_8u_C3, &im_cpu_8u_C3);
    iu::copy(&im_gpu_8u_C4, &im_cpu_8u_C4);
    iu::copy(&im_gpu_32f_C1, &im_cpu_32f_C1);
    iu::copy(&im_gpu_32f_C2, &im_cpu_32f_C2);
    iu::copy(&im_gpu_32f_C3, &im_cpu_32f_C3);
    iu::copy(&im_gpu_32f_C4, &im_cpu_32f_C4);

    std::cout << "  check copied values on cpu ..." << std::endl;
    // check if set values are correct
    for (unsigned int y = 0; y<sz.height; ++y)
    {
      for (unsigned int x = 0; x<sz.width; ++x)
      {
        // 8-bit
        if( *im_cpu_8u_C1.data(x,y) != set_value_8u_C1)
          return EXIT_FAILURE;
        if( *im_cpu_8u_C2.data(x,y) != set_value_8u_C2)
          return EXIT_FAILURE;
        if( *im_cpu_8u_C3.data(x,y) != set_value_8u_C3)
          return EXIT_FAILURE;
        if( *im_cpu_8u_C4.data(x,y) != set_value_8u_C4)
          return EXIT_FAILURE;

        // 32-bit
        if( *im_cpu_32f_C1.data(x,y) != set_value_32f_C1)
          return EXIT_FAILURE;
        if( *im_cpu_32f_C2.data(x,y) != set_value_32f_C2)
          return EXIT_FAILURE;
        if( *im_cpu_32f_C3.data(x,y) != set_value_32f_C3)
          return EXIT_FAILURE;
        if( *im_cpu_32f_C4.data(x,y) != set_value_32f_C4)
          return EXIT_FAILURE;
      }
    }
  }

  // set values on gpu
  {
    std::cout << "Testing setValue on gpu (implecitely testing copy gpu->cpu) ..." << std::endl;

    iu::setValue(set_value_8u_C1, &im_gpu_8u_C1, im_gpu_8u_C1.roi());
    iu::setValue(set_value_8u_C2, &im_gpu_8u_C2, im_gpu_8u_C2.roi());
    iu::setValue(set_value_8u_C3, &im_gpu_8u_C3, im_gpu_8u_C3.roi());
    iu::setValue(set_value_8u_C4, &im_gpu_8u_C4, im_gpu_8u_C4.roi());
    iu::setValue(set_value_32f_C1, &im_gpu_32f_C1, im_gpu_32f_C1.roi());
    iu::setValue(set_value_32f_C2, &im_gpu_32f_C2, im_gpu_32f_C2.roi());
    iu::setValue(set_value_32f_C3, &im_gpu_32f_C3, im_gpu_32f_C3.roi());
    iu::setValue(set_value_32f_C4, &im_gpu_32f_C4, im_gpu_32f_C4.roi());

    std::cout << "Copy gpu images to cpu for checking the set values." << std::endl;
    iu::copy(&im_gpu_8u_C1, &im_cpu_8u_C1);
    iu::copy(&im_gpu_8u_C2, &im_cpu_8u_C2);
    iu::copy(&im_gpu_8u_C3, &im_cpu_8u_C3);
    iu::copy(&im_gpu_8u_C4, &im_cpu_8u_C4);
    iu::copy(&im_gpu_32f_C1, &im_cpu_32f_C1);
    iu::copy(&im_gpu_32f_C2, &im_cpu_32f_C2);
    iu::copy(&im_gpu_32f_C3, &im_cpu_32f_C3);
    iu::copy(&im_gpu_32f_C4, &im_cpu_32f_C4);

    // check if set values are correct
    for (unsigned int y = 0; y<sz.height; ++y)
    {
      for (unsigned int x = 0; x<sz.width; ++x)
      {
        // 8-bit
        if( *im_cpu_8u_C1.data(x,y) != set_value_8u_C1)
          return EXIT_FAILURE;
        if( *im_cpu_8u_C2.data(x,y) != set_value_8u_C2)
          return EXIT_FAILURE;
        if( *im_cpu_8u_C3.data(x,y) != set_value_8u_C3)
          return EXIT_FAILURE;
        if( *im_cpu_8u_C4.data(x,y) != set_value_8u_C4)
          return EXIT_FAILURE;

        // 32-bit
        if( *im_cpu_32f_C1.data(x,y) != set_value_32f_C1)
          return EXIT_FAILURE;
        if( *im_cpu_32f_C2.data(x,y) != set_value_32f_C2)
          return EXIT_FAILURE;
        if( *im_cpu_32f_C3.data(x,y) != set_value_32f_C3)
          return EXIT_FAILURE;
        if( *im_cpu_32f_C4.data(x,y) != set_value_32f_C4)
          return EXIT_FAILURE;
      }
    }
  }

  // copy gpu -> gpu test
  {
    std::cout << "testing copy gpu -> gpu  ..." << std::endl;

    iu::ImageGpu_8u_C1 cp_gpu_8u_C1(sz);
    iu::ImageGpu_8u_C2 cp_gpu_8u_C2(sz);
    iu::ImageGpu_8u_C3 cp_gpu_8u_C3(sz);
    iu::ImageGpu_8u_C4 cp_gpu_8u_C4(sz);
    iu::ImageGpu_32f_C1 cp_gpu_32f_C1(sz);
    iu::ImageGpu_32f_C2 cp_gpu_32f_C2(sz);
    iu::ImageGpu_32f_C3 cp_gpu_32f_C3(sz);
    iu::ImageGpu_32f_C4 cp_gpu_32f_C4(sz);

    iu::copy(&im_gpu_8u_C1, &cp_gpu_8u_C1);
    iu::copy(&im_gpu_8u_C2, &cp_gpu_8u_C2);
    iu::copy(&im_gpu_8u_C3, &cp_gpu_8u_C3);
    iu::copy(&im_gpu_8u_C4, &cp_gpu_8u_C4);
    iu::copy(&im_gpu_32f_C1, &cp_gpu_32f_C1);
    iu::copy(&im_gpu_32f_C2, &cp_gpu_32f_C2);
    iu::copy(&im_gpu_32f_C3, &cp_gpu_32f_C3);
    iu::copy(&im_gpu_32f_C4, &cp_gpu_32f_C4);

    iu::copy(&cp_gpu_8u_C1, &im_cpu_8u_C1);
    iu::copy(&cp_gpu_8u_C2, &im_cpu_8u_C2);
    iu::copy(&cp_gpu_8u_C3, &im_cpu_8u_C3);
    iu::copy(&cp_gpu_8u_C4, &im_cpu_8u_C4);
    iu::copy(&cp_gpu_32f_C1, &im_cpu_32f_C1);
    iu::copy(&cp_gpu_32f_C2, &im_cpu_32f_C2);
    iu::copy(&cp_gpu_32f_C3, &im_cpu_32f_C3);
    iu::copy(&cp_gpu_32f_C4, &im_cpu_32f_C4);

    // check if set values are correct
    for (unsigned int y = 0; y<sz.height; ++y)
    {
      for (unsigned int x = 0; x<sz.width; ++x)
      {
        // 8-bit
        if( *im_cpu_8u_C1.data(x,y) != set_value_8u_C1)
          return EXIT_FAILURE;
        if( *im_cpu_8u_C2.data(x,y) != set_value_8u_C2)
          return EXIT_FAILURE;
        if( *im_cpu_8u_C3.data(x,y) != set_value_8u_C3)
          return EXIT_FAILURE;
        if( *im_cpu_8u_C4.data(x,y) != set_value_8u_C4)
          return EXIT_FAILURE;

        // 32-bit
        if( *im_cpu_32f_C1.data(x,y) != set_value_32f_C1)
          return EXIT_FAILURE;
        if( *im_cpu_32f_C2.data(x,y) != set_value_32f_C2)
          return EXIT_FAILURE;
        if( *im_cpu_32f_C3.data(x,y) != set_value_32f_C3)
          return EXIT_FAILURE;
        if( *im_cpu_32f_C4.data(x,y) != set_value_32f_C4)
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
