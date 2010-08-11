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
 * Description : Unit tests for math/arithmetic module
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

// system includes
#include <iostream>
#include <cuda_runtime.h>
#include <iucore.h>
#include <iumath.h>

int main(int argc, char** argv)
{
  std::cout << "Starting iu_math_arithmetic_unittest ..." << std::endl;
  // 'parameters'
  IuSize sz(78,63);

  /* *************************************************************************
    NPP IMAGES; 32-bit WEIGHTED ADD
  * *************************************************************************/
  {
    // Npp images
    iu::ImageNpp_32f_C1 src1_npp_32f_C1(sz);
    iu::ImageNpp_32f_C1 src2_npp_32f_C1(sz);
    iu::ImageNpp_32f_C1 dst_npp_32f_C1(sz);
    iu::ImageCpu_32f_C1 chk_ipp_32f_C1(sz);

    // set an init value
    Npp32f set_value1_C1 = 7.72f, factor1_C1 = 1.2;
    Npp32f set_value2_C1 = 2.72f, factor2_C1 = -0.8f;
    iu::setValue(set_value1_C1, &src1_npp_32f_C1, src1_npp_32f_C1.roi());
    iu::setValue(set_value2_C1, &src2_npp_32f_C1, src2_npp_32f_C1.roi());

    // weighted add
    iu::addWeighted(&src1_npp_32f_C1, factor1_C1, &src2_npp_32f_C1, factor2_C1, &dst_npp_32f_C1, dst_npp_32f_C1.roi());
    iu::copy(&dst_npp_32f_C1, dst_npp_32f_C1.roi(), &chk_ipp_32f_C1, chk_ipp_32f_C1.roi());
    // iterate over every pixel and check its correctness
    Npp32f desired_value = set_value1_C1*factor1_C1 + set_value2_C1*factor2_C1;
    for (unsigned int x = 0; x<sz.width; ++x)
    {
      for (unsigned int y = 0; y<sz.height; ++y)
      {
        // 1-channel
        assert(chk_ipp_32f_C1.data()[y*chk_ipp_32f_C1.stride() + x] == desired_value);
      }
    }

  }

  /* *************************************************************************
    IPP IMAGES; 8-bit NOT-IN-PLACE MULTIPLICATION
  * *************************************************************************/
  {
    // Ipp images
    iu::ImageCpu_8u_C1 src_ipp_8u_C1(sz);
    iu::ImageCpu_8u_C3 src_ipp_8u_C3(sz);
    iu::ImageCpu_8u_C4 src_ipp_8u_C4(sz);
    iu::ImageCpu_8u_C1 dst_ipp_8u_C1(sz);
    iu::ImageCpu_8u_C3 dst_ipp_8u_C3(sz);
    iu::ImageCpu_8u_C4 dst_ipp_8u_C4(sz);

    // set an init value
    Npp8u set_value_C1 = 8, factor_C1 = 3;
    Npp8u set_value_C3[3] = {8,8,8}, factor_C3[3] = {3,3,3};
    Npp8u set_value_C4[4] = {8,8,8,8}, factor_C4[4] = {3,3,3,3};
    iu::setValue(set_value_C1, &src_ipp_8u_C1, src_ipp_8u_C1.roi());
    iu::setValue(set_value_C3, &src_ipp_8u_C3, src_ipp_8u_C3.roi());
    iu::setValue(set_value_C4, &src_ipp_8u_C4, src_ipp_8u_C4.roi());

    // multiply
    iu::mulC(&src_ipp_8u_C1, factor_C1, &dst_ipp_8u_C1, dst_ipp_8u_C1.roi());
    iu::mulC(&src_ipp_8u_C3, factor_C3, &dst_ipp_8u_C3, dst_ipp_8u_C3.roi());
    iu::mulC(&src_ipp_8u_C4, factor_C4, &dst_ipp_8u_C4, dst_ipp_8u_C4.roi());

    // iterate over every pixel and check its correctness
    Npp8u desired_value = set_value_C1*factor_C1;
    for (unsigned int x = 0; x<sz.width; ++x)
    {
      for (unsigned int y = 0; y<sz.height; ++y)
      {
        // 1-channel
        assert(dst_ipp_8u_C1.data()[y*dst_ipp_8u_C1.stride() + x] == desired_value);

        // 3-channel
        assert(dst_ipp_8u_C3.data()[y*dst_ipp_8u_C3.stride() + x] == desired_value);
        assert(dst_ipp_8u_C3.data()[y*dst_ipp_8u_C3.stride() + x+1] == desired_value);
        assert(dst_ipp_8u_C3.data()[y*dst_ipp_8u_C3.stride() + x+2] == desired_value);

        // 4-channel
        assert(dst_ipp_8u_C4.data()[y*dst_ipp_8u_C4.stride() + x] == desired_value);
        assert(dst_ipp_8u_C4.data()[y*dst_ipp_8u_C4.stride() + x+1] == desired_value);
        assert(dst_ipp_8u_C4.data()[y*dst_ipp_8u_C4.stride() + x+2] == desired_value);
        assert(dst_ipp_8u_C4.data()[y*dst_ipp_8u_C4.stride() + x+3] == desired_value);
      }
    }
  }

  /* *************************************************************************
    IPP IMAGES; 32-bit NOT-IN-PLACE MULTIPLICATION
  * *************************************************************************/
  {
    // Ipp images
    iu::ImageCpu_32f_C1 src_ipp_32f_C1(sz);
    iu::ImageCpu_32f_C3 src_ipp_32f_C3(sz);
    iu::ImageCpu_32f_C4 src_ipp_32f_C4(sz);
    iu::ImageCpu_32f_C1 dst_ipp_32f_C1(sz);
    iu::ImageCpu_32f_C3 dst_ipp_32f_C3(sz);
    iu::ImageCpu_32f_C4 dst_ipp_32f_C4(sz);

    // set an init value
    Npp32f set_value_C1 = 7.72f, factor_C1 = -0.24f;
    Npp32f set_value_C3[3] = {7.72f,7.72f,7.72f}, factor_C3[3] = {-0.24f,-0.24f,-0.24f};
    Npp32f set_value_C4[4] = {7.72f,7.72f,7.72f,7.72f}, factor_C4[4] = {-0.24f,-0.24f,-0.24f,-0.24f};
    iu::setValue(set_value_C1, &src_ipp_32f_C1, src_ipp_32f_C1.roi());
    iu::setValue(set_value_C3, &src_ipp_32f_C3, src_ipp_32f_C3.roi());
    iu::setValue(set_value_C4, &src_ipp_32f_C4, src_ipp_32f_C4.roi());

    // multiply
    iu::mulC(&src_ipp_32f_C1, factor_C1, &dst_ipp_32f_C1, dst_ipp_32f_C1.roi());
    iu::mulC(&src_ipp_32f_C3, factor_C3, &dst_ipp_32f_C3, dst_ipp_32f_C3.roi());
    iu::mulC(&src_ipp_32f_C4, factor_C4, &dst_ipp_32f_C4, dst_ipp_32f_C4.roi());

    // iterate over every pixel and check its correctness
    Npp32f desired_value = set_value_C1*factor_C1;
    for (unsigned int x = 0; x<sz.width; ++x)
    {
      for (unsigned int y = 0; y<sz.height; ++y)
      {
        // 1-channel
        assert(dst_ipp_32f_C1.data()[y*dst_ipp_32f_C1.stride() + x] == desired_value);

        // 3-channel
        assert(dst_ipp_32f_C3.data()[y*dst_ipp_32f_C3.stride() + x] == desired_value);
        assert(dst_ipp_32f_C3.data()[y*dst_ipp_32f_C3.stride() + x+1] == desired_value);
        assert(dst_ipp_32f_C3.data()[y*dst_ipp_32f_C3.stride() + x+2] == desired_value);

        // 4-channel
        assert(dst_ipp_32f_C4.data()[y*dst_ipp_32f_C4.stride() + x] == desired_value);
        assert(dst_ipp_32f_C4.data()[y*dst_ipp_32f_C4.stride() + x+1] == desired_value);
        assert(dst_ipp_32f_C4.data()[y*dst_ipp_32f_C4.stride() + x+2] == desired_value);
        assert(dst_ipp_32f_C4.data()[y*dst_ipp_32f_C4.stride() + x+3] == desired_value);
      }
    }
  }



  /* *************************************************************************
    Npp IMAGES; 8-bit NOT-IN-PLACE MULTIPLICATION
  * *************************************************************************/
  {
    // Npp images
    iu::ImageNpp_8u_C1 src_npp_8u_C1(sz);
    iu::ImageNpp_8u_C4 src_npp_8u_C4(sz);
    iu::ImageNpp_8u_C1 dst_npp_8u_C1(sz);
    iu::ImageNpp_8u_C4 dst_npp_8u_C4(sz);
    iu::ImageCpu_8u_C1 chk_ipp_8u_C1(sz);
    iu::ImageCpu_8u_C4 chk_ipp_8u_C4(sz);

    // set an init value
    Npp8u set_value_C1 = 8, factor_C1 = 3;
    Npp8u set_value_C4[4] = {8,8,8,8}, factor_C4[4] = {3,3,3,3};
    iu::setValue(set_value_C1, &src_npp_8u_C1, src_npp_8u_C1.roi());
    iu::setValue(set_value_C4, &src_npp_8u_C4, src_npp_8u_C4.roi());

    // multiply
    iu::mulC(&src_npp_8u_C1, factor_C1, &dst_npp_8u_C1, dst_npp_8u_C1.roi());
    iu::mulC(&src_npp_8u_C4, factor_C4, &dst_npp_8u_C4, dst_npp_8u_C4.roi());

    iu::copy(&dst_npp_8u_C1, dst_npp_8u_C1.roi(), &chk_ipp_8u_C1, chk_ipp_8u_C1.roi());
    iu::copy(&dst_npp_8u_C4, dst_npp_8u_C4.roi(), &chk_ipp_8u_C4, chk_ipp_8u_C4.roi());
    // iterate over every pixel and check its correctness
    Npp8u desired_value = set_value_C1*factor_C1;
    for (unsigned int x = 0; x<sz.width; ++x)
    {
      for (unsigned int y = 0; y<sz.height; ++y)
      {
        // 1-channel
        assert(chk_ipp_8u_C1.data()[y*chk_ipp_8u_C1.stride() + x] == desired_value);

        // 4-channel
        assert(chk_ipp_8u_C4.data()[y*chk_ipp_8u_C4.stride() + x] == desired_value);
        assert(chk_ipp_8u_C4.data()[y*chk_ipp_8u_C4.stride() + x+1] == desired_value);
        assert(chk_ipp_8u_C4.data()[y*chk_ipp_8u_C4.stride() + x+2] == desired_value);
        assert(chk_ipp_8u_C4.data()[y*chk_ipp_8u_C4.stride() + x+3] == desired_value);
      }
    }
  }

  /* *************************************************************************
    Npp IMAGES; 32-bit NOT-IN-PLACE MULTIPLICATION
  * *************************************************************************/
  {
    // Npp images
    iu::ImageNpp_32f_C1 src_npp_32f_C1(sz);
    iu::ImageNpp_32f_C4 src_npp_32f_C4(sz);
    iu::ImageNpp_32f_C1 dst_npp_32f_C1(sz);
    iu::ImageNpp_32f_C4 dst_npp_32f_C4(sz);
    iu::ImageCpu_32f_C1 chk_ipp_32f_C1(sz);
    iu::ImageCpu_32f_C4 chk_ipp_32f_C4(sz);

    // set an init value
    Npp32f set_value_C1 = 7.72f, factor_C1 = -0.24f;
    Npp32f set_value_C4[4] = {7.72f,7.72f,7.72f,7.72f}, factor_C4[4] = {-0.24f,-0.24f,-0.24f,-0.24f};
    iu::setValue(set_value_C1, &src_npp_32f_C1, src_npp_32f_C1.roi());
    iu::setValue(set_value_C4, &src_npp_32f_C4, src_npp_32f_C4.roi());

    // multiply
    iu::mulC(&src_npp_32f_C1, factor_C1, &dst_npp_32f_C1, dst_npp_32f_C1.roi());
    iu::mulC(&src_npp_32f_C4, factor_C4, &dst_npp_32f_C4, dst_npp_32f_C4.roi());

    iu::copy(&dst_npp_32f_C1, dst_npp_32f_C1.roi(), &chk_ipp_32f_C1, chk_ipp_32f_C1.roi());
    iu::copy(&dst_npp_32f_C4, dst_npp_32f_C4.roi(), &chk_ipp_32f_C4, chk_ipp_32f_C4.roi());
    // iterate over every pixel and check its correctness
    Npp32f desired_value = set_value_C1*factor_C1;
    for (unsigned int x = 0; x<sz.width; ++x)
    {
      for (unsigned int y = 0; y<sz.height; ++y)
      {
        // 1-channel
        assert(chk_ipp_32f_C1.data()[y*chk_ipp_32f_C1.stride() + x] == desired_value);

        // 4-channel
        assert(chk_ipp_32f_C4.data()[y*chk_ipp_32f_C4.stride() + x] == desired_value);
        assert(chk_ipp_32f_C4.data()[y*chk_ipp_32f_C4.stride() + x+1] == desired_value);
        assert(chk_ipp_32f_C4.data()[y*chk_ipp_32f_C4.stride() + x+2] == desired_value);
        assert(chk_ipp_32f_C4.data()[y*chk_ipp_32f_C4.stride() + x+3] == desired_value);
      }
    }
  }



  std::cout << std::endl;
  std::cout << "**************************************************************************" << std::endl;
  std::cout << "*  Everything seem to be ok. -- All assertions passed.                   *" << std::endl;
//  std::cout << "*  Look at the images and close the windows to derminate the unittests.  *" << std::endl;
  std::cout << "**************************************************************************" << std::endl;
  std::cout << std::endl;

  return EXIT_SUCCESS;
}
