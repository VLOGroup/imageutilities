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
 * Description : Unit tests for NppWidget class
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
#include <iugui.h>

using namespace iu;

int main(int argc, char** argv)
{
#ifdef Q_WS_X11
  bool use_gui = getenv("DISPLAY") != 0;
#else
  bool use_gui= true;
#endif
  assert(use_gui == true);

  QApplication app(argc, argv, use_gui);

  if(argc < 2)
  {
    std::cout << "You have to provide a filename." << std::endl;
    exit(EXIT_FAILURE);
  }

  // read testimage
  cv::Mat mat1_8u_C1 = cv::imread(argv[1], 0);
  cv::Mat mat1_bgr_8u_C3 = cv::imread(argv[1], 1);

  cv::Mat mat1_8u_C3(mat1_bgr_8u_C3.size(), CV_8UC3);
  cv::cvtColor(mat1_bgr_8u_C3, mat1_8u_C3, CV_BGR2RGB);

  Ipp8u* ipp1_8u_C1 = (Ipp8u*)mat1_8u_C1.data;
  Ipp8u* ipp1_8u_C3 = (Ipp8u*)mat1_8u_C3.data;

  // ipp/npp images
  IuSize sz(mat1_8u_C1.cols, mat1_8u_C1.rows);
  iu::ImageIpp_32f_C1 im1_32f_C1(sz);
  iu::ImageIpp_32f_C3 im1_32f_C3(sz);
  iu::ImageNpp_32f_C1 im1_cu32f_C1(sz);
  iu::ImageNpp_32f_C3 im1_cu32f_C3(sz);
  iu::ImageNpp_32f_C4 im1_cu32f_C4(sz);

  // convert 8u to 32f
  ippiScale_8u32f_C1R(ipp1_8u_C1, (int)mat1_8u_C1.step,
                      im1_32f_C1.data(), im1_32f_C1.pitch(), im1_32f_C1.ippSize(), 0.0f, 1.0f);
  ippiScale_8u32f_C3R(ipp1_8u_C3, (int)mat1_8u_C3.step,
                      im1_32f_C3.data(), im1_32f_C3.pitch(), im1_32f_C3.ippSize(), 0.0f, 1.0f);

  // copy host -> device
  iu::copy(&im1_32f_C1, im1_32f_C1.roi(), &im1_cu32f_C1, im1_cu32f_C1.roi());
  iu::copy(&im1_32f_C3, im1_32f_C3.roi(), &im1_cu32f_C3, im1_cu32f_C3.roi());
  iu::convert(&im1_cu32f_C3, im1_cu32f_C3.roi(), &im1_cu32f_C4, im1_cu32f_C4.roi());

  // display
  iu::NppGLWidget widget1_32f_C1(0, &im1_cu32f_C1, true, true);
  widget1_32f_C1.show();
  iu::NppGLWidget widget1_32f_C4(0, &im1_cu32f_C4, true, true);
  widget1_32f_C4.show();


  std::cout << std::endl;
  std::cout << "**************************************************************************" << std::endl;
  std::cout << "*  Everything seem to be ok. -- All assertions passed.                   *" << std::endl;
  std::cout << "*  Look at the images and close the windows to derminate the unittests.  *" << std::endl;
  std::cout << "**************************************************************************" << std::endl;
  std::cout << std::endl;


  int retval = app.exec();
  // CLEANUP


  return retval;

}
