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
 * Description : Unit tests for ImageReader class
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

// system includes
#include <iostream>
#include <cuda_runtime.h>


#include <cv.h>
#include <highgui.h>

#include <iucore.h>
#include <iuio.h>

int main(int argc, char** argv)
{
  if(argc < 2)
  {
    std::cout << "You have to provide at least a filename for reading an image." << std::endl;
    exit(EXIT_FAILURE);
  }



  const std::string filename = argv[1];

  cv::Mat cvim = cv::imread(filename, 0);
  cv::imshow("OpenCV grayscale image", cvim);
  cv::Mat cvim2;
  cv::resize(cvim, cvim2, cv::Size(128,128), 0, 0, cv::INTER_CUBIC);

  cv::Mat cvim_rgb = cv::imread(filename, 1);
  cv::imshow("OpenCV color image", cvim_rgb);

  cv::imwrite("out_cv_gray.png", cvim);
  cv::imwrite("out_cv_gray_128.png", cvim2);
  cv::imwrite("out_cv_color.png", cvim_rgb);

  /* host images */
  iu::ImageCpu_8u_C1* im_8u_C1 = iu::imread_8u_C1(filename);
  iu::imshow(im_8u_C1, "8u_C1 host image");
  iu::imsave(im_8u_C1, "out_8u_C1_host.png");

  iu::ImageCpu_8u_C3* im_8u_C3 = iu::imread_8u_C3(filename);
  iu::imshow(im_8u_C3, "8u_C3 host image");
  iu::imsave(im_8u_C3, "out_8u_C3_host.png");

  iu::ImageCpu_8u_C4* im_8u_C4 = iu::imread_8u_C4(filename);
  iu::imshow(im_8u_C4, "8u_C4 host image");
  iu::imsave(im_8u_C4, "out_8u_C4_host.png");

  iu::ImageCpu_32f_C1* im_32f_C1 = iu::imread_32f_C1(filename);
  iu::imshow(im_32f_C1, "32f_C1 host image");
  iu::imsave(im_32f_C1, "out_32f_C1_host.png");

  iu::ImageCpu_32f_C3* im_32f_C3 = iu::imread_32f_C3(filename);
  iu::imshow(im_32f_C3, "32f_C3 host image");
  iu::imsave(im_32f_C3, "out_32f_C3_host.png");

  iu::ImageCpu_32f_C4* im_32f_C4 = iu::imread_32f_C4(filename);
  iu::imshow(im_32f_C4, "32f_C4 host image");
  iu::imsave(im_32f_C4, "out_32f_C4_host.png");


  /* device images */
  iu::ImageGpu_8u_C1* cuim_8u_C1 = iu::imread_cu8u_C1(filename);
  iu::imshow(cuim_8u_C1, "8u_C1 device image");
  iu::imsave(cuim_8u_C1, "out_8u_C1_device.png");

  iu::ImageGpu_8u_C4* cuim_8u_C4 = iu::imread_cu8u_C4(filename);
  iu::imshow(cuim_8u_C4, "8u_C4 device image");
  iu::imsave(cuim_8u_C4, "out_8u_C4_device.png");

  iu::ImageGpu_32f_C1* cuim_32f_C1 = iu::imread_cu32f_C1(filename);
  iu::imshow(cuim_32f_C1, "32f_C1 device image");
  iu::imsave(cuim_32f_C1, "out_32f_C1_device.png");

  iu::ImageGpu_32f_C4* cuim_32f_C4 = iu::imread_cu32f_C4(filename);
  iu::imshow(cuim_32f_C4, "32f_C4 device image");
  iu::imsave(cuim_32f_C4, "out_32f_C4_device.png");



  std::cout << std::endl;
  std::cout << "**************************************************************************" << std::endl;
//  std::cout << "*  Everything seem to be ok. -- All assertions passed.                   *" << std::endl;
  std::cout << "*  Look at the images and press a key to close the windows and derminate the unittests.  *" << std::endl;
  std::cout << "**************************************************************************" << std::endl;
  std::cout << std::endl;

  cv::waitKey();

  // CLEANUP
  delete(im_8u_C1);
  delete(cuim_8u_C1);
  delete(im_32f_C1);
  delete(cuim_32f_C1);

  return EXIT_SUCCESS;

}
