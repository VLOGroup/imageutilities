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
 * Description : Unit tests for ImageReader class for 16bit
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

// system includes
#include <iostream>
#include <cuda_runtime.h>
#include <cutil.h>

//#include <cv.h>
#include <highgui.h>

#include <iucore.h>
#include <iuio.h>
#include <iugui.h>

using namespace iu;

int main(int argc, char** argv)
{
  if(argc < 2)
  {
    std::cout << "You have to provide at least a filename for reading an image." << std::endl;
    exit(EXIT_FAILURE);
  }

  const std::string filename = argv[1];

//  iu::ImageNpp_32f_C1 image(256, 256);
//  printf("image width=%d, height=%d, stride=%d, pitch=%d\n", image.width(), image.height(),
//         image.stride(), image.pitch());

//  cv::Mat mat = cv::imread(filename, CV_LOAD_IMAGE_ANYDEPTH);
//  switch(mat.depth())
//  {
//  case CV_8U:
//    printf("CV_8U\n");
//    break;
//  case CV_16U:
//    printf("CV_16U\n");
//    break;

//  case CV_8S:
//    printf("CV_8S\n");
//    break;
//  case CV_16S:
//    printf("CV_16S\n");
//    break;

//  default:
//    printf("Bit-Depth not in switch list\n");
//    break;
//  }
//  printf("bit depth of read image = %d\n", mat.depth());
//  cv::imshow("OpenCV Mat image", mat);

//  // convert to 32bit and than to 8-bit. should be fine to display :-)
//  cv::Mat mat_32f_C1(mat.rows, mat.cols, CV_32FC1);
//  mat.convertTo(mat_32f_C1, mat_32f_C1.type(), 1.0f/65535.0f, 0);
//  cv::Mat mat_8u_C1(mat.rows, mat.cols, CV_8UC1);
//  mat_32f_C1.convertTo(mat_8u_C1, mat_8u_C1.type(), 255, 0);
//  cv::imshow("bla", mat_8u_C1);
//  cv::imwrite("out_16bit_to_32bit_to_8bit.png", mat_8u_C1);


  iu::ImageCpu_32f_C1 *image = iu::imread_16u_C1(filename);
  iu::imshow(image, "[host] 16bit -> 32bit image");

  iu::ImageNpp_32f_C1 *image_32f_C1 = iu::imread_cu16u_C1(filename);
  iu::imshow(image_32f_C1, "[device] 16bit -> 32bit image");


//  cv::Mat mat_8u(cvim.rows, cvim.cols, CV_8UC1);
//  cvim.convertTo(mat_8u, mat_8u.type(), 255.0/655335.0, 0.0);
//  cv::imshow("8-bit", mat_8u);

//  /* host images */
//  iu::ImageCpu_8u_C1* im_8u_C1 = iu::imread_8u_C1(filename);
//  iu::imshow(im_8u_C1, "8u_C1 host image");
//  iu::imsave(im_8u_C1, "out_8u_C1_host.png");

//  iu::ImageCpu_8u_C3* im_8u_C3 = iu::imread_8u_C3(filename);
//  iu::imshow(im_8u_C3, "8u_C3 host image");
//  iu::imsave(im_8u_C3, "out_8u_C3_host.png");

//  iu::ImageCpu_8u_C4* im_8u_C4 = iu::imread_8u_C4(filename);
//  iu::imshow(im_8u_C4, "8u_C4 host image");
//  iu::imsave(im_8u_C4, "out_8u_C4_host.png");

//  iu::ImageCpu_32f_C1* im_32f_C1 = iu::imread_32f_C1(filename);
//  iu::imshow(im_32f_C1, "32f_C1 host image");
//  iu::imsave(im_32f_C1, "out_32f_C1_host.png");

//  iu::ImageCpu_32f_C3* im_32f_C3 = iu::imread_32f_C3(filename);
//  iu::imshow(im_32f_C3, "32f_C3 host image");
//  iu::imsave(im_32f_C3, "out_32f_C3_host.png");

//  iu::ImageCpu_32f_C4* im_32f_C4 = iu::imread_32f_C4(filename);
//  iu::imshow(im_32f_C4, "32f_C4 host image");
//  iu::imsave(im_32f_C4, "out_32f_C4_host.png");


//  /* device images */
//  iu::ImageNpp_8u_C1* cuim_8u_C1 = iu::imread_cu8u_C1(filename);
//  iu::imshow(cuim_8u_C1, "8u_C1 device image");
//  iu::imsave(cuim_8u_C1, "out_8u_C1_device.png");

//  iu::ImageNpp_8u_C4* cuim_8u_C4 = iu::imread_cu8u_C4(filename);
//  iu::imshow(cuim_8u_C4, "8u_C4 device image");
//  iu::imsave(cuim_8u_C4, "out_8u_C4_device.png");

//  iu::ImageNpp_32f_C1* cuim_32f_C1 = iu::imread_cu32f_C1(filename);
//  iu::imshow(cuim_32f_C1, "32f_C1 device image");
//  iu::imsave(cuim_32f_C1, "out_32f_C1_device.png");

//  iu::ImageNpp_32f_C4* cuim_32f_C4 = iu::imread_cu32f_C4(filename);
//  iu::imshow(cuim_32f_C4, "32f_C4 device image");
//  iu::imsave(cuim_32f_C4, "out_32f_C4_device.png");



  std::cout << std::endl;
  std::cout << "**************************************************************************" << std::endl;
//  std::cout << "*  Everything seem to be ok. -- All assertions passed.                   *" << std::endl;
  std::cout << "*  Look at the images and press a key to close the windows and derminate the unittests.  *" << std::endl;
  std::cout << "**************************************************************************" << std::endl;
  std::cout << std::endl;

  cv::waitKey();

  // CLEANUP
//  delete(im_8u_C1);
//  delete(cuim_8u_C1);
//  delete(im_32f_C1);
//  delete(cuim_32f_C1);

  return EXIT_SUCCESS;

}
