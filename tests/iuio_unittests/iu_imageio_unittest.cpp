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

  iu::ImageCpu_32f_C1* im = iu::imread_32f_C1(filename);
  iu::ImageNpp_32f_C1* cuim = iu::imread_cu32f_C1(filename);
  iu::ImageNpp_32f_C1 d_cp_im(im->size());
  iu::copy(im, &d_cp_im);

  iu::imshow(im, "host image");
  iu::imshow(cuim, "device image");
  iu::imshow(&d_cp_im, "copied to device");

  std::cout << std::endl;
  std::cout << "**************************************************************************" << std::endl;
//  std::cout << "*  Everything seem to be ok. -- All assertions passed.                   *" << std::endl;
  std::cout << "*  Look at the images and press a key to close the windows and derminate the unittests.  *" << std::endl;
  std::cout << "**************************************************************************" << std::endl;
  std::cout << std::endl;

  cv::waitKey();

  // CLEANUP
  delete(im);
  delete(cuim);

  return EXIT_SUCCESS;

}
