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
#include <iuiopgm.h>

using namespace iu;

int main(int argc, char** argv)
{
  if(argc < 2)
  {
    std::cout << "You have to provide at least a filename for reading an image." << std::endl;
    exit(EXIT_FAILURE);
  }

  const std::string filename = argv[1];


  iu::ImageCpu_32f_C1 *image = iu::imread_16u32f_C1(filename, 4096);
  iu::imshow(image, "[host] 16bit -> 32bit image");

  iu::ImageGpu_32f_C1 *image_32f_C1 = iu::imread_cu16u32f_C1(filename, 4096);
  iu::imshow(image_32f_C1, "[device] 16bit -> 32bit image");


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
