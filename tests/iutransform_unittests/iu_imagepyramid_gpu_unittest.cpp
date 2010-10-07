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
 * Description : Unit tests for gpu imagepyramids
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

// system includes
#include <iostream>
#include <highgui.h>
#include <cuda_runtime.h>
#include <cutil_math.h>
#include <iu/iudefs.h>
#include <iu/iucore.h>
#include <iu/iucutil.h>
#include <iu/iuio.h>
#include <iu/iutransform.h>

int main(int argc, char** argv)
{
  std::cout << "Starting iu_imagepyramid_gpu_unittest ..." << std::endl;

  if(argc < 2)
  {
    std::cout << "You have to provide at least a filename for reading an image." << std::endl;
    exit(EXIT_FAILURE);
  }
  const std::string filename = argv[1];

  std::cout << "reading image " << filename << std::endl;
  iu::ImageGpu_32f_C1* image = iu::imread_cu32f_C1(filename);

  iu::imshow(image, "input");


  // reduce to half image size
  iu::ImageGpu_32f_C1 reduce1_linear(image->width()/2.0f, image->height()/2.0f);
  iu::reduce(image, &reduce1_linear, IU_INTERPOLATE_LINEAR, true, false);
  iu::imshow(&reduce1_linear, "reduce1_linear");

  iu::ImageGpu_32f_C1 reduce1_cubic(image->width()/2.0f, image->height()/2.0f);
  iu::reduce(image, &reduce1_cubic, IU_INTERPOLATE_CUBIC, false, false);
  iu::imshow(&reduce1_cubic, "reduce1_cubic");

  iu::ImageGpu_32f_C1 reduce1_cubic_filtered(image->width()*0.8f, image->height()*0.8f);
  iu::reduce(image, &reduce1_cubic_filtered, IU_INTERPOLATE_CUBIC, false, true);
  iu::imshow(&reduce1_cubic_filtered, "reduce1_cubic_filtered");

  cv::waitKey();

  std::cout << std::endl;
  std::cout << "**************************************************************************" << std::endl;
  std::cout << "*  Everything seem to be ok. -- All assertions passed.                   *" << std::endl;
  std::cout << "*  Look at the images and close the windows to derminate the unittests.  *" << std::endl;
  std::cout << "**************************************************************************" << std::endl;
  std::cout << std::endl;

  return EXIT_SUCCESS;
}
