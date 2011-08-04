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
 * Description : Unit tests for exception handling
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

// system includes
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <iucore.h>
#include <iu/iucutil.h>

int main(int argc, char** argv)
{
  std::cout << "Starting iu_image_gpu_unittest ..." << std::endl;

  // test image size
  IuSize sz(1000,1000);

  std::vector<iu::ImageGpu_32f_C1*> image_vector;

  try
  {
    while (true)
    {
      iu::ImageGpu_32f_C1* image = new iu::ImageGpu_32f_C1(sz);
      image_vector.push_back(image);
    }
  }
  catch (std::exception& e)
  {
    std::cout << "added " << image_vector.size() << " images until memory exception" << std::endl;
    while (!image_vector.empty())
    {
      iu::ImageGpu_32f_C1* image = image_vector.back();
      image_vector.pop_back();
      delete(image);
    }

    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
