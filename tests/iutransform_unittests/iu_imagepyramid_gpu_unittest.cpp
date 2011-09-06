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
#include <iu/iucontainers.h>
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
  std::cout << "image pixel_type = " << image->pixelType() << std::endl;

  iu::imshow(image, "original input");

  unsigned int levels = 100;
  iu::ImagePyramid pyramid(levels, image->size(), 0.8f, 32);

  std::cout << "#levels changed to " << levels << " to fit the pyramid" << std::endl;

  std::stringstream stream;
  std::string winname;

  std::cout << "--pyramid pixel type = " << pyramid.pixelType() << std::endl;
  pyramid.setImage(image);
  std::cout << "--pyramid pixel type = " << pyramid.pixelType() << std::endl;

//  iu::copy(image, reinterpret_cast<iu::ImageGpu_32f_C1*>(pyramid.image(0)));
//  iu::imshow(reinterpret_cast<iu::ImageGpu_32f_C1*>(pyramid.image(0)), "level 0");

  double time = iu::getTime();
  std::cout << "num_levels=" << pyramid.numLevels() << std::endl;

//  std::cout << "level " << 0 << ": size=" << pyramid.size(0).width << "/" << pyramid.size(0).height
//            << "; rate=" << pyramid.scaleFactor(0) << std::endl;


  std::cout << "image pixel type = " << image->pixelType() << std::endl;
  std::cout << "pyramid pixel type = " << pyramid.pixelType() << std::endl;

  // print all the created level sizes:
  for(unsigned int i=0; i<pyramid.numLevels(); ++i)
  {
    std::cout << "level " << i << ":" << std::endl;

    std::cout << "  size=" << pyramid.size(i).width << "/" << pyramid.size(i).height
              << "; rate=" << pyramid.scaleFactor(i) << std::endl;

    stream.clear();
    stream << "level " << i;
    winname = stream.str();
    iu::imshow(pyramid.imageGpu_32f_C1(i), winname);
  }

  time = iu::getTime() - time;
  std::cout << "time for building imagepyramid = " << time << "ms" << std::endl;

  cv::waitKey();

  std::cout << std::endl;
  std::cout << "**************************************************************************" << std::endl;
  std::cout << "*  Everything seem to be ok. -- All assertions passed.                   *" << std::endl;
  std::cout << "*  Look at the images and close the windows to derminate the unittests.  *" << std::endl;
  std::cout << "**************************************************************************" << std::endl;
  std::cout << std::endl;

  return EXIT_SUCCESS;
}
