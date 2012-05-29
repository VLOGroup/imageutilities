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

#include <iu/iucore.h>
#include <iu/iuio.h>
#include <iu/iufilter.h>

int main(int argc, char** argv)
{
  if(argc < 2)
  {
    std::cout << "You have to provide at least a filename for reading an image." << std::endl;
    exit(EXIT_FAILURE);
  }

  const std::string filename = argv[1];

  /* device images */
  iu::ImageGpu_32f_C1* in_C1 = iu::imread_cu32f_C1(filename);
  iu::imshow(in_C1 , "32f_C1 device image");

  iu::ImageGpu_32f_C4* in_C4= iu::imread_cu32f_C4(filename);
  iu::imshow(in_C4, "32f_C4 device image");

  iu::ImageGpu_32f_C1 out_C1(in_C1 ->size());

  // warm-up
  iu::filterBilateral(in_C1, &out_C1, in_C1->roi());

  double t=iu::getTime();
  for (int i=0; i<100; ++i)
    iu::filterBilateral(in_C1, &out_C1, in_C1->roi());
  t=iu::getTime()-t;
  t/=100;
  printf("filtering a %dx%d image took %fms\n", in_C1->width(), in_C1->height(), t);
  iu::imshow(&out_C1, "bilat filtered C1");

  iu::ImageGpu_32f_C4 out_C4(in_C4 ->size());
  iu::filterBilateral(in_C4, &out_C4, in_C4->roi());
  iu::imshow(&out_C4, "bilat filtered C4");


  iu::ImageGpu_32f_C1 out_C1C4(in_C1 ->size());
  iu::filterBilateral(in_C1, &out_C1C4, in_C1->roi(), in_C1, 1, 12.0f, 0.05f);
  iu::imshow(&out_C1C4, "bilat filtered C1 with C4 prior");


  std::cout << std::endl;
  std::cout << "**************************************************************************" << std::endl;
//  std::cout << "*  Everything seem to be ok. -- All assertions passed.                   *" << std::endl;
  std::cout << "*  Look at the images and press a key to close the windows and derminate the unittests.  *" << std::endl;
  std::cout << "**************************************************************************" << std::endl;
  std::cout << std::endl;

  // escape key terminates program
  int key=0;
  while(key!=27)
    key=cvWaitKey(10);

  // CLEANUP
  delete(in_C1 );
  delete(in_C4);

  return EXIT_SUCCESS;

}
