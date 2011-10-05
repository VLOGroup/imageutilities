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
 * Description : Unit tests for VideoCapture class
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

// system includes
#include <iostream>
#include <cuda_runtime.h>

#include <iudefs.h>
#include <iucore.h>
#include <iuio.h>
#include <highgui.h>

int main(int argc, char** argv)
{
  iu::VideoCapture* cap = 0;

  int device = -1;
  if(argc < 2)
  {
    std::cout << "using camera" << std::endl;
    cap = new iu::VideoCapture(device);
  }
  else
  {
    std::string filename = argv[1];
    std::cout << "reading video " << filename << std::endl;
    cap = new iu::VideoCapture(filename);
  }

  //    printf("cap image size = %d/%d\n", cap->size().width, cap->size().height);
  iu::ImageCpu_8u_C1 im_8u_C1(cap->size());
  iu::ImageCpu_32f_C1 im_32f_C1(cap->size());
  iu::ImageGpu_8u_C1 im_gpu_8u_C1(cap->size());
  iu::ImageGpu_32f_C1 im_gpu_32f_C1(cap->size());

  for(;;)
  {

    if(cap->grab())
    {
//      cap->retrieve(&im_8u_C1);
//      iu::imshow(&im_8u_C1, "cpu_8u_C1 image");

//      cap->retrieve(&im_32f_C1);
//      iu::imshow(&im_32f_C1, "cpu_32f_C1 image");

//      cap->retrieve(&im_gpu_8u_C1);
//      iu::imshow(&im_gpu_8u_C1, "gpu_8u_C1 image");

      cap->retrieve(&im_gpu_32f_C1);
      iu::imshow(&im_gpu_32f_C1, "gpu_32f_C1 image");
    }
    //printf("wait\n");
    //cv::waitKey();
    //printf("main next round...\n");
    if(cv::waitKey(30) >= 0) break;
  }

  std::cout << std::endl;
  std::cout << "**************************************************************************" << std::endl;
//  std::cout << "*  Everything seem to be ok. -- All assertions passed.                   *" << std::endl;
  std::cout << "*  Look at the images and press a key to close the windows and derminate the unittests.  *" << std::endl;
  std::cout << "**************************************************************************" << std::endl;
  std::cout << std::endl;

  cv::waitKey();

  // CLEANUP
  delete(cap);

  return EXIT_SUCCESS;

}
