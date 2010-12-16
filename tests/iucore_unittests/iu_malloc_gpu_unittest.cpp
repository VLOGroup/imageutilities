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
 * Description : Unit tests for gpu images and volumes testing memory layout
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

// system includes
#include <iostream>
#include <cuda_runtime.h>
//#include <cutil_math.h>
#include <iucore.h>
#include <iu/iucutil.h>

int main(int argc, char** argv)
{
  std::cout << "Starting iu_malloc_gpu_unittest ..." << std::endl;

  IuSize max_size(26, 60, 30);

//  for(unsigned int z = 1; z<max_size.depth; ++z)
//    for(unsigned int y = 1; y<max_size.height; ++y)

  unsigned int y = max_size.height;
  unsigned int z = max_size.depth;
  for(unsigned int x = 1; x<max_size.width; ++x)
      {
        IuSize sz(x,y,z);
        iu::ImageGpu_32f_C1 im_32f_C1(sz);
        std::cout << "img w=" << sz.width << " h=" << sz.height << " d=" << sz.depth
                  << ": pitch=" << im_32f_C1.pitch()
                  << ": stride=" << im_32f_C1.stride() << std::endl;

        iu::VolumeGpu_32f_C1 vol_32f_C1(sz);
        std::cout << "vol w=" << sz.width << " h=" << sz.height << " d=" << sz.depth
                  << ": pitch=" << vol_32f_C1.pitch()
                  << " stride=" << vol_32f_C1.stride()
                  << " slice_pitch=" << vol_32f_C1.slice_pitch()
                  << " slice_stride=" << vol_32f_C1.slice_stride() << std::endl;



      }


  std::cout << std::endl;
  std::cout << "**************************************************************************" << std::endl;
  std::cout << "*  Everything seem to be ok. -- All assertions passed.                   *" << std::endl;
  std::cout << "*  Look at the images and close the windows to derminate the unittests.  *" << std::endl;
  std::cout << "**************************************************************************" << std::endl;
  std::cout << std::endl;

  return EXIT_SUCCESS;
}
