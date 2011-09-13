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
 * Description : Unit tests for linear buffers in the image utilities library
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

// system includes
#include <iostream>
#include <cuda_runtime.h>
#include <iucore.h>
#include <iu/iucutil.h>
#include <float.h>
#include <iu/iumath.h>

extern IuStatus cuMinMax(iu::LinearDeviceMemory_32f_C1 *src1,  iu::LinearDeviceMemory_32f_C1 *src2,
                         iu::LinearDeviceMemory_32f_C1 *minim, iu::LinearDeviceMemory_32f_C1 *maxim);


int main(int argc, char** argv)
{
  unsigned int length = 1e6;

  // create linar hostbuffer
  iu::LinearHostMemory_32f_C1* h_in1_32f_C1 = new iu::LinearHostMemory_32f_C1(length);
  iu::LinearHostMemory_32f_C1* h_in2_32f_C1 = new iu::LinearHostMemory_32f_C1(length);

  iu::LinearHostMemory_32f_C1* h_min_32f_C1 = new iu::LinearHostMemory_32f_C1(length);
  iu::LinearHostMemory_32f_C1* h_max_32f_C1 = new iu::LinearHostMemory_32f_C1(length);

  // fill input and calculate ground truth
  float minim_gt1 = FLT_MAX;
  float maxim_gt1 = FLT_MIN;

  float minim_gt2 = FLT_MAX;
  float maxim_gt2 = FLT_MIN;

  for (unsigned int i=0; i<length; i++)
  {
    float v1 = float(rand())/float(rand());
    minim_gt1 = std::min(v1, minim_gt1);
    maxim_gt1 = std::max(v1, maxim_gt1);

    float v2 = float(rand())/float(rand());
    minim_gt2 = std::min(v2, minim_gt2);
    maxim_gt2 = std::max(v2, maxim_gt2);

    *h_in1_32f_C1->data(i) = v1;
    *h_in2_32f_C1->data(i) = v2;

    *h_min_32f_C1->data(i) = std::min(*h_in1_32f_C1->data(i), *h_in2_32f_C1->data(i));
    *h_max_32f_C1->data(i) = std::max(*h_in1_32f_C1->data(i), *h_in2_32f_C1->data(i));
  }

  std::cout << "CPU: Image1: min=" << minim_gt1 << ", max=" << maxim_gt1 <<
               "   Image2: min=" << minim_gt2 << ", max=" << maxim_gt2 << std::endl;

  // copy data to device
  iu::LinearDeviceMemory_32f_C1* d_in1_32f_C1 = new iu::LinearDeviceMemory_32f_C1(length);
  iu::LinearDeviceMemory_32f_C1* d_in2_32f_C1 = new iu::LinearDeviceMemory_32f_C1(length);

  iu::LinearDeviceMemory_32f_C1* d_min_32f_C1 = new iu::LinearDeviceMemory_32f_C1(length);
  iu::LinearDeviceMemory_32f_C1* d_max_32f_C1 = new iu::LinearDeviceMemory_32f_C1(length);

  iu::copy(h_in1_32f_C1, d_in1_32f_C1);
  iu::copy(h_in2_32f_C1, d_in2_32f_C1);

  // Calculate min / max on GPU
//  float minim_gt1_gpu = FLT_MAX;
//  float maxim_gt1_gpu = FLT_MIN;
//  iu::minMax(d_in1_32f_C1, minim_gt1_gpu, maxim_gt1_gpu);

//  float minim_gt2_gpu = FLT_MAX;
//  float maxim_gt2_gpu = FLT_MIN;
//  iu::minMax(d_in2_32f_C1, minim_gt2_gpu, maxim_gt2_gpu);

//  std::cout << "GPU: Image1: min=" << minim_gt1_gpu << ", max=" << maxim_gt1_gpu <<
//               "   Image2: min=" << minim_gt2_gpu << ", max=" << maxim_gt2_gpu << std::endl;

  // Pointwise min/max
  cuMinMax(d_in1_32f_C1, d_in2_32f_C1, d_min_32f_C1, d_max_32f_C1);

  // copy back to host
  iu::LinearHostMemory_32f_C1* h_min_GPU_32f_C1 = new iu::LinearHostMemory_32f_C1(length);
  iu::LinearHostMemory_32f_C1* h_max_GPU_32f_C1 = new iu::LinearHostMemory_32f_C1(length);

  iu::copy(d_min_32f_C1, h_min_GPU_32f_C1);
  iu::copy(d_max_32f_C1, h_max_GPU_32f_C1);

  // compare to ground truth
  for (unsigned int i=0; i<length; i++)
  {
    if ((*h_min_32f_C1->data(i) != *h_min_GPU_32f_C1->data(i)) || (*h_max_32f_C1->data(i) != *h_max_GPU_32f_C1->data(i)))
    {
      std::cout << "ERROR: i=" << i << " Input=" <<  *h_in1_32f_C1->data(i) << ", " << *h_in2_32f_C1->data(i) << "     Min/Max: CPU="
                << *h_min_32f_C1->data(i) << ", " << *h_max_32f_C1->data(i) << " GPU=" << *h_min_GPU_32f_C1->data(i) << ", "
                << *h_max_GPU_32f_C1->data(i) << std::endl;
    }
  }


//  std::cout << std::endl;
//  std::cout << "**************************************************************************" << std::endl;
//  std::cout << "*  Everything seem to be ok. -- All assertions passed.                   *" << std::endl;
//  std::cout << "**************************************************************************" << std::endl;
//  std::cout << std::endl;

}

