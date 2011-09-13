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
 * Module      : Test
 * Class       : none
 * Language    : C++
 * Description :
 *
 * Author     :
 * EMail      :
 *
 */

#ifndef IU_MINMAX_UNITTEST_CU
#define IU_MINMAX_UNITTEST_CU

#include <iucore.h>
#include <iu/iucutil.h>

/******************************************************************************
    CUDA KERNELS
*******************************************************************************/

// kernel; find min/max; 32f_C1
__global__ void cuMinMaxKernel_32f_C1(float* src1, float* src2, float* minim, float* maxim, int length)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;

  // find minima of columns
  if (x<length)
  {
    minim[x] = IUMIN(src1[x], src2[x]);
    maxim[x] = IUMAX(src1[x], src2[x]);
  }
}


/******************************************************************************
  CUDA INTERFACES
*******************************************************************************/

// wrapper: find min/max; 32f_C1
IuStatus cuMinMax(iu::LinearDeviceMemory_32f_C1 *src1,  iu::LinearDeviceMemory_32f_C1 *src2,
                  iu::LinearDeviceMemory_32f_C1 *minim, iu::LinearDeviceMemory_32f_C1 *maxim)
{
  dim3 dimBlock(512, 1);
  dim3 dimGrid(iu::divUp(src1->length(), dimBlock.x), 1);

  cuMinMaxKernel_32f_C1<<< dimGrid,dimBlock >>>(src1->data(), src2->data(),
                                                minim->data(), maxim->data(),
                                                minim->length());

  return iu::checkCudaErrorState();
}


#endif // IU_MINMAX_UNITTEST_CU

