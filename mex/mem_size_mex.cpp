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
 * Project     : VMLibraries
 * Module      : ImageUtilities
 * Class       : non
 * Language    : C/MEX
 * Description : Function that returns the needed memory size of gpu memory.
 *
 * Author     :
 * EMail      :
 *
 */


// system includes
#include <iostream>
using namespace std;
#include <math.h>
#include <stdlib.h>
#include <list>
#include <mex.h>

// #ifdef _WIN32
// #  include <windows.h>
// #endif

#include <cuda.h>
#include <iucore.h>

void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) 
{
  if(nrhs > 2)
    mexErrMsgTxt("Too many input arguments");

  if(mxGetNumberOfElements(prhs[1]) != 1)
    mexErrMsgTxt("number of channels must be a scalar value");

  // Input Size
  const mwSize* dims = mxGetDimensions(prhs[0]);
  int dimension = dims[1];
  double* input_size = (double*)mxGetPr(prhs[0]);
  // Output Size
  plhs[0] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS,  mxREAL);
  double* output_size = (double*)mxGetPr(plhs[0]);

  // Number of channels
  int channels = mxGetScalar(prhs[1]);

  if (dimension == 1)
  {
    switch(channels)
    {
      case 1:
      {
        iu::LinearDeviceMemory_32f_C1 temp(input_size[0]);
        output_size[0] = temp.length();
      }
      break;
      case 2:
      {
        iu::LinearDeviceMemory_32f_C2 temp(input_size[0]);
        output_size[0] = temp.length();
      }
      break;
      case 3:
      {
        iu::LinearDeviceMemory_32f_C3 temp(input_size[0]);
        output_size[0] = temp.length();
      }
      break;
      case 4:
      {
        iu::LinearDeviceMemory_32f_C4 temp(input_size[0]);
        output_size[0] = temp.length();
      }
      break;
    }
  }
  else if (dimension == 2)
  {
    switch(channels)
    {
      case 1:
      {
        iu::ImageGpu_32f_C1 temp(input_size[1], input_size[0]);
        output_size[0] = temp.height();
        output_size[1] = temp.stride();
      }
      break;
      case 2:
      {
        iu::ImageGpu_32f_C2 temp(input_size[1], input_size[0]);
        output_size[0] = temp.height();
        output_size[1] = temp.stride();
      }
      break;
      case 3:
      {
        iu::ImageGpu_32f_C3 temp(input_size[1], input_size[0]);
        output_size[0] = temp.height();
        output_size[1] = temp.stride();
      }
      break;
      case 4:
      {
        iu::ImageGpu_32f_C4 temp(input_size[1], input_size[0]);
        output_size[0] = temp.height();
        output_size[1] = temp.stride();
      }
      break;
    }
  }
  else if (dimension == 3)
  {
    switch(channels)
    {
      case 1:
      {
        iu::VolumeGpu_32f_C1 temp(input_size[1], input_size[0], input_size[2]);
        output_size[0] = temp.slice_stride()/temp.stride();
        output_size[1] = temp.stride();
        output_size[2] = temp.depth();
      }
      break;
      case 2:
      {
        iu::VolumeGpu_32f_C2 temp(input_size[1], input_size[0], input_size[2]);
        output_size[0] = temp.slice_stride()/temp.stride();
        output_size[1] = temp.stride();
        output_size[2] = temp.depth();
      }
      break;
      case 3:
      {
        printf("3-channel volume not supported!\n");
        return;
      }
      break;
      case 4:
      {
        iu::VolumeGpu_32f_C4 temp(input_size[1], input_size[0], input_size[2]);
        output_size[0] = temp.slice_stride()/temp.stride();
        output_size[1] = temp.stride();
        output_size[2] = temp.depth();
      }
      break;
    }
  }
  else
  {
    printf("%d dimensions are not supported!\n", dimension);
    return;
  }

}         
