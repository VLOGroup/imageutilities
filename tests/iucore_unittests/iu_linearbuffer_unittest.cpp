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


int main(int argc, char** argv)
{
  unsigned int length = 1e7;

  // create linar hostbuffer
  iu::LinearHostMemory_8u_C1* h_8u_C1 = new iu::LinearHostMemory_8u_C1(length);
  iu::LinearHostMemory_32f_C1* h_32f_C1 = new iu::LinearHostMemory_32f_C1(length);

  // create linear devicebuffer
  iu::LinearDeviceMemory_8u_C1* d_8u_C1 = new iu::LinearDeviceMemory_8u_C1(length);
  iu::LinearDeviceMemory_32f_C1* d_32f_C1 = new iu::LinearDeviceMemory_32f_C1(length);

  // temporary variables to check the values
  iu::LinearHostMemory_8u_C1 check_8u_C1(d_8u_C1->length());
  iu::LinearHostMemory_32f_C1 check_32f_C1(d_32f_C1->length());

  // dummy copy tests:
  // host->host
  iu::copy(h_8u_C1, &check_8u_C1);
  iu::copy(h_32f_C1, &check_32f_C1);
  // host->device
  iu::copy(h_8u_C1, d_8u_C1);
  iu::copy(h_32f_C1, d_32f_C1);
  // device->host
  iu::copy(d_8u_C1, h_8u_C1);
  iu::copy(d_32f_C1, h_32f_C1);


  // values to be set
  unsigned char val_8u = 1;
  float val_32f = 1.0f;

  // set host values
  iu::setValue(val_8u, h_8u_C1);
  iu::setValue(val_32f, h_32f_C1);

  // set device values
  iu::setValue(val_8u, d_8u_C1);
  iu::setValue(val_32f, d_32f_C1);


  ////////////////////////////////////////////////////////////////////////////
  /* check if device set values works
   */

  // copy device variables back to the cpu and check its values
  iu::copy(d_8u_C1, &check_8u_C1);
  iu::copy(d_32f_C1, &check_32f_C1);

  // check if the values are ok
  for (unsigned int i = 0; i<check_8u_C1.length(); ++i)
  {
    assert(*h_8u_C1->data(i) == val_8u);
    assert(*h_32f_C1->data(i) == val_32f);
    assert(*check_8u_C1.data(i) == val_8u);
    assert(*check_32f_C1.data(i) == val_32f);
  }

  ////////////////////////////////////////////////////////////////////////////
  /* check if host to device and device to host copy works
   */

  // values to be set
  val_8u = 2;
  val_32f = 2.0f;

  // set host values
  iu::setValue(val_8u, h_8u_C1);
  iu::setValue(val_32f, h_32f_C1);

  // copy host to device
  iu::copy(h_8u_C1, d_8u_C1);
  iu::copy(h_32f_C1, d_32f_C1);

  // copy device variables back to the cpu and check its values
  iu::copy(d_8u_C1, &check_8u_C1);
  iu::copy(d_32f_C1, &check_32f_C1);

  // check if the values are ok
  for (unsigned int i = 0; i<check_8u_C1.length(); ++i)
  {
    assert(h_8u_C1->data()[i] == val_8u);
    assert(h_32f_C1->data()[i] == val_32f);
    assert(check_8u_C1.data()[i] == val_8u);
    assert(check_32f_C1.data()[i] == val_32f);
  }

  ////////////////////////////////////////////////////////////////////////////
  /* Further test
   */


//  // create buffer with given host buffer (fb_ ... from buffer)
//  length = 10;
//  Npp8u hb_8u[length];
//  for (unsigned int i = 0; i < length; ++i)
//    hb_8u[i] = i;
//  Npp8u* hb_8u_p = &hb_8u[0];
//  iu::LinearDeviceMemory_8u fb_d_8u(hb_8u, length);
//  iu::LinearHostMemory_8u fb_h_8u(hb_8u, length);
//  iu::LinearHostMemory_8u fb_h_8u_tmp1(hb_8u, length, true);
//  iu::LinearHostMemory_8u fb_h_8u_tmp2(fb_h_8u_tmp1);

  //  CLEANUP
  delete(h_8u_C1);
  delete(h_32f_C1);

  delete(d_8u_C1);
  delete(d_32f_C1);

  std::cout << std::endl;
  std::cout << "**************************************************************************" << std::endl;
  std::cout << "*  Everything seem to be ok. -- All assertions passed.                   *" << std::endl;
  std::cout << "*  Look at the images and close the windows to derminate the unittests.  *" << std::endl;
  std::cout << "**************************************************************************" << std::endl;
  std::cout << std::endl;

}
