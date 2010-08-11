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
#include <QApplication>
#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <cuda_runtime.h>
#include <iucore.h>

using namespace iu;

int main(int argc, char** argv)
{
#ifdef Q_WS_X11
  bool useGUI = getenv("DISPLAY") != 0;
#else
  bool useGUI = true;
#endif
  QApplication app(argc, argv, useGUI);

//  if (useGUI) {
//    // start GUI version
//        ...
//     } else {
//        // start non-GUI version
//        ...
//     }

  unsigned int length = 1e7;

  // create linar hostbuffer
  iu::LinearHostMemory_8u* h_8u = new iu::LinearHostMemory_8u(length);
  iu::LinearHostMemory_32f* h_32f = new iu::LinearHostMemory_32f(length);

  // create linear devicebuffer
  iu::LinearDeviceMemory_8u* d_8u = new iu::LinearDeviceMemory_8u(length);
  iu::LinearDeviceMemory_32f* d_32f = new iu::LinearDeviceMemory_32f(length);

  // temporary variables to check the values
  iu::LinearHostMemory_8u check_8u(d_8u->length());
  iu::LinearHostMemory_32f check_32f(d_32f->length());

  // dummy copy tests:
  // host->host
  iu::copy(h_8u, &check_8u);
  iu::copy(h_32f, &check_32f);
  // host->device
  iu::copy(h_8u, d_8u);
  iu::copy(h_32f, d_32f);
  // device->host
  iu::copy(d_8u, h_8u);
  iu::copy(d_32f, h_32f);


  // values to be set
  Npp8u val_8u = 1;
  float val_32f = 1.0f;

  // set host values
  iu::setValue(val_8u, h_8u);
  iu::setValue(val_32f, h_32f);

  // set device values
  iu::setValue(val_8u, d_8u);
  iu::setValue(val_32f, d_32f);


  ////////////////////////////////////////////////////////////////////////////
  /* check if device set values works
   */

  // copy device variables back to the cpu and check its values
  iu::copy(d_8u, &check_8u);
  iu::copy(d_32f, &check_32f);

  // check if the values are ok
  for (unsigned int i = 0; i<check_8u.length(); ++i)
  {
    assert(h_8u->data()[i] == val_8u);
    assert(h_32f->data()[i] == val_32f);
    assert(check_8u.data()[i] == val_8u);
    assert(check_32f.data()[i] == val_32f);
  }

  ////////////////////////////////////////////////////////////////////////////
  /* check if host to device and device to host copy works
   */

  // values to be set
  val_8u = 2;
  val_32f = 2.0f;

  // set host values
  iu::setValue(val_8u, h_8u);
  iu::setValue(val_32f, h_32f);

  // copy host to device
  iu::copy(h_8u, d_8u);
  iu::copy(h_32f, d_32f);

  // copy device variables back to the cpu and check its values
  iu::copy(d_8u, &check_8u);
  iu::copy(d_32f, &check_32f);

  // check if the values are ok
  for (unsigned int i = 0; i<check_8u.length(); ++i)
  {
    assert(h_8u->data()[i] == val_8u);
    assert(h_32f->data()[i] == val_32f);
    assert(check_8u.data()[i] == val_8u);
    assert(check_32f.data()[i] == val_32f);
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
  delete(h_8u);
  delete(h_32f);

  delete(d_8u);
  delete(d_32f);

  std::cout << std::endl;
  std::cout << "**************************************************************************" << std::endl;
  std::cout << "*  Everything seem to be ok. -- All assertions passed.                   *" << std::endl;
  std::cout << "*  Look at the images and close the windows to derminate the unittests.  *" << std::endl;
  std::cout << "**************************************************************************" << std::endl;
  std::cout << std::endl;

}
