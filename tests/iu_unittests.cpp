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
 * Description : Unit tests for image utilities library
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
#include <iugui.h>
#include <iumath.h>

//#include "fl_debug.h"

using namespace iu;

/* compares two float values.
   taken from [1]
   [1] http://www.cygnus-software.com/papers/comparingfloats/Comparing%20floating%20point%20numbers.htm
*/
bool almostEquals(float A, float B, int maxUlps = 1)
{
  // Make sure maxUlps is non-negative and small enough that the
  // default NAN won't compare as equal to anything.
  assert(maxUlps > 0 && maxUlps < 4 * 1024 * 1024);
  int aInt = *(int*)&A;
  // Make aInt lexicographically ordered as a twos-complement int
  if (aInt < 0)
      aInt = 0x80000000 - aInt;
  // Make bInt lexicographically ordered as a twos-complement int
  int bInt = *(int*)&B;
  if (bInt < 0)
      bInt = 0x80000000 - bInt;
  int intDiff = abs(aInt - bInt);
  if (intDiff <= maxUlps)
      return true;
  return false;
}


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

  IuSize sz(311,221);
  IuSize sz2(311,221);

  /* *************************************************************************
    Npp IMAGES; Multichannel set operations
  * *************************************************************************/
  {
    iu::ImageNpp_32f_C1 im_cu32f_C1(sz);
    iu::ImageNpp_32f_C2 im_cu32f_C2(sz);
    iu::ImageNpp_32f_C4 im_cu32f_C4(sz);

    Npp32f set_value_C1 = 1.1f;
    Npp32f set_value_C2[2] = {2.2f,2.2f};
    Npp32f set_value_C4[4] = {4.4f,4.4f,4.4f,4.4f};

    iu::setValue(set_value_C1, &im_cu32f_C1, im_cu32f_C1.roi());
    iu::setValue(set_value_C2, &im_cu32f_C2, im_cu32f_C2.roi());
    iu::setValue(set_value_C4, &im_cu32f_C4, im_cu32f_C4.roi());

    float min_C1,max_C1, min_C2[2],max_C2[2], min_C3[3],max_C3[3], min_C4[4],max_C4[4];
    iu::minMax(&im_cu32f_C1, im_cu32f_C1.roi(), min_C1, max_C1);
    iu::minMax(&im_cu32f_C2, im_cu32f_C2.roi(), min_C2, max_C2);
    iu::minMax(&im_cu32f_C4, im_cu32f_C4.roi(), min_C4, max_C4);


    std::cout << "Npp set test/32f/C1: min=" << min_C1 << " max=" << max_C1 << std::endl

        << "Npp set test/32f/C2: min=" << min_C2[0] << "/" << min_C2[1]
        << " max=" << max_C2[0] << "/" << max_C2[1] << std::endl

        << "Npp set test/32f/C4: min=" << min_C4[0] << "/" << min_C4[1] << "/" << min_C4[2] << "/" << min_C4[3]
        << " max=" << max_C4[0] << "/" << max_C4[1] << "/" << max_C4[2] << "/" << min_C4[3]  << std::endl
        << std::endl;

    assert(almostEquals(min_C1, set_value_C1, 1));
    assert(almostEquals(min_C2[0], set_value_C2[0], 1));
    assert(almostEquals(min_C2[1], set_value_C2[1], 1));
    assert(almostEquals(min_C4[0], set_value_C4[0], 1));
    assert(almostEquals(min_C4[1], set_value_C4[1], 1));
    assert(almostEquals(min_C4[2], set_value_C4[2], 1));
    assert(almostEquals(min_C4[3], set_value_C4[3], 1));

    assert(almostEquals(max_C1, set_value_C1, 1));
    assert(almostEquals(max_C2[0], set_value_C2[0], 1));
    assert(almostEquals(max_C2[1], set_value_C2[1], 1));
    assert(almostEquals(max_C4[0], set_value_C4[0], 1));
    assert(almostEquals(max_C4[1], set_value_C4[1], 1));
    assert(almostEquals(max_C4[2], set_value_C4[2], 1));
    assert(almostEquals(max_C4[3], set_value_C4[3], 1));
  }


  // create Ipp images

  // 8u
  iu::ImageIpp_8u_C1 *im_8u_C1 = new iu::ImageIpp_8u_C1(sz);
  iu::ImageIpp_8u_C3 *im_8u_C3 = new iu::ImageIpp_8u_C3(sz);
  iu::ImageIpp_8u_C4 *im_8u_C4 = new iu::ImageIpp_8u_C4(sz);

  iu::setValue(0, im_8u_C1, im_8u_C1->roi());

  // 32f
  iu::ImageIpp_32f_C1 *im_32f_C1 = new iu::ImageIpp_32f_C1(sz2);
  iu::ImageIpp_32f_C3 *im_32f_C3 = new iu::ImageIpp_32f_C3(sz2);
  iu::ImageIpp_32f_C4 *im_32f_C4 = new iu::ImageIpp_32f_C4(sz2);

  // fill ipp images with test data

  // 8u
  IppStatus status;
  status = ippiImageJaehne_8u_C1R(im_8u_C1->data(), im_8u_C1->pitch(), im_8u_C1->ippSize());
  status = ippiImageJaehne_8u_C3R(im_8u_C3->data(), im_8u_C3->pitch(), im_8u_C3->ippSize());
  status = ippiImageJaehne_8u_C4R(im_8u_C4->data(), im_8u_C4->pitch(), im_8u_C4->ippSize());

  // 32f
  status = ippiImageJaehne_32f_C1R(im_32f_C1->data(), im_32f_C1->pitch(), im_32f_C1->ippSize());
  status = ippiImageJaehne_32f_C3R(im_32f_C3->data(), im_32f_C3->pitch(), im_32f_C3->ippSize());
  status = ippiImageJaehne_32f_C4R(im_32f_C4->data(), im_32f_C4->pitch(), im_32f_C4->ippSize());


  // create Npp images

  // 8u
  iu::ImageNpp_8u_C1 *im_cu8u_C1 = new iu::ImageNpp_8u_C1(sz);
  iu::ImageNpp_8u_C3 *im_cu8u_C3 = new iu::ImageNpp_8u_C3(sz);
  iu::ImageNpp_8u_C4 *im_cu8u_C4 = new iu::ImageNpp_8u_C4(sz);

  // 32f
  iu::ImageNpp_32f_C1 *im_cu32f_C1 = new iu::ImageNpp_32f_C1(sz2);
  iu::ImageNpp_32f_C3 *im_cu32f_C3 = new iu::ImageNpp_32f_C3(sz2);
  iu::ImageNpp_32f_C4 *im_cu32f_C4 = new iu::ImageNpp_32f_C4(sz2);


  // copy data

  // 8u
  iu::copy(im_8u_C1, im_8u_C1->roi(), im_cu8u_C1, im_cu8u_C1->roi());
  iu::copy(im_8u_C3, im_8u_C3->roi(), im_cu8u_C3, im_cu8u_C3->roi());
  iu::copy(im_8u_C4, im_8u_C4->roi(), im_cu8u_C4, im_cu8u_C4->roi());

  // 32f
  iu::copy(im_32f_C1, im_32f_C1->roi(), im_cu32f_C1, im_cu32f_C1->roi());
  iu::copy(im_32f_C3, im_32f_C3->roi(), im_cu32f_C3, im_cu32f_C3->roi());
  iu::copy(im_32f_C4, im_32f_C4->roi(), im_cu32f_C4, im_cu32f_C4->roi());


  // display npp images
  // FIXMEEEE why the hell is an instance working and an object isn't???!
#if 1
  iu::NppGLWidget* widget_32f_C1 = new iu::NppGLWidget(0, im_cu32f_C1, true, true );
  widget_32f_C1->show();
  widget_32f_C1->setWindowTitle("NPP: 32f_C1");

  iu::NppGLWidget* widget_32f_C4 = new iu::NppGLWidget(0, im_cu32f_C4, true, true );
  widget_32f_C4->show();
  widget_32f_C4->setWindowTitle("NPP: 32f_C4");

  QObject::connect(widget_32f_C1, SIGNAL(mousePressed(int,int)), widget_32f_C4, SLOT(close()));

#else
  iu::NppGLWidget widget_32f_C1(0, im_cu32f_C1, true, true );
  widget_32f_C1.show();
  widget_32f_C1.setWindowTitle("NPP: 32f_C1");

  iu::NppGLWidget widget_32f_C4(0, im_cu32f_C4, true, true );
  widget_32f_C4.show();
  widget_32f_C4.setWindowTitle("NPP: 32f_C4");
#endif

  // search min/max value

  // ipp; 8u
  Ipp8u min_8u_C1, max_8u_C1;
  Ipp8u min_8u_C3[3], max_8u_C3[3];
  Ipp8u min_8u_C4[4], max_8u_C4[4];
  Ipp64f sum_8u_C1, sum_8u_C3[3], sum_8u_C4[4];

  iu::minMax(im_8u_C1, im_8u_C1->roi(), min_8u_C1, max_8u_C1);
  iu::minMax(im_8u_C3, im_8u_C3->roi(), min_8u_C3, max_8u_C3);
  iu::minMax(im_8u_C4, im_8u_C4->roi(), min_8u_C4, max_8u_C4);

  iu::summation(im_8u_C1,  im_8u_C1->roi(), sum_8u_C1);
  iu::summation(im_8u_C3,  im_8u_C3->roi(), sum_8u_C3);
  iu::summation(im_8u_C4,  im_8u_C4->roi(), sum_8u_C4);

  std::cout << "Ipp/8u/C1: min=" << (int)min_8u_C1 << " max=" << (int)max_8u_C1 << std::endl
      << "           sum=" << (int)sum_8u_C1 << std::endl

      << "Ipp/8u/C3: min=" << (int)min_8u_C3[0] << "/" << (int)min_8u_C3[1] << "/" << (int)min_8u_C3[2]
      << " max=" << (int)max_8u_C3[0] << "/" << (int)max_8u_C3[1] << "/" << (int)max_8u_C3[2]  << std::endl
      << "           sum=" << (int)sum_8u_C3[0] << "/" << (int)sum_8u_C3[1] << "/" << (int)sum_8u_C3[2] << std::endl

      << "Ipp/8u/C4: min=" << (int)min_8u_C4[0] << "/" << (int)min_8u_C4[1] << "/" << (int)min_8u_C4[2] << "/" << (int)min_8u_C4[3]
      << " max=" << (int)max_8u_C4[0] << "/" << (int)max_8u_C4[1] << "/" << (int)max_8u_C4[2] << "/" << (int)min_8u_C4[3]  << std::endl
      << "           sum=" << (int)sum_8u_C4[0] << "/" << (int)sum_8u_C4[1] << "/" << (int)sum_8u_C4[2] << "/" << (int)sum_8u_C4[3] << std::endl
      << std::endl;


  // ipp; 32f;
  Ipp32f min_32f_C1, max_32f_C1;
  Ipp32f min_32f_C3[3], max_32f_C3[3];
  Ipp32f min_32f_C4[4], max_32f_C4[4];
  Ipp64f sum_32f_C1, sum_32f_C3[3], sum_32f_C4[4];

  iu::minMax(im_32f_C1, im_32f_C1->roi(), min_32f_C1, max_32f_C1);
  iu::minMax(im_32f_C3, im_32f_C3->roi(), min_32f_C3, max_32f_C3);
  iu::minMax(im_32f_C4, im_32f_C4->roi(), min_32f_C4, max_32f_C4);

  iu::summation(im_32f_C1,  im_32f_C1->roi(), sum_32f_C1);
  iu::summation(im_32f_C3,  im_32f_C3->roi(), sum_32f_C3);
  iu::summation(im_32f_C4,  im_32f_C4->roi(), sum_32f_C4);

  std::cout << "Ipp/32f/C1: min=" << min_32f_C1 << " max=" << max_32f_C1 << std::endl
      << "           sum=" << (int)sum_32f_C1 << std::endl
      << "Ipp/32f/C3: min=" << min_32f_C3[0] << "/" << min_32f_C3[1] << "/" << min_32f_C3[2]
      << " max=" << max_32f_C3[0] << "/" << max_32f_C3[1] << "/" << max_32f_C3[2]  << std::endl
      << "           sum=" << (int)sum_32f_C3[0] << "/" << (int)sum_32f_C3[1] << "/" << (int)sum_32f_C3[2] << std::endl
      << "Ipp/32f/C4: min=" << min_32f_C4[0] << "/" << min_32f_C4[1] << "/" << min_32f_C4[2] << "/" << min_32f_C4[3]
      << " max=" << max_32f_C4[0] << "/" << max_32f_C4[1] << "/" << max_32f_C4[2] << "/" << min_32f_C4[3]  << std::endl
      << "           sum=" << (int)sum_32f_C4[0] << "/" << (int)sum_32f_C4[1] << "/" << (int)sum_32f_C4[2] << "/" << (int)sum_32f_C4[3] << std::endl
      << std::endl;

  // Npp; 8u
  Npp8u min_cu8u_C1, max_cu8u_C1;
  Npp8u min_cu8u_C4[4] = {0,0,0,0}, max_cu8u_C4[4] = {0,0,0,0};
  Npp32u sum_cu32u_C1;

  iu::minMax(im_cu8u_C1, im_cu8u_C1->roi(), min_cu8u_C1, max_cu8u_C1);
  iu::minMax(im_cu8u_C4, im_cu8u_C4->roi(), min_cu8u_C4, max_cu8u_C4);
  iu::summation(im_cu8u_C1, im_cu8u_C1->roi(), sum_cu32u_C1);

  std::cout
      << "Npp/8u/C1: min=" << (int)min_cu8u_C1 << " max=" << (int)max_cu8u_C1 << std::endl
      << "           sum=" << sum_cu32u_C1 << std::endl

      << "Npp/8u/C4: min=" << (int)min_cu8u_C4[0] << "/" << (int)min_cu8u_C4[1] << "/" << (int)min_cu8u_C4[2] << "/" << (int)min_cu8u_C4[3]
      << " max=" << (int)max_cu8u_C4[0] << "/" << (int)max_cu8u_C4[1] << "/" << (int)max_cu8u_C4[2] << "/" << (int)min_cu8u_C4[3]  << std::endl

      << std::endl;

  // npp; 32f;
  Npp32f min_cu32f_C1, max_cu32f_C1;
  Npp32f min_cu32f_C4[4], max_cu32f_C4[4];
  Npp32f sum_cu32f_C1;

  iu::minMax(im_cu32f_C1, im_cu32f_C1->roi(), min_cu32f_C1, max_cu32f_C1);
  iu::minMax(im_cu32f_C4, im_cu32f_C4->roi(), min_cu32f_C4, max_cu32f_C4);
  iu::summation(im_cu32f_C1, im_cu32f_C1->roi(), sum_cu32f_C1);
  std::cout
      << "Npp/32f/C1: min=" << min_cu32f_C1 << " max=" << max_cu32f_C1 << std::endl
      << "            sum=" << sum_cu32f_C1 << std::endl
      << "Npp/32f/C4: min=" << min_cu32f_C4[0] << "/" << min_cu32f_C4[1] << "/" << min_cu32f_C4[2] << "/" << min_cu32f_C4[3]
      << " max=" << max_cu32f_C4[0] << "/" << max_cu32f_C4[1] << "/" << max_cu32f_C4[2] << "/" << min_cu32f_C4[3]  << std::endl
      << std::endl;

  // assert - compare: host vs device results (min/max)
  // 8u_C1
  assert(almostEquals(min_8u_C1, min_cu8u_C1, 1));
  assert(almostEquals(max_8u_C1, max_cu8u_C1, 1));
  assert(almostEquals(sum_8u_C1, sum_cu32u_C1, 1));
  // 8u_C4
  assert(almostEquals(min_8u_C4[0], min_cu8u_C4[0], 1));
  assert(almostEquals(min_8u_C4[1], min_cu8u_C4[1], 1));
  assert(almostEquals(min_8u_C4[2], min_cu8u_C4[2], 1));
  assert(almostEquals(min_8u_C4[3], min_cu8u_C4[3], 1));
  assert(almostEquals(max_8u_C4[0], max_cu8u_C4[0], 1));
  assert(almostEquals(max_8u_C4[1], max_cu8u_C4[1], 1));
  assert(almostEquals(max_8u_C4[2], max_cu8u_C4[2], 1));
  assert(almostEquals(max_8u_C4[3], max_cu8u_C4[3], 1));
  // 32f_C1
  assert(almostEquals(min_32f_C1, min_cu32f_C1, 1));
  assert(almostEquals(max_32f_C1, max_cu32f_C1, 1));
  assert(almostEquals(sum_32f_C1, sum_cu32f_C1, 2));
  // 32f_C4
  assert(almostEquals(min_32f_C4[0], min_cu32f_C4[0], 1));
  assert(almostEquals(min_32f_C4[1], min_cu32f_C4[1], 1));
  assert(almostEquals(min_32f_C4[2], min_cu32f_C4[2], 1));
  assert(almostEquals(min_32f_C4[3], min_cu32f_C4[3], 1));
  assert(almostEquals(max_32f_C4[0], max_cu32f_C4[0], 1));
  assert(almostEquals(max_32f_C4[1], max_cu32f_C4[1], 1));
  assert(almostEquals(max_32f_C4[2], max_cu32f_C4[2], 1));
  assert(almostEquals(max_32f_C4[3], max_cu32f_C4[3], 1));


  std::cout << std::endl;
  std::cout << "**************************************************************************" << std::endl;
  std::cout << "*  Everything seem to be ok. -- All assertions passed.                   *" << std::endl;
  std::cout << "*  Look at the images and close the windows to derminate the unittests.  *" << std::endl;
  std::cout << "**************************************************************************" << std::endl;
  std::cout << std::endl;

  int retval = app.exec();
  // CLEANUP
#if 1
  delete(widget_32f_C1);
  delete(widget_32f_C4);
#endif

  delete(im_8u_C1);
  delete(im_8u_C3);
  delete(im_8u_C4);

  delete(im_32f_C1);
  delete(im_32f_C3);
  delete(im_32f_C4);

  delete(im_cu8u_C1);
  delete(im_cu8u_C3);
  delete(im_cu8u_C4);

  delete(im_cu32f_C1);
  delete(im_cu32f_C3);
  delete(im_cu32f_C4);

  return retval;
}
