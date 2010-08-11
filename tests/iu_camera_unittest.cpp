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

#include <QApplication>
#include <QTime>
#include <QImage>

//#define PERFORMANCE_TEST

#ifdef PERFORMANCE_TEST
#include <cv.h>
#include <highgui.h>
#endif

#include <iucore.h>
#include <iuio.h>
#include <iugui.h>

#include <cv.h>
#include <highgui.h>

using namespace iu;

int main(int argc, char** argv)
{
#ifdef Q_WS_X11
  bool use_gui = getenv("DISPLAY") != 0;
#else
  bool use_gui= true;
#endif
  assert(use_gui == true);

  QApplication app(argc, argv, use_gui);

//  if(argc < 2)
//  {
//    std::cout << "You have to provide at least a filename for reading an image." << std::endl;
//    exit(EXIT_FAILURE);
//  }

//  const std::string filename = argv[1];



  cv::VideoCapture cap(0); // open the default camera
  if(!cap.isOpened()) // check if we succeeded
    return -1;
  double bla = cap.get(CV_CAP_PROP_FORMAT);
  std::cout << "CV_CAP_PROP_FORMAT=" << bla << "; ";
  bla = cap.get(CV_CAP_PROP_MODE);
  std::cout << "CV_CAP_PROP_MODE=" << bla << "; " << std::endl;
//  cap.set(CV_CAP_ANY, 0);
  //cap.set(CV_CAP_, 0);

  cv::Mat edges;
  cv::namedWindow("edges",1);
  cv::namedWindow("camera",1);
  for(;;)
  {
    cv::Mat frame;
    cap >> frame; // get a new frame from camera
    //cv::cvtColor(frame, edges, CV_BayerBG2);
    cv::GaussianBlur(frame, edges, cv::Size(7,7), 1.5, 1.5);
    cv::Canny(edges, edges, 0, 30, 3);
    cv::imshow("edges", edges);
    cv::imshow("camera", frame);
    if(cv::waitKey(30) >= 0) break;
  }
  // the camera will be deinitialized automatically in VideoCapture destructor

  /*
  iu::ImageCpu_32f_C1* im = iu::imread_32f_C1(filename);
  iu::ImageNpp_32f_C1* cuim = iu::imread_cu32f_C1(filename);
  iu::ImageNpp_32f_C1 d_cp_im(im->size());
  iu::copy(im, &d_cp_im);
  iu::NppGLWidget cpu_widget(0, &d_cp_im, false, false);
  cpu_widget.setWindowTitle("cpu (copied to gpu for display)");
  cpu_widget.show();
  iu::NppGLWidget widget(0, cuim, false, false);
  widget.setWindowTitle("gpu");
  widget.show();
*/
  std::cout << std::endl;
  std::cout << "**************************************************************************" << std::endl;
  std::cout << "*  Everything seem to be ok. -- All assertions passed.                   *" << std::endl;
  std::cout << "*  Look at the images and close the windows to derminate the unittests.  *" << std::endl;
  std::cout << "**************************************************************************" << std::endl;
  std::cout << std::endl;

//  int retval = app.exec();

  // CLEANUP

  //return retval;
  return EXIT_SUCCESS;



}
