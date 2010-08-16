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

  if(argc < 2)
  {
    std::cout << "You have to provide at least a filename for reading an image." << std::endl;
    exit(EXIT_FAILURE);
  }

  const QString filename(argv[1]);

#ifdef PERFORMANCE_TEST
  QTime t;
  t.start();
  qDebug("Time measurements ... \n");
  for(int i = 0; i < 10; ++i)
  {
    ImageReader filereader(filename);
    QImage image = filereader.read();
  }
  qDebug("Time to read image (ImageReader): %d ms\n", t.elapsed()/10);

  // OpenCV comparison
  t.restart();
  for(int i = 0; i < 10; ++i)
  {
    QImage image;
    image.load(filename);
  }
  qDebug("Time to read image (QImage): %d ms\n", t.elapsed()/10);


  // OpenCV comparison
  t.restart();
  for(int i = 0; i < 10; ++i)
  {
    cv::Mat im1 = cv::imread(qPrintable(filename), 0);
  }
  qDebug("Time to read image (OpenCV): %d ms\n", t.elapsed()/10);

  // OpenCV comparison
  t.restart();
  for(int i = 0; i < 10; ++i)
  {
    IplImage* image = cvLoadImage(qPrintable(filename));
//    delete(image);
  }
  qDebug("Time to read image (OpenCV/Ipl): %d ms\n", t.elapsed()/10);
#endif

  ImageReader filereader(filename);
  QImage image = filereader.read();
  IuSize sz = filereader.iusize();
  std::cout << "image size=" << sz.width << "x" << sz.height << std::endl;

  QSize sz1 = filereader.size();
  std::cout << "image size=" << sz1.width() << "x" << sz1.height() << std::endl;

  IuSize sz2 (image.width(), image.height());
  std::cout << "image size=" << sz2.width << "x" << sz2.height << std::endl;

  if(image.isNull())
  {
    qDebug("failed: %s\n", filereader.errorString());
    exit(EXIT_FAILURE);
  }

  image.invertPixels();
  image.save("test_imagereader_invert.png");

  std::cout << std::endl;
  std::cout << "**************************************************************************" << std::endl;
  std::cout << "*  Everything seem to be ok. -- All assertions passed.                   *" << std::endl;
  std::cout << "*  Look at the images and close the windows to derminate the unittests.  *" << std::endl;
  std::cout << "**************************************************************************" << std::endl;
  std::cout << std::endl;

  return(EXIT_SUCCESS);
  int retval = app.exec();
  // CLEANUP


  return retval;

}
