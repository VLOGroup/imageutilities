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
//#include <QTime>
//#include <QImage>

#include <iucore.h>
#include <iuio.h>
#include <iugui.h>
#include <iufilter.h>

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

  const std::string filename = argv[1];

  iu::ImageNpp_32f_C1* im = iu::imread_cu32f_C1(filename);
  iu::NppGLWidget widget(0, im, false, false);
  widget.setWindowTitle("unfiltered");
  widget.show();

  // MEDIAN FILTER
  iu::ImageNpp_32f_C1 median_filtered(im->size());
  iu::filterMedian3x3(im, &median_filtered, im->roi());
  iu::NppGLWidget median_filtered_widget(0, &median_filtered, false, false);
  median_filtered_widget.setWindowTitle("Median filtered");
  median_filtered_widget.show();

  // GAUSS FILTER
  iu::ImageNpp_32f_C1 gauss_filtered(im->size());
  iu::filterGauss(im, &gauss_filtered, im->roi(), 10.0f);
  iu::NppGLWidget gauss_filtered_widget(0, &gauss_filtered, false, false);
  gauss_filtered_widget.setWindowTitle("Gauss filtered");
  gauss_filtered_widget.show();

  // ROF FILTER
  iu::ImageNpp_32f_C1 rof_filtered(im->size());
  iu::filterRof(im, &rof_filtered, rof_filtered.roi(), 1.0f, 100);
  iu::NppGLWidget rof_filtered_widget(0, &rof_filtered, false, false);
  rof_filtered_widget.setWindowTitle("ROF filtered");
  rof_filtered_widget.show();


  // GAUSSIAN STRUCTURE-TEXTURE DECOMPOSITION
  iu::ImageNpp_32f_C1 decomposed_gauss(im->size());
  iu::decomposeStructureTextureGauss(im, &decomposed_gauss, decomposed_gauss.roi());
  iu::NppGLWidget decomposed_gauss_widget(0, &decomposed_gauss, false, false);
  decomposed_gauss_widget.setWindowTitle("Gaussian structure-texture decomposition");
  decomposed_gauss_widget.show();

  // ROF STRUCTURE-TEXTURE DECOMPOSITION
  iu::ImageNpp_32f_C1 decomposed_rof(im->size());
  iu::decomposeStructureTextureRof(im, &decomposed_rof, decomposed_rof.roi(), 0.8f, 1.0f, 100);
  iu::NppGLWidget decomposed_rof_widget(0, &decomposed_rof, false, false);
  decomposed_rof_widget.setWindowTitle("ROF structure-texture decomposition");
  decomposed_rof_widget.show();


  std::cout << std::endl;
  std::cout << "**************************************************************************" << std::endl;
  std::cout << "*  Everything seem to be ok. -- All assertions passed.                   *" << std::endl;
  std::cout << "*  Look at the images and close the windows to derminate the unittests.  *" << std::endl;
  std::cout << "**************************************************************************" << std::endl;
  std::cout << std::endl;

  int retval = app.exec();
  // CLEANUP

  delete(im);
  return retval;

}
