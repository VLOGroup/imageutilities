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
 * Module      : IO;
 * Class       : none
 * Language    : C++
 * Description : Implementation of image I/O functions
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#include <cv.h>
#include <highgui.h>
#include <iucore/copy.h>
#include "imageio.h"


namespace iuprivate {

//-----------------------------------------------------------------------------
iu::ImageCpu_32f_C1* imread_32f_C1(const std::string& filename)
{
  cv::Mat mat = cv::imread(filename, 0);
  IuSize sz(mat.cols, mat.rows);

  iu::ImageCpu_32f_C1* im = new iu::ImageCpu_32f_C1(sz);
  cv::Mat im_mat(sz.height, sz.width, CV_32FC1, im->data(), im->pitch());

  IU_ASSERT( mat.size() == im_mat.size() && mat.channels() == im_mat.channels() );
  mat.convertTo(im_mat, im_mat.type(), 1.0f/255.0f, 0);
  return im;
}

//-----------------------------------------------------------------------------
iu::ImageNpp_32f_C1* imread_cu32f_C1(const std::string& filename)
{
  iu::ImageCpu_32f_C1* im = imread_32f_C1(filename);
  iu::ImageNpp_32f_C1* cu_im = new iu::ImageNpp_32f_C1(im->size());
  iuprivate::copy(im, cu_im);
  return cu_im;
}

//-----------------------------------------------------------------------------
bool imsave(iu::ImageCpu_32f_C1* image, const std::string& filename)
{
  IuSize sz = image->size();
  cv::Mat mat_8u(sz.height, sz.width, CV_8UC1);
  cv::Mat mat_32f(sz.height, sz.width, CV_32FC1, image->data(), image->pitch());
  mat_32f.convertTo(mat_8u, mat_8u.type(), 255, 0);
  return cv::imwrite(filename, mat_8u);
}

//-----------------------------------------------------------------------------
bool imsave(iu::ImageNpp_32f_C1* image, const std::string& filename)
{
  iu::ImageCpu_32f_C1 cpu_image(image->size());
  iuprivate::copy(image, &cpu_image);
  return iuprivate::imsave(&cpu_image, filename);
}


} // namespace iuprivate
