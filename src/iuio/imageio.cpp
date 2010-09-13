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

/* ****************************************************************************

   imread

 */

//-----------------------------------------------------------------------------
iu::ImageCpu_8u_C1* imread_8u_C1(const std::string& filename)
{
  cv::Mat mat = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
  IuSize sz(mat.cols, mat.rows);

  iu::ImageCpu_8u_C1* im = new iu::ImageCpu_8u_C1(sz);
  cv::Mat im_mat(sz.height, sz.width, CV_8UC1, im->data(), im->pitch());

  IU_ASSERT( mat.size() == im_mat.size() && mat.channels() == im_mat.channels() );
  mat.copyTo(im_mat);
  return im;
}

//-----------------------------------------------------------------------------
iu::ImageCpu_8u_C3* imread_8u_C3(const std::string& filename)
{
  cv::Mat mat = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
  IuSize sz(mat.cols, mat.rows);

  iu::ImageCpu_8u_C3* im = new iu::ImageCpu_8u_C3(sz);
  cv::Mat im_mat(sz.height, sz.width, CV_8UC3, im->data(), im->pitch());

  IU_ASSERT( mat.size() == im_mat.size() );
  cv::cvtColor(mat, im_mat, CV_BGR2RGB);
  return im;
}

//-----------------------------------------------------------------------------
iu::ImageCpu_8u_C4* imread_8u_C4(const std::string& filename)
{
  cv::Mat mat = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
  IuSize sz(mat.cols, mat.rows);

  iu::ImageCpu_8u_C4* im = new iu::ImageCpu_8u_C4(sz);
  cv::Mat im_mat(sz.height, sz.width, CV_8UC4, im->data(), im->pitch());

  IU_ASSERT( mat.size() == im_mat.size() );
  cv::cvtColor(mat, im_mat, CV_BGR2RGBA);
  return im;
}

//-----------------------------------------------------------------------------
iu::ImageCpu_32f_C1* imread_32f_C1(const std::string& filename)
{
  cv::Mat mat = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
  IuSize sz(mat.cols, mat.rows);

  iu::ImageCpu_32f_C1* im = new iu::ImageCpu_32f_C1(sz);
  cv::Mat im_mat(sz.height, sz.width, CV_32FC1, im->data(), im->pitch());

  IU_ASSERT( mat.size() == im_mat.size() && mat.channels() == im_mat.channels() );
  mat.convertTo(im_mat, im_mat.type(), 1.0f/255.0f, 0);
  return im;
}

//-----------------------------------------------------------------------------
iu::ImageCpu_32f_C3* imread_32f_C3(const std::string& filename)
{
  cv::Mat mat = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
  IuSize sz(mat.cols, mat.rows);

  cv::Mat mat_32f_C3(mat.rows, mat.cols, CV_32FC3);
  mat.convertTo(mat_32f_C3, mat_32f_C3.type(), 1.0f/255.0f, 0);

  iu::ImageCpu_32f_C3* im = new iu::ImageCpu_32f_C3(sz);
  cv::Mat im_mat(sz.height, sz.width, CV_32FC3, im->data(), im->pitch());

  IU_ASSERT( mat_32f_C3.size() == im_mat.size() );
  cv::cvtColor(mat_32f_C3, im_mat, CV_BGR2RGB);
  return im;
}

//-----------------------------------------------------------------------------
iu::ImageCpu_32f_C4* imread_32f_C4(const std::string& filename)
{
  cv::Mat mat = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
  IuSize sz(mat.cols, mat.rows);

  cv::Mat mat_32f_C3(mat.rows, mat.cols, CV_32FC3);
  mat.convertTo(mat_32f_C3, mat_32f_C3.type(), 1.0f/255.0f, 0);

  iu::ImageCpu_32f_C4* im = new iu::ImageCpu_32f_C4(sz);
  cv::Mat im_mat(sz.height, sz.width, CV_32FC4, im->data(), im->pitch());

  IU_ASSERT( mat_32f_C3.size() == im_mat.size() );
  cv::cvtColor(mat_32f_C3, im_mat, CV_BGR2RGBA);
  return im;
}

//-----------------------------------------------------------------------------
iu::ImageNpp_8u_C1* imread_cu8u_C1(const std::string& filename)
{
  iu::ImageCpu_8u_C1* im = imread_8u_C1(filename);
  iu::ImageNpp_8u_C1* cu_im = new iu::ImageNpp_8u_C1(im->size());
  iuprivate::copy(im, cu_im);
  return cu_im;
}

//-----------------------------------------------------------------------------
iu::ImageNpp_8u_C4* imread_cu8u_C4(const std::string& filename)
{
  iu::ImageCpu_8u_C4* im = imread_8u_C4(filename);
  iu::ImageNpp_8u_C4* cu_im = new iu::ImageNpp_8u_C4(im->size());
  iuprivate::copy(im, cu_im);
  return cu_im;
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
iu::ImageNpp_32f_C4* imread_cu32f_C4(const std::string& filename)
{
  iu::ImageCpu_32f_C4* im = imread_32f_C4(filename);
  iu::ImageNpp_32f_C4* cu_im = new iu::ImageNpp_32f_C4(im->size());
  iuprivate::copy(im, cu_im);
  return cu_im;
}

/* ****************************************************************************

   imsave

 */

//-----------------------------------------------------------------------------
bool imsave(iu::ImageCpu_8u_C1* image, const std::string& filename)
{
  IuSize sz = image->size();
  cv::Mat mat_8u(sz.height, sz.width, CV_8UC1, image->data(), image->pitch());
  return cv::imwrite(filename, mat_8u);
}

//-----------------------------------------------------------------------------
bool imsave(iu::ImageCpu_8u_C3* image, const std::string& filename)
{
  IuSize sz = image->size();
  cv::Mat mat_8u(sz.height, sz.width, CV_8UC3, image->data(), image->pitch());
  cv::Mat bgr(sz.height, sz.width, CV_8UC3);
  cv::cvtColor(mat_8u, bgr, CV_RGB2BGR);
  return cv::imwrite(filename, bgr);
}

//-----------------------------------------------------------------------------
bool imsave(iu::ImageCpu_8u_C4* image, const std::string& filename)
{
  IuSize sz = image->size();
  cv::Mat mat_8u(sz.height, sz.width, CV_8UC4, image->data(), image->pitch());
  cv::Mat bgr(sz.height, sz.width, CV_8UC3);
  cv::cvtColor(mat_8u, bgr, CV_RGBA2BGR);
  return cv::imwrite(filename, bgr);
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
bool imsave(iu::ImageCpu_32f_C3* image, const std::string& filename)
{
  IuSize sz = image->size();
  cv::Mat mat_8u(sz.height, sz.width, CV_8UC3);
  cv::Mat mat_32f(sz.height, sz.width, CV_32FC3, image->data(), image->pitch());
  mat_32f.convertTo(mat_8u, mat_8u.type(), 255, 0);
  cv::Mat bgr(sz.height, sz.width, CV_8UC3);
  cv::cvtColor(mat_8u, bgr, CV_RGBA2BGR);
  return cv::imwrite(filename, bgr);
}

//-----------------------------------------------------------------------------
bool imsave(iu::ImageCpu_32f_C4* image, const std::string& filename)
{
  IuSize sz = image->size();
  cv::Mat mat_8u(sz.height, sz.width, CV_8UC4);
  cv::Mat mat_32f(sz.height, sz.width, CV_32FC4, image->data(), image->pitch());
  mat_32f.convertTo(mat_8u, mat_8u.type(), 255, 0);
  cv::Mat bgr(sz.height, sz.width, CV_8UC3);
  cv::cvtColor(mat_8u, bgr, CV_RGBA2BGR);
  return cv::imwrite(filename, bgr);
}

//-----------------------------------------------------------------------------
bool imsave(iu::ImageNpp_8u_C1* image, const std::string& filename)
{
  iu::ImageCpu_8u_C1 cpu_image(image->size());
  iuprivate::copy(image, &cpu_image);
  return iuprivate::imsave(&cpu_image, filename);
}

//-----------------------------------------------------------------------------
bool imsave(iu::ImageNpp_8u_C4* image, const std::string& filename)
{
  iu::ImageCpu_8u_C4 cpu_image(image->size());
  iuprivate::copy(image, &cpu_image);
  return iuprivate::imsave(&cpu_image, filename);
}

//-----------------------------------------------------------------------------
bool imsave(iu::ImageNpp_32f_C1* image, const std::string& filename)
{
  iu::ImageCpu_32f_C1 cpu_image(image->size());
  iuprivate::copy(image, &cpu_image);
  return iuprivate::imsave(&cpu_image, filename);
}

//-----------------------------------------------------------------------------
bool imsave(iu::ImageNpp_32f_C4* image, const std::string& filename)
{
  iu::ImageCpu_32f_C4 cpu_image(image->size());
  iuprivate::copy(image, &cpu_image);
  return iuprivate::imsave(&cpu_image, filename);
}


/* ****************************************************************************

   imshow

 */

//-----------------------------------------------------------------------------
void imshow(iu::ImageCpu_8u_C1* image, const std::string& winname)
{
  IuSize sz = image->size();
  cv::Mat mat_8u(sz.height, sz.width, CV_8UC1, image->data(), image->pitch());
  cv::imshow(winname, mat_8u);
}

//-----------------------------------------------------------------------------
void imshow(iu::ImageCpu_8u_C3* image, const std::string& winname)
{
  IuSize sz = image->size();
  cv::Mat mat_8u(sz.height, sz.width, CV_8UC3, image->data(), image->pitch());
  cv::Mat bgr(sz.height, sz.width, CV_8UC3);
  cv::cvtColor(mat_8u, bgr, CV_RGB2BGR);
  cv::imshow(winname, bgr);
}

//-----------------------------------------------------------------------------
void imshow(iu::ImageCpu_8u_C4* image, const std::string& winname)
{
  IuSize sz = image->size();
  cv::Mat mat_8u(sz.height, sz.width, CV_8UC4, image->data(), image->pitch());
  cv::Mat bgr(sz.height, sz.width, CV_8UC3);
  cv::cvtColor(mat_8u, bgr, CV_RGBA2BGR);
  cv::imshow(winname, bgr);
}

//-----------------------------------------------------------------------------
void imshow(iu::ImageCpu_32f_C1* image, const std::string& winname)
{
  IuSize sz = image->size();
  cv::Mat mat_32f(sz.height, sz.width, CV_32FC1, image->data(), image->pitch());
  cv::imshow(winname, mat_32f);
}

//-----------------------------------------------------------------------------
void imshow(iu::ImageCpu_32f_C3* image, const std::string& winname)
{
  IuSize sz = image->size();
  cv::Mat mat_32f(sz.height, sz.width, CV_32FC3, image->data(), image->pitch());
  cv::Mat bgr;
  cv::cvtColor(mat_32f, bgr, CV_RGB2BGR);
  cv::imshow(winname, bgr);
}

//-----------------------------------------------------------------------------
void imshow(iu::ImageCpu_32f_C4* image, const std::string& winname)
{
  IuSize sz = image->size();
  cv::Mat mat_32f(sz.height, sz.width, CV_32FC4, image->data(), image->pitch());
  cv::Mat bgr;
  cv::cvtColor(mat_32f, bgr, CV_RGBA2BGR);
  cv::imshow(winname, bgr);
}

//-----------------------------------------------------------------------------
void imshow(iu::ImageNpp_8u_C1* image, const std::string& winname)
{
  iu::ImageCpu_8u_C1 cpu_image(image->size());
  iuprivate::copy(image, &cpu_image);
  iuprivate::imshow(&cpu_image, winname);
}

//-----------------------------------------------------------------------------
void imshow(iu::ImageNpp_8u_C4* image, const std::string& winname)
{
  iu::ImageCpu_8u_C4 cpu_image(image->size());
  iuprivate::copy(image, &cpu_image);
  iuprivate::imshow(&cpu_image, winname);
}

//-----------------------------------------------------------------------------
void imshow(iu::ImageNpp_32f_C1* image, const std::string& winname)
{
  iu::ImageCpu_32f_C1 cpu_image(image->size());
  iuprivate::copy(image, &cpu_image);
  iuprivate::imshow(&cpu_image, winname);
}

//-----------------------------------------------------------------------------
void imshow(iu::ImageNpp_32f_C4* image, const std::string& winname)
{
  iu::ImageCpu_32f_C4 cpu_image(image->size());
  iuprivate::copy(image, &cpu_image);
  iuprivate::imshow(&cpu_image, winname);
}


} // namespace iuprivate
