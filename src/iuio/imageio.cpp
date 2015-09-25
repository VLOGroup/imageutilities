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

#include <stdio.h>
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
//   cv::Mat mat = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat mat = cv::imread(filename);
  if(mat.channels() > 2)
    cvtColor(mat, mat, CV_BGR2GRAY);
      
  IuSize sz(mat.cols, mat.rows);

  iu::ImageCpu_8u_C1* im = new iu::ImageCpu_8u_C1(sz);
  cv::Mat im_mat(sz.height, sz.width, CV_8UC1, im->data(), im->pitch());

  if ((mat.size()!=im_mat.size()) || (mat.channels()!=im_mat.channels()))
    throw IuException("mat sizes do not match", __FILE__, __FUNCTION__, __LINE__);
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

  if (mat.size()!=im_mat.size())
    throw IuException("mat sizes do not match", __FILE__, __FUNCTION__, __LINE__);

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

  if (mat.size()!=im_mat.size())
    throw IuException("mat sizes do not match", __FILE__, __FUNCTION__, __LINE__);

  cv::cvtColor(mat, im_mat, CV_BGR2RGBA);
  return im;
}

//-----------------------------------------------------------------------------
iu::ImageCpu_32f_C1* imread_32f_C1(const std::string& filename)
{
  cv::Mat mat = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
  if (!mat.data)
    throw IuException("Can't load file", filename.c_str(), __FUNCTION__, __LINE__);
  IuSize sz(mat.cols, mat.rows);

  iu::ImageCpu_32f_C1* im = new iu::ImageCpu_32f_C1(sz);
  cv::Mat im_mat(sz.height, sz.width, CV_32FC1, im->data(), im->pitch());

  if ((mat.size()!=im_mat.size()) || (mat.channels()!=im_mat.channels()))
    throw IuException("mat sizes do not match", __FILE__, __FUNCTION__, __LINE__);
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

  if (mat.size()!=im_mat.size())
    throw IuException("mat sizes do not match", __FILE__, __FUNCTION__, __LINE__);
  cv::cvtColor(mat_32f_C3, im_mat, CV_BGR2RGB);
  return im;
}

//-----------------------------------------------------------------------------
iu::ImageCpu_32f_C4* imread_32f_C4(const std::string& filename)
{
  cv::Mat mat = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
  if (!mat.data)
    throw IuException("Can't load file", filename.c_str(), __FUNCTION__, __LINE__);
  IuSize sz(mat.cols, mat.rows);

  cv::Mat mat_32f_C3(mat.rows, mat.cols, CV_32FC3);
  mat.convertTo(mat_32f_C3, mat_32f_C3.type(), 1.0f/255.0f, 0);

  iu::ImageCpu_32f_C4* im = new iu::ImageCpu_32f_C4(sz);
  cv::Mat im_mat(sz.height, sz.width, CV_32FC4, im->data(), im->pitch());

  if (mat.size()!=im_mat.size())
    throw IuException("mat sizes do not match", __FILE__, __FUNCTION__, __LINE__);
  cv::cvtColor(mat_32f_C3, im_mat, CV_BGR2RGBA);
  // FIXMEEEE the alpha layer is 0 !!!
  return im;
}

//-----------------------------------------------------------------------------
iu::ImageGpu_8u_C1* imread_cu8u_C1(const std::string& filename)
{
  iu::ImageCpu_8u_C1* im = imread_8u_C1(filename);
  iu::ImageGpu_8u_C1* cu_im = new iu::ImageGpu_8u_C1(im->size());
  iuprivate::copy(im, cu_im);
  delete im;
  return cu_im;
}

//-----------------------------------------------------------------------------
iu::ImageGpu_8u_C4* imread_cu8u_C4(const std::string& filename)
{
  iu::ImageCpu_8u_C4* im = imread_8u_C4(filename);
  iu::ImageGpu_8u_C4* cu_im = new iu::ImageGpu_8u_C4(im->size());
  iuprivate::copy(im, cu_im);
  delete im;
  return cu_im;
}

//-----------------------------------------------------------------------------
iu::ImageGpu_32f_C1* imread_cu32f_C1(const std::string& filename)
{
  iu::ImageCpu_32f_C1* im = imread_32f_C1(filename);
  iu::ImageGpu_32f_C1* cu_im = new iu::ImageGpu_32f_C1(im->size());
  iuprivate::copy(im, cu_im);
  delete im;
  return cu_im;
}

//-----------------------------------------------------------------------------
iu::ImageGpu_32f_C4* imread_cu32f_C4(const std::string& filename)
{
  iu::ImageCpu_32f_C4* im = imread_32f_C4(filename);
  iu::ImageGpu_32f_C4* cu_im = new iu::ImageGpu_32f_C4(im->size());
  iuprivate::copy(im, cu_im);
  delete im;
  return cu_im;
}

/* ****************************************************************************

   imsave

 */

//-----------------------------------------------------------------------------
bool imsave(iu::ImageCpu_8u_C1* image, const std::string& filename, const bool& normalize)
{
  IuSize sz = image->size();
  cv::Mat mat_imageCpu(sz.height, sz.width, CV_8UC1, image->data(), image->pitch());
  cv::Mat mat_8u = mat_imageCpu.clone();
  if(normalize)
    cv::normalize(mat_8u, mat_8u, 0, 255, cv::NORM_MINMAX);
  return cv::imwrite(filename, mat_8u);
}

//-----------------------------------------------------------------------------
bool imsave(iu::ImageCpu_8u_C3* image, const std::string& filename, const bool& normalize)
{
  // TODO do normalization as in 32f_C3
  IuSize sz = image->size();
  cv::Mat mat_imageCpu(sz.height, sz.width, CV_8UC3, image->data(), image->pitch());
  cv::Mat mat_8u = mat_imageCpu.clone();
  if(normalize)
    cv::normalize(mat_8u, mat_8u, 0, 255, cv::NORM_MINMAX);
  cv::Mat bgr(sz.height, sz.width, CV_8UC3);
  cv::cvtColor(mat_8u, bgr, CV_RGB2BGR);
  return cv::imwrite(filename, bgr);
}

//-----------------------------------------------------------------------------
bool imsave(iu::ImageCpu_8u_C4* image, const std::string& filename, const bool& normalize)
{
  // TODO do normalization as in 32f_C4
  IuSize sz = image->size();
  cv::Mat mat_imageCpu(sz.height, sz.width, CV_8UC4, image->data(), image->pitch());
  cv::Mat mat_8u = mat_imageCpu.clone();
  if(normalize)
    cv::normalize(mat_8u, mat_8u, 0, 255, cv::NORM_MINMAX);
  cv::Mat bgr(sz.height, sz.width, CV_8UC3);
  cv::cvtColor(mat_8u, bgr, CV_RGBA2BGR);
  return cv::imwrite(filename, bgr);
}

//-----------------------------------------------------------------------------
bool imsave(iu::ImageCpu_32f_C1* image, const std::string& filename, const bool& normalize)
{
  IuSize sz = image->size();
  cv::Mat mat_imageCpu(sz.height, sz.width, CV_32FC1, image->data(), image->pitch());
  cv::Mat mat_32f = mat_imageCpu.clone();
  if(normalize)
    cv::normalize(mat_32f, mat_32f, 0.0, 1.0, cv::NORM_MINMAX);
  cv::Mat mat_8u(sz.height, sz.width, CV_8UC1);
  mat_32f.convertTo(mat_8u, mat_8u.type(), 255, 0);
  return cv::imwrite(filename, mat_8u);
}

//-----------------------------------------------------------------------------
bool imsave(iu::ImageCpu_32f_C3* image, const std::string& filename, const bool& normalize)
{
  IuSize sz = image->size();
  cv::Mat mat_imageCpu(sz.height, sz.width, CV_32FC3, image->data(), image->pitch());
  cv::Mat mat_32f = mat_imageCpu.clone();
  // get/normalize all the channels seperately
  cv::Mat r(mat_32f.rows, mat_32f.cols, CV_32FC1);
  cv::Mat g(mat_32f.rows, mat_32f.cols, CV_32FC1);
  cv::Mat b(mat_32f.rows, mat_32f.cols, CV_32FC1);
  cv::Mat rgb[] = {r, g, b};
  cv::split(mat_32f, rgb);

  if(normalize)
  {
    cv::normalize(r, r, 0.0, 1.0, cv::NORM_MINMAX);
    cv::normalize(g, g, 0.0, 1.0, cv::NORM_MINMAX);
    cv::normalize(b, b, 0.0, 1.0, cv::NORM_MINMAX);
  }
  cv::Mat mat_8u(sz.height, sz.width, CV_8UC3);
  mat_32f.convertTo(mat_8u, mat_8u.type(), 255, 0);
  cv::Mat bgr(sz.height, sz.width, CV_8UC3);
  cv::cvtColor(mat_8u, bgr, CV_RGB2BGR);
  return cv::imwrite(filename, bgr);
}

//-----------------------------------------------------------------------------
bool imsave(iu::ImageCpu_32f_C4* image, const std::string& filename, const bool& normalize)
{
  IuSize sz = image->size();
  cv::Mat mat_imageCpu(sz.height, sz.width, CV_32FC4, image->data(), image->pitch());
  cv::Mat mat_32f = mat_imageCpu.clone();
  // get/normalize all the channels seperately
  cv::Mat r(mat_32f.rows, mat_32f.cols, CV_32FC1);
  cv::Mat g(mat_32f.rows, mat_32f.cols, CV_32FC1);
  cv::Mat b(mat_32f.rows, mat_32f.cols, CV_32FC1);
  cv::Mat a(mat_32f.rows, mat_32f.cols, CV_32FC1);
  cv::Mat rgba[] = {r, g, b, a};
  cv::split(mat_32f, rgba);

  if(normalize)
  {
    cv::normalize(r, r, 0.0, 1.0, cv::NORM_MINMAX);
    cv::normalize(g, g, 0.0, 1.0, cv::NORM_MINMAX);
    cv::normalize(b, b, 0.0, 1.0, cv::NORM_MINMAX);
    cv::normalize(a, a, 0.0, 1.0, cv::NORM_MINMAX);
  }
  cv::Mat mat_8u(sz.height, sz.width, CV_8UC4);
  cv::merge(rgba, 4, mat_32f);
  mat_32f.convertTo(mat_8u, mat_8u.type(), 255, 0);
  cv::Mat bgr(sz.height, sz.width, CV_8UC3);
  cv::cvtColor(mat_8u, bgr, CV_RGBA2BGR);
  return cv::imwrite(filename, bgr);
}


//-----------------------------------------------------------------------------
bool imsave(iu::ImageCpu_32s_C1* image, const std::string& filename, const bool& normalize)
{
  IuSize sz = image->size();
  cv::Mat mat_imageCpu(sz.height, sz.width, CV_32SC1, image->data(), image->pitch());
  cv::Mat mat_32f(sz.height, sz.width, CV_32FC1);
  mat_imageCpu.convertTo(mat_32f, mat_32f.type());
  if(normalize)
    cv::normalize(mat_32f, mat_32f, 0.0, 255.0, cv::NORM_MINMAX);
  cv::Mat mat_8u(sz.height, sz.width, CV_8UC1);
  mat_32f.convertTo(mat_8u, mat_8u.type());
  return cv::imwrite(filename, mat_8u);
}


//-----------------------------------------------------------------------------
bool imsave(iu::ImageGpu_8u_C1* image, const std::string& filename, const bool& normalize)
{
  iu::ImageCpu_8u_C1 cpu_image(image->size());
  iuprivate::copy(image, &cpu_image);
  return iuprivate::imsave(&cpu_image, filename, normalize);
}

//-----------------------------------------------------------------------------
bool imsave(iu::ImageGpu_8u_C4* image, const std::string& filename, const bool& normalize)
{
  iu::ImageCpu_8u_C4 cpu_image(image->size());
  iuprivate::copy(image, &cpu_image);
  return iuprivate::imsave(&cpu_image, filename, normalize);
}

//-----------------------------------------------------------------------------
bool imsave(iu::ImageGpu_32f_C1* image, const std::string& filename, const bool& normalize)
{
  iu::ImageCpu_32f_C1 cpu_image(image->size());
  iuprivate::copy(image, &cpu_image);
  return iuprivate::imsave(&cpu_image, filename, normalize);
}

//-----------------------------------------------------------------------------
bool imsave(iu::ImageGpu_32f_C4* image, const std::string& filename, const bool& normalize)
{
  iu::ImageCpu_32f_C4 cpu_image(image->size());
  iuprivate::copy(image, &cpu_image);
  return iuprivate::imsave(&cpu_image, filename, normalize);
}

//-----------------------------------------------------------------------------
bool imsave(iu::ImageGpu_32s_C1* image, const std::string& filename, const bool& normalize)
{
  iu::ImageCpu_32s_C1 cpu_image(image->size());
  iuprivate::copy(image, &cpu_image);
  return iuprivate::imsave(&cpu_image, filename, normalize);
}



/* ****************************************************************************

   imshow

 */

void imshow(const std::string& winname, const cv::Mat& mat)
{
  //cv::namedWindow(winname, CV_WINDOW_NORMAL || CV_WINDOW_KEEPRATIO);
  cv::namedWindow(winname, CV_WINDOW_NORMAL);
  cv::imshow(winname, mat);
}

//-----------------------------------------------------------------------------
void imshow(iu::ImageCpu_8u_C1* image, const std::string& winname, const bool& normalize)
{
  IuSize sz = image->size();
  cv::Mat mat_8u(sz.height, sz.width, CV_8UC1, image->data(), image->pitch());
  if(normalize)
    cv::normalize(mat_8u, mat_8u, 0, 255, cv::NORM_MINMAX);

  iuprivate::imshow(winname, mat_8u);
}

//-----------------------------------------------------------------------------
void imshow(iu::ImageCpu_8u_C3* image, const std::string& winname, const bool& normalize)
{
  IuSize sz = image->size();
  cv::Mat mat_8u(sz.height, sz.width, CV_8UC3, (unsigned char*)image->data(), image->pitch());
  if(normalize)
    cv::normalize(mat_8u, mat_8u, 0, 255, cv::NORM_MINMAX);
  cv::Mat bgr(sz.height, sz.width, CV_8UC3);
  cv::cvtColor(mat_8u, bgr, CV_RGB2BGR);

  iuprivate::imshow(winname, bgr);
}

//-----------------------------------------------------------------------------
void imshow(iu::ImageCpu_8u_C4* image, const std::string& winname, const bool& normalize)
{
  IuSize sz = image->size();
  cv::Mat mat_8u(sz.height, sz.width, CV_8UC4, (unsigned char*)image->data(), image->pitch());
  if(normalize)
    cv::normalize(mat_8u, mat_8u, 0, 255, cv::NORM_MINMAX);
  cv::Mat bgr(sz.height, sz.width, CV_8UC3);
  cv::cvtColor(mat_8u, bgr, CV_RGBA2BGR);

  iuprivate::imshow(winname, bgr);
}

//-----------------------------------------------------------------------------
void imshow(iu::ImageCpu_32f_C1* image, const std::string& winname, const bool& normalize)
{
  IuSize sz = image->size();
  cv::Mat mat_32f(sz.height, sz.width, CV_32FC1, image->data(), image->pitch());
  if(normalize)
    cv::normalize(mat_32f, mat_32f, 0.0, 1.0, cv::NORM_MINMAX);

  iuprivate::imshow(winname, mat_32f);
}

//-----------------------------------------------------------------------------
void imshow(iu::ImageCpu_32f_C3* image, const std::string& winname, const bool& normalize)
{
  IuSize sz = image->size();
  cv::Mat mat_32f(sz.height, sz.width, CV_32FC3, (float*)image->data(), image->pitch());
  if(normalize)
    cv::normalize(mat_32f, mat_32f, 0.0, 1.0, cv::NORM_MINMAX);
  cv::Mat bgr;
  cv::cvtColor(mat_32f, bgr, CV_RGB2BGR);

  iuprivate::imshow(winname, bgr);
}

//-----------------------------------------------------------------------------
void imshow(iu::ImageCpu_32f_C4* image, const std::string& winname, const bool& normalize)
{
  IuSize sz = image->size();
  cv::Mat mat_32f(sz.height, sz.width, CV_32FC4, (float*)image->data(), image->pitch());
  cv::Mat bgr;
  cv::cvtColor(mat_32f, bgr, CV_RGBA2BGR);

  iuprivate::imshow(winname, bgr);
}

//-----------------------------------------------------------------------------
void imshow(iu::ImageGpu_8u_C1* image, const std::string& winname, const bool& normalize)
{
  iu::ImageCpu_8u_C1 cpu_image(image->size());
  iuprivate::copy(image, &cpu_image);

  iuprivate::imshow(&cpu_image, winname, normalize);
}

//-----------------------------------------------------------------------------
void imshow(iu::ImageGpu_8u_C4* image, const std::string& winname, const bool& normalize)
{
  iu::ImageCpu_8u_C4 cpu_image(image->size());
  iuprivate::copy(image, &cpu_image);

  iuprivate::imshow(&cpu_image, winname, normalize);
}

//-----------------------------------------------------------------------------
void imshow(iu::ImageGpu_32f_C1* image, const std::string& winname, const bool& normalize)
{
  iu::ImageCpu_32f_C1 cpu_image(image->size());
  iuprivate::copy(image, &cpu_image);

  iuprivate::imshow(&cpu_image, winname, normalize);
}

//-----------------------------------------------------------------------------
void imshow(iu::ImageGpu_32f_C4* image, const std::string& winname, const bool& normalize)
{
  iu::ImageCpu_32f_C4 cpu_image(image->size());
  iuprivate::copy(image, &cpu_image);

  iuprivate::imshow(&cpu_image, winname, normalize);
}


void printToFile(iu::ImageCpu_32f_C1 *image, const std::string &name)
{
  printf("Writing file %s\n", name.c_str());

  FILE *fh = fopen(name.c_str(), "w");
  if (!fh)
  {
    printf("Error: Could not open file %s for writing\n", name.c_str());
    return;
  }

  fprintf(fh, "%d %d\n", image->width(), image->height());
  for (unsigned int y=0; y < image->height(); y++)
  {
    for (unsigned int x=0; x < image->width(); x++)
    {
      fprintf(fh, "%.7f ", *image->data(x,y));
    }
    fprintf(fh, "\n");
  }

  fclose(fh);
}

void printToFile(iu::ImageCpu_8u_C1 *image, const std::string &name)
{
  printf("Writing file %s\n", name.c_str());

  FILE *fh = fopen(name.c_str(), "w");
  if (!fh)
  {
    printf("Error: Could not open file %s for writing\n", name.c_str());
    return;
  }

  fprintf(fh, "%d %d\n", image->width(), image->height());
  for (unsigned int y=0; y < image->height(); y++)
  {
    for (unsigned int x=0; x < image->width(); x++)
    {
      fprintf(fh, "%.0f ", static_cast<float>(*image->data(x,y)));
    }
    fprintf(fh, "\n");
  }

  fclose(fh);
}


void printToFile(iu::ImageGpu_8u_C1 *image, const std::string &name)
{
  iu::ImageCpu_8u_C1 host_image(image->size());
  iuprivate::copy(image, &host_image);

  printToFile(&host_image, name);
}

void printToFile(iu::ImageGpu_32f_C1 *image, const std::string &name)
{
  iu::ImageCpu_32f_C1 host_image(image->size());
  iuprivate::copy(image, &host_image);

  printToFile(&host_image, name);
}


void printToFile(iu::LinearDeviceMemory_32f_C1 *data, const std::string &name)
{
  iu::LinearHostMemory_32f_C1 host_data(data->length());
  iuprivate::copy(data, &host_data);

  iuprivate::printToFile(&host_data, name);
}

void printToFile(iu::LinearHostMemory_32f_C1 *data, const std::string &name)
{
  printf("Writing file %s\n", name.c_str());

  FILE *fh = fopen(name.c_str(), "w");
  if (!fh)
  {
    printf("Error: Could not open file %s for writing\n", name.c_str());
    return;
  }

  for (unsigned int x=0; x < data->length(); x++)
    fprintf(fh, "%.6f ", *data->data(x));

  fclose(fh);
}


} // namespace iuprivate
