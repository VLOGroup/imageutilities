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
 * Module      : GUI
 * Class       : QImageCpuDisplay
 * Language    : C++
 * Description : Definition of an image display for ImageCpu
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IU_IMAGE_CPU_DISPLAY_H
#define IU_IMAGE_CPU_DISPLAY_H

#include "image_cpu_display_p.h"

namespace iu {

//-----------------------------------------------------------------------------
// QImageDisplay32f
//-----------------------------------------------------------------------------
class IU_DLLAPI QImageCpuDisplay : public iuprivate::QImageCpuDisplay
{
  Q_OBJECT

public:
  QImageCpuDisplay(iu::ImageCpu_8u_C1* image, const std::string& title,
                   unsigned char minval=0x00, unsigned char maxval=0xff);
  QImageCpuDisplay(iu::ImageCpu_8u_C4* image, const std::string& title,
                   unsigned char minval=0x00, unsigned char maxval=0xff);
  QImageCpuDisplay(iu::ImageCpu_32f_C1* image, const std::string& title,
                   float minval=0.0f, float maxval=1.0f);
  QImageCpuDisplay(iu::ImageCpu_32f_C4* image, const std::string& title,
                   float minval=0.0f, float maxval=1.0f);

  virtual ~QImageCpuDisplay();

};


////-----------------------------------------------------------------------------
//// QMultiImageDisplay32f
////-----------------------------------------------------------------------------
//class QMultiImageDisplay32f: public QWidget
//{
//  Q_OBJECT

//public:
//  QMultiImageDisplay32f(const float* data,
//    size_t width, size_t height, size_t depth,
//    size_t pitch, size_t stride,
//    const std::string& title = "", float minval = 0.0f, float maxval = 1.0f);

//  ~QMultiImageDisplay32f();

//public slots:
//  void setSlice(int);

//private:
//  inline int scale(float value) {
//    return (int) (255.0f / (maxval_ - minval_) * (value - minval_));
//  };

//  inline bool isInside(int x, int y){
//    if (x >= 0 && x < (int)width_ && y >= 0 && y < (int)height_)
//      return true;
//    return false;
//  };

//  void showSlice(int plane);

//  float minval_, maxval_;
//  size_t width_, height_, depth_, pitch_, stride_;

//  std::vector<QImage*> images_;

//  QLabel* my_label_;
//  QSlider* my_slider_;
//  QSpinBox* my_spinbox_;

//  QMultiImageDisplay32f(const QMultiImageDisplay32f&);
//  QMultiImageDisplay32f();
//};

} // namespace iu

#endif // IU_IMAGE_CPU_DISPLAY_H
