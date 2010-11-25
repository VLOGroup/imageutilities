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
 * Class       : QGL_IMAGE_GPU_WIDGET
 * Language    : C++
 * Description : Definition of a QGLWidget rendering GPU memory (2D)
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */


#ifndef IU_QGL_IMAGE_GPU_WIDGET_H
#define IU_QGL_IMAGE_GPU_WIDGET_H

#include "qgl_image_gpu_widget_p.h"

namespace iu {

class QGLImageGpuWidget : public iuprivate::QGLImageGpuWidget
{
  //Q_OBJECT        // must include this if you use Qt signals/slots

public:
  QGLImageGpuWidget(QWidget *parent);
  virtual ~QGLImageGpuWidget();
//  void setImage(iu::ImageGpu_8u_C4* image);

protected:

};

} // namespace iuprivate

#endif // IU_QGL_IMAGE_GPU_WIDGET_H
