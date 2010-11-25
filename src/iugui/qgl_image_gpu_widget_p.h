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


#ifndef IUPRIVATE_QGL_IMAGE_GPU_WIDGET_H
#define IUPRIVATE_QGL_IMAGE_GPU_WIDGET_H

#include <GL/glew.h>
//#include <QObject>
#include <QGLWidget>
#include <cuda_gl_interop.h>

#include "iudefs.h"

namespace iuprivate {

class QGLImageGpuWidget : public QGLWidget
{
  //Q_OBJECT        // must include this if you use Qt signals/slots

public:
  QGLImageGpuWidget(QWidget *parent);
  virtual ~QGLImageGpuWidget();
  void setImage(iu::ImageGpu_8u_C1* image);
  void setImage(iu::ImageGpu_32f_C1* image, float min=0.0f, float max=1.f);
  void setImage(iu::ImageGpu_8u_C4* image);
  void setImage(iu::ImageGpu_32f_C4* image);

protected:

  void initializeGL();
  void resizeGL(int w, int h);
  void createTexture();
  void deleteTexture();
  void createPbo();
  void deletePbo();
  bool init();
  //void resizeGL(int w, int h);
  void paintGL();

  GLuint gl_pbo_; /**< OpenGL PBO name. */
  GLuint gl_tex_; /**< OpenGL texture name. */
  struct cudaGraphicsResource *cuda_pbo_resource_; /**< Handles OpenGL <-> CUDA exchange. */

  iu::Image* image_; /**< image that will be displayed. Currently 1-channel and 4-channel images are supported. */
  unsigned int num_channels_;
  unsigned int bit_depth_;
  float min_;
  float max_;
  bool init_ok_;
  float zoom_;
};

} // namespace iuprivate

#endif // IUPRIVATE_QGL_IMAGE_GPU_WIDGET_H
