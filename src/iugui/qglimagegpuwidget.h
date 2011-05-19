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

#include <GL/glew.h>
#include <QObject>
#include <QGLWidget>
#include <cuda_gl_interop.h>

#include "iudefs.h"

namespace iu {

class IU_DLLAPI QGLImageGpuWidget : public QGLWidget
{
  Q_OBJECT        // must include this if you use Qt signals/slots

public:
  QGLImageGpuWidget(QWidget *parent=0);
  virtual ~QGLImageGpuWidget();
  void setImage(iu::ImageGpu_8u_C1* image, bool normalize = false);
  void setImage(iu::ImageGpu_32f_C1* image, bool normalize = false);
  void setImage(iu::ImageGpu_8u_C4* image, bool normalize = false);
  void setImage(iu::ImageGpu_32f_C4* image, bool normalize = false);

  void setMinMax(float min, float max);
  void autoMinMax();
  void setAutoNormalize(bool flag);

signals:
  void mouseMoved(int from_x, int from_y, int to_x, int to_y);
  void mousePressed(int x, int y);
  void mousePressed(int x, int y, int global_x, int global_y);
  void mouseReleased(int x, int y);

private slots:
//  /** Invokes timer triggered updates if the autoupdate ability is set. */
//  void slotTimerCallback();

//  /** Resets the zoom to 100% (1:1). */
//  void slotZoomReset();

//  void slotZoom0p25();
//  void slotZoom0p33();
//  void slotZoom0p5();
//  void slotZoom2();
//  void slotZoom3();
//  void slotZoom4();

//  /** Sets the minimum gray value that should be displayed from the context menu slider. */
//  void slotMinimumValue(int value);

//  /** Sets the maximum gray value that should be displayed from the context menu slider. */
//  void slotMaximumValue(int value);

//  /** Activates the corresponing overlay to be displayed. */
//  void slotActivateOverlay(const QString& overlay_name);

protected:
  void mousePressEvent(QMouseEvent *event);
  void mouseReleaseEvent(QMouseEvent *event);
  void mouseMoveEvent(QMouseEvent *event);

  void wheelEvent(QWheelEvent *event);


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
  bool normalize_;
  float min_;
  float max_;
  bool init_ok_;
  float zoom_;

  // mouse interaction
   int  mouse_x_old_; /**< Previous mouse position (x coordinate). */
   int  mouse_y_old_; /**< Previous mouse position (y coordinate). */
   int  mouse_x_;     /**< Current mouse position (x coordinate). */
   int  mouse_y_;     /**< Current mouse position (y coordinate). */

};

} // namespace iuprivate

#endif // IUPRIVATE_QGL_IMAGE_GPU_WIDGET_H
