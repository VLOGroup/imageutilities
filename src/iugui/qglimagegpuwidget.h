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
#include <QString>
#include <QRadioButton>
#include <cuda_gl_interop.h>

#include "iudefs.h"
#include "overlay.h"

// forward declarations
class QMenu;
class QSignalMapper;
class QWidgetAction;

namespace iu {

//-----------------------------------------------------------------------------
/** \brief QGLImageGpuWidget: An Qt OpenGL widget for CUDA/GPU memory.
  \ingroup iugui
  */
class IUGUI_DLLAPI QGLImageGpuWidget : public QGLWidget
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

  void addOverlay(QString name, iu::Image* constraint_image,
                  iu::LinearMemory* lut_values, iu::LinearDeviceMemory_8u_C4* lut_colors,
                  bool active = true, IuComparisonOperator comp_op = IU_EQUAL);

  void getPboOutput(iu::ImageGpu_8u_C4* image);
  unsigned int imageWidth();
  unsigned int imageHeight();

signals:
  void mouseMoved(int from_x, int from_y, int to_x, int to_y);
  void mousePressed(int x, int y);
  void mousePressed(int x, int y, int global_x, int global_y);
  void mouseReleased(int x, int y);

  void pan(int from_x, int from_y, int to_x, int to_y);

  void zoomChanged(float val);
  void zoomed(double factor);

  void showToolbar(bool val);
  void pixelInfo(QString text);

public slots:
  void setZoom(float val);

private slots:
  //  /** Invokes timer triggered updates if the autoupdate ability is set. */
  //  void slotTimerCallback();

  /** Resets the zoom to 100% (1:1). */
  void slotZoomReset();

  void slotZoom0p25();
  void slotZoom0p33();
  void slotZoom0p5();
  void slotZoom2();
  void slotZoom3();
  void slotZoom4();

  void useNN(bool val);
  void useLinear(bool val);

  //  /** Sets the minimum gray value that should be displayed from the context menu slider. */
  //  void slotMinimumValue(int value);

  //  /** Sets the maximum gray value that should be displayed from the context menu slider. */
  //  void slotMaximumValue(int value);

  /** Activates the corresponing overlay to be displayed. */
  void slotActivateOverlay(const QString& overlay_name);

protected:
  void mousePressEvent(QMouseEvent *event);
  void mouseReleaseEvent(QMouseEvent *event);
  void mouseMoveEvent(QMouseEvent *event);
  void wheelEvent(QWheelEvent *event);
  void contextMenuEvent(QContextMenuEvent *);

  void createActions();
  void initializeGL();
  void resizeGL(int w, int h);
  void createTexture();
  void deleteTexture();
  void createPbo();
  void deletePbo();
  bool init();
  //void resizeGL(int w, int h);
  void paintGL();
  void fillPbo(iu::ImageGpu_8u_C4* output=NULL);

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

  // overlays
  iuprivate::OverlayList overlay_list_;

  // context menus and actions
  QMenu *context_menu_; /**< Menu that is displayed when a context menu event is invoked. */
  QSignalMapper *overlay_signal_mapper_; /**< Maps the signal from the overlay checkbox to active the overlay. */
  QList<QWidgetAction*> action_list_overlays_; /**< Checkboxes for all added overlays. */
  QAction *action_close_; /**< Action that closes the widget. */
  QAction *action_zoom_reset_; /**< Action that sets the zoom to 100% (1:1) */
  QAction *action_zoom_0p25_;
  QAction *action_zoom_0p33_;
  QAction *action_zoom_0p5_;
  QAction *action_zoom_2_;
  QAction *action_zoom_3_;
  QAction *action_zoom_4_;
  QMenu *action_zoom_group_;

  QMenu *action_interpolate_group_;
  QWidgetAction *action_NN_;
  QWidgetAction *action_Linear_;

  QWidgetAction *show_Toolbar_;

  bool filter_linear_;
  int button_;

  QCursor old_cursor_;

  //  QWidgetAction *action_slider_min_; /**< Slider for the min gray value. */
  //  QWidgetAction *action_slider_max_; /**< Slider for the max gray value. */
  //  QWidgetAction *action_slider_range_label_; /**< Slider range label. */
  //  QLabel* min_max_slider_label_; /**< Slider range label text. */
  //  QSlider *slider_min_;
  //  QSlider *slider_max_;

private:
  // Copy and asignment operator intentionally declared private.
  QGLImageGpuWidget(const QGLImageGpuWidget&);
  QGLImageGpuWidget& operator= (const QGLImageGpuWidget&);
};

} // namespace iu

#endif // IUPRIVATE_QGL_IMAGE_GPU_WIDGET_H
