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
 * Class       : NppGLWidget
 * Language    : C++
 * Description : Definition of a QGLWidget rendering GPU memory (2D)
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUPRIVATE_NPPGLWIDGETPRIVATE_H
#define IUPRIVATE_NPPGLWIDGETPRIVATE_H

#include <QObject>
#include <QGLWidget>

// forward declarations
class QMenu;
class QTimer;
class QSignalMapper;
class QWidgetAction;
class QLabel;
class QLabel;
class QSlider;

#include <iucore/coredefs.h>
#include <iucore/memorydefs.h>

//
//  W A R N I N G
//  -------------
//
// This file is not part of the IU API.  It exists purely as an
// implementation detail.  This header file may change from version to
// version without notice, or even be removed.
//

namespace iuprivate {

class NppGLWidgetPrivate : public QGLWidget
{
  Q_OBJECT

public:
//  NppGLWidgetPrivate(QWidget* parent); // default constructur might be needed later?
  NppGLWidgetPrivate(QWidget* parent, iu::Image* image, bool autoupdate = false, bool autoresize = true);
  virtual ~NppGLWidgetPrivate();

  void setAutoresize(bool val);
  void setFps(float fps);
  void setGrayValueRange(float minimum, float maximum, float true_minimum, float true_maximum);

signals:

  void closeWindow();
  void mouseMoved(int from_x, int from_y, int to_x, int to_y);
  void mousePressed(int x, int y);
  void mousePressed(int x, int y, int global_x, int global_y);
  void mouseReleased(int x, int y);

public slots:

  void updateImage(iu::Image* image);

private slots:

  void slotTimerCallback();
  void slotZoomReset();
  void slotMinimumValue(int value);
  void slotMaximumValue(int value);
  void slotActivateOverlay(const QString& overlay_name);

protected:

//  void addOverlay(Cuda::DeviceOverlayBase* overlay);
  virtual void setImage(iu::Image* image);
  virtual void init();
  virtual void createActions();
  virtual void closeEvent(QCloseEvent *event);

  virtual void initializeGL();
  virtual void createPbo(int width, int height);
  virtual void deletePbo();
  virtual void createTexture(int width, int height);
  virtual void deleteTexture();
  virtual void bind();
  virtual void unbind();
  virtual void paintGL();
  virtual void renderScene();
  virtual void resizeGL(int w, int h);

  virtual void mousePressEvent(QMouseEvent *event);
  virtual void mouseReleaseEvent(QMouseEvent *event);
  virtual void mouseMoveEvent(QMouseEvent *event);

  virtual void contextMenuEvent(QContextMenuEvent* event);
  virtual void wheelEvent(QWheelEvent *event);
  virtual void checkForOpenGLError(const std::string& message);

  QTimer* timer_; /**< Timer to trigger automated updates if enabled. */
  GLuint pbo_;   /**< OpenGL picture buffer. */
  bool registered_; /**< Flag if pbo is registered or not. */
  GLuint tex_;   /**< OpenGL texture. */

//  OverlayList overlay_list_; /**< List with all available overlays. */

  bool image_loaded_; /**< Flag if an image was already loaded. */
  iu::Image* image_; /**< Rendered single channel float image. */

  int  mouse_x_old_; /**< Previous mouse position (x coordinate). */
  int  mouse_y_old_; /**< Previous mouse position (y coordinate). */
  int  mouse_x_;     /**< Current mouse position (x coordinate). */
  int  mouse_y_;     /**< Current mouse position (y coordinate). */

  float minimum_value_; /**< Min gray value to display. */
  float maximum_value_; /**< Max gray value to display. */

  float minimum_;       /** Minimum value for sliders */
  float maximum_;       /** Maximum value for sliders */
  float true_minimum_;  /** True minimum value for sliders */
  float true_maximum_;  /** True maximum value for sliders */

  IuSize image_size_;
  bool autoupdate_;
  bool autoresize_; /**< Flag if image is scaled and fitted due to widget resizes. */
  float  zoom_;  /**< Zoom value that influences the OpenGL viewport. */
  float fps_;

  QMenu *context_menu_; /**< Menu that is displayed when a context menu event is invoked. */
  QSignalMapper *overlay_signal_mapper_; /**< Maps the signal from the overlay checkbox to active the overlay. */
  QList<QWidgetAction*> action_list_overlays_; /**< Checkboxes for all added overlays. */
  QAction *action_close_; /**< Action that closes the widget. */
  QAction *action_zoom_reset_; /**< Action that sets the zoom to 100% (1:1) */
  QWidgetAction *action_slider_min_; /**< Slider for the min gray value. */
  QWidgetAction *action_slider_max_; /**< Slider for the max gray value. */
  QWidgetAction *action_slider_range_label_; /**< Slider range label. */
  QLabel* min_max_slider_label_; /**< Slider range label text. */
  QSlider *slider_min_;
  QSlider *slider_max_;


private:
  const int float_num_slider_values;

  NppGLWidgetPrivate();
  NppGLWidgetPrivate(const NppGLWidgetPrivate&);
  NppGLWidgetPrivate& operator= (const NppGLWidgetPrivate&);

};

} // namespace iuprivate

#endif // IUPRIVATE_NppGLWidgetPrivate_H
