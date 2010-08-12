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

#ifndef IU_NPPGLWIDGET_H
#define IU_NPPGLWIDGET_H

#include <QObject>
#include <iucore/globaldefs.h>
#include <iucore/coredefs.h>
#include <iucore/memorydefs.h>
#include "nppglwidget_p.h" // TODO FIXMEEE I really want to remove this include but have the dependency due to inheritence.

//// forward declarations
//namespace iuprivate {
//  class NppGLWidgetPrivate;
//}
class QMenu;
class QTimer;
class QSignalMapper;
class QWidgetAction;
class QLabel;
class QLabel;
class QSlider;

namespace iu {

class IU_DLLAPI NppGLWidget : public iuprivate::NppGLWidgetPrivate
{
  Q_OBJECT

public:
  NppGLWidget(QWidget* parent);

  /** Constructor for single channel images.
     * Constructs a NppGLWidget with a parent widget and renders the given image.
     * @param[in] parent Parent widget.
     * @param[in] image Image to render. Currently only device images are supported.
     * @param[in] autoupdate Flag that toggles automated updates via a timer. (default = false)
     * @param[in] autoresize Flag if widget is automatically resized when a different sized input image is set (default = true).
     */
  NppGLWidget(QWidget* parent, iu::Image* image, bool autoupdate = false, bool autoresize = true);

  /** Destructor. */
  virtual ~NppGLWidget();

  /** Rendered image is automatically resized to the widget size.
   * @param[in] val Flag if automatic resize should be turned on (TRUE, DEFAULT) or off (FALSE).
   */
  void setAutoresize(bool val);

  /** Sets the number of frames per second that will be drawn if run with autoupdate = true
  * @param[in] fps Frames per second.
  */
  void setFps(float fps);

  /** Sets new minimum and maximum values as they are set to 0 and 1 by default.
    * Additionally 'true' values can be given, that correspond to the given minimum and maximum values.
    * The 'true' values are used for display, while the other values are used for calculations with the
    * image
    * @param[in] minimum New minimum value.
    * @param[in] maximum New maximum value.
    * @param[in] true_minimum True minimum value corresponding to minimum.
    * @param[in] true_maximum True maximum value corresponding to maximum.
    */
  void setGrayValueRange(float minimum, float maximum, float true_minimum, float true_maximum);

public slots:

  /** Slot to set a new image to be rendered.
   * @param[in] image Image to render.
   */
  void updateImage(iu::Image* image);

protected:
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


private:

};

} // namespace iu

#endif // IU_NPPGLWIDGET_H
