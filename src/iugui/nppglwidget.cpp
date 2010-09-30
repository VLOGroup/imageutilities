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
 * Description : Implementation of a QGLWidget rendering GPU memory (2D)
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#include <assert.h>
#include <GL/glew.h>
#include <math.h>

#include <QCoreApplication>
#include <QApplication>
#include <QMouseEvent>
#include <QMenu>
#include <QTimer>
#include <QSignalMapper>
#include <QLabel>
#include <QWidgetAction>
#include <QSlider>

#include "nppglwidget_p.h"
#include "nppglwidget.h"
#include "nppglwidget.cuh"


namespace iuprivate {

//-----------------------------------------------------------------------------
NppGLWidgetPrivate::NppGLWidgetPrivate(QWidget* parent, iu::Image* image, bool autoupdate, bool autoresize) :
  QGLWidget(parent), timer_(0),
  pbo_(0), registered_(false), tex_(0),
  image_loaded_(false), image_(NULL),
  minimum_value_(0.0f), maximum_value_(1.0f), minimum_(0.0f), maximum_(1.0f), true_minimum_(0.0f), true_maximum_(1.0f),
  autoupdate_(autoupdate), autoresize_(autoresize), zoom_(1.0), fps_(30.0f),
  float_num_slider_values(10000)
{
  Q_ASSERT(image);
  Q_ASSERT(image->onDevice()); // only device images are supported for the moment!
  timer_ = new QTimer();
  context_menu_ = new QMenu("Widget properties", this);
  this->createActions();

  // forward all signals from the implementation class towards the public interface signals
  //connect(this, SIGNAL(closeWindow()), public_interface, SIGNAL(closeWindow()));
//  connect(this, SIGNAL(mouseMoved(int, int, int, int)), public_interface, SIGNAL(mouseMoved(int, int, int, int)));
//  connect(this, SIGNAL(mousePressed(int, int)), public_interface, SIGNAL(mousePressed(int, int)));
//  connect(this, SIGNAL(mousePressed(int, int, int, int)), public_interface, SIGNAL(mousePressed(int, int, int, int)));
//  connect(this, SIGNAL(mouseReleased(int, int)), public_interface, SIGNAL(mouseReleased(int, int)));

  this->init();
  this->updateImage(image);
}

//-----------------------------------------------------------------------------
NppGLWidgetPrivate::~NppGLWidgetPrivate()
{
  if (autoupdate_ && timer_->isActive())
    timer_->stop();

  delete(timer_);

  this->deletePbo();
  this->deleteTexture();

//  while(!overlay_list_.isEmpty())
//    delete(overlay_list_.takeFirst());
  checkForOpenGLError(std::string("Destructor"));
//  VMLIB_CUT_CHECK_ERROR_GL ();
}

//-----------------------------------------------------------------------------
void NppGLWidgetPrivate::setAutoresize(bool val)
{
  autoresize_ = val;
}

//-----------------------------------------------------------------------------
void NppGLWidgetPrivate::closeEvent(QCloseEvent *event)
{
  if (autoupdate_)
    timer_->stop();
  checkForOpenGLError(std::string("closeEvent"));
//  VMLIB_CUT_CHECK_ERROR_GL ();
}

//-----------------------------------------------------------------------------
void NppGLWidgetPrivate::initializeGL()
{
  // initialize necessary OpenGL extensions
  glewInit ();
  if (!glewIsSupported("GL_VERSION_2_0 " "GL_ARB_pixel_buffer_object " "GL_EXT_framebuffer_object "))
  {
    fprintf (stderr, "ERROR: Support for necessary OpenGL extensions missing.\n");
    fflush (stderr);
    exit(-1);
  }

  // Set up the rendering context, define display lists etc.:
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glEnable(GL_DEPTH_TEST);
  glShadeModel(GL_FLAT);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  checkForOpenGLError(std::string("initializeGL"));
//  VMLIB_CUT_CHECK_ERROR_GL ();
}

//-----------------------------------------------------------------------------
void NppGLWidgetPrivate::init()
{
  // initialize cuda textures
  cuInitTextures();

  // initialize OpenGL
  updateGL();

  // create a timer
  timer_->setSingleShot(false);
  connect(timer_, SIGNAL(timeout()),this, SLOT(slotTimerCallback()));
  checkForOpenGLError(std::string("init"));
}


//-----------------------------------------------------------------------------
void NppGLWidgetPrivate::checkForOpenGLError(const std::string& message)
{
  this->makeCurrent();
  GLenum errCode;

  if( (errCode = glGetError()) != GL_NO_ERROR )
  {
    const GLubyte *errString = gluErrorString(errCode);
    fprintf(stderr, "OpenGL error: %s @ %s\n", errString, message.c_str());
  }
}

//-----------------------------------------------------------------------------
void NppGLWidgetPrivate::createPbo(int width, int height)
{
  this->makeCurrent();
  // set up vertex data parameter
  uint num_texels = width * height;
  uint num_values = num_texels * 4;
  uint size_tex_data = sizeof (GLubyte) * num_values;
  void *data = malloc (size_tex_data);

  // create buffer object
  glGenBuffers(1, &pbo_);
  IU_ASSERT(pbo_ != 0);

  this->bind();

  // buffer data
  glBufferData(GL_ARRAY_BUFFER, size_tex_data, data, GL_DYNAMIC_DRAW);
  free(data);

  unbind();

  checkForOpenGLError(std::string("createPbo"));
//  VMLIB_CUT_CHECK_ERROR_GL ();
}

//-----------------------------------------------------------------------------
void NppGLWidgetPrivate::deletePbo()
{
  this->makeCurrent();
  cuPboUnregister(pbo_, registered_);
  glBindBuffer (GL_ARRAY_BUFFER, pbo_);
  glDeleteBuffers (1, &pbo_);
  checkForOpenGLError(std::string("deletePbo"));
//  VMLIB_CUT_CHECK_ERROR_GL ();
}

//-----------------------------------------------------------------------------
void NppGLWidgetPrivate::createTexture(int width, int height)
{
  this->makeCurrent();
  // create a tex as attachment
  glGenTextures(1, &tex_);
  glBindTexture(GL_TEXTURE_2D, tex_);

  // buffer data
  // texture width/height must be something like 2^m and >64x64;
//  int tex_width = pow(2, ceil(log(width)/log(2)));
//  int tex_height = pow(2, ceil(log(height)/log(2)));
//  if(tex_width < 64)
//    tex_width = 64;
//  if(tex_height < 64)
//    tex_height = 64;
//
//  printf("width=%d -> %d; height=%d -> %d\n", width, tex_width, height, tex_height);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB,
               GL_UNSIGNED_BYTE, NULL);

  // set basic parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  checkForOpenGLError(std::string("createTexture"));
//  VMLIB_CUT_CHECK_ERROR_GL();
}


//-----------------------------------------------------------------------------
void NppGLWidgetPrivate::deleteTexture()
{
  this->makeCurrent();
  if(tex_ == 0)
    return;

  glDisable(GL_TEXTURE_2D);
  glDeleteTextures(1, &tex_);
  tex_ = 0;

  checkForOpenGLError(std::string("deleteTexture"));
  // VMLIB_CUT_CHECK_ERROR_GL();
}


//-----------------------------------------------------------------------------
void NppGLWidgetPrivate::bind()
{
  this->makeCurrent();
  glBindBuffer(GL_ARRAY_BUFFER, pbo_);
  checkForOpenGLError(std::string("NppGLWidgetPrivate::bind()"));
}

//-----------------------------------------------------------------------------
void NppGLWidgetPrivate::unbind()
{
  this->makeCurrent();
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  checkForOpenGLError(std::string("NppGLWidgetPrivate::unbind()"));
}

//-----------------------------------------------------------------------------
void NppGLWidgetPrivate::paintGL()
{
  if (image_ == NULL) // no image available
    return;

  bool success = true;
  if (image_loaded_)   // always check if ready for processing
  {
    // attach to CUDA
    success &= (IU_SUCCESS == cuPboRegister(pbo_, registered_));

    // write intensity values to pbo
    success &= (IU_SUCCESS == cuGetOutput(pbo_, image_, minimum_value_,
                                           maximum_value_, image_size_));

//    // check all overlays if they should be painted
//    OverlayList::iterator it;
//    for ( it=overlay_list_.begin() ; it != overlay_list_.end(); it++ )
//    {
//      bool check_state = (*it)->isActive();
//      if(check_state)
//      {
//        QColor o_color = (*it)->getColor();
//        int r,g,b,a;
//        o_color.getRgb(&r, &g, &b, &a);
//
//        // distinguish between datatypes:
//        if( static_cast<Cuda::DeviceOverlay<float,2>* >(*it) != NULL )
//        {
//          Cuda::DeviceOverlay<float,2>* cur_overlay = static_cast<Cuda::DeviceOverlay<float,2>* >(*it);
//          success &= CudaNppGLWidgetPrivate::createOverlayF(
//              pbo_, cur_overlay->getOverlay(), r, g, b, a,
//              cur_overlay->getMaskValue(), image_size_);
//        }
//        else if( static_cast<Cuda::DeviceOverlay<unsigned char,2>* >(*it) != NULL )
//        {
//          Cuda::DeviceOverlay<unsigned char,2>* cur_overlay = static_cast<Cuda::DeviceOverlay<unsigned char,2>* >(*it);
//          success &= CudaNppGLWidgetPrivate::createOverlayUC(
//              pbo_, cur_overlay->getOverlay(), r, g, b, a,
//              cur_overlay->getMaskValue(), image_size_);
//        }
//        else
//        {
//          fprintf(stderr, "unknown datatype for overlay\n");
//        }
//      }
//    }


    // detach from CUDA
    success &= (IU_SUCCESS == cuPboUnregister(pbo_, registered_));

    IU_ASSERT(success == true);
    if(!success)
    {
//      qCritical() << "Error in NppGLWidgetPrivate::paintGL(): could not get current image data to display "
//          << this->windowTitle();
      image_loaded_ = false;
      emit closeWindow();
      return;
    }
  }
  checkForOpenGLError(std::string("paintGL"));

  this->renderScene();
}

//-----------------------------------------------------------------------------
void NppGLWidgetPrivate::renderScene()
{
  this->makeCurrent();
  // download texture from PBO
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_);
  glBindTexture(GL_TEXTURE_2D, tex_);
  glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, image_size_.width, image_size_.height,
                   GL_RGBA, GL_UNSIGNED_BYTE, NULL);

  // render a screen sized quad
  glDisable (GL_DEPTH_TEST);
  glDisable (GL_LIGHTING);
  glEnable  (GL_TEXTURE_2D);
  glTexEnvf (GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity ();
  glOrtho (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glViewport(0, 0, (int)floor(image_size_.width*zoom_), (int)floor(image_size_.height*zoom_));

  glBegin(GL_QUADS);

  glTexCoord2f( 0.0, 0.0);
  glVertex3f(-1.0, 1.0, 0.5);

  glTexCoord2f( 1.0, 0.0);
  glVertex3f(1.0, 1.0, 0.5);

  glTexCoord2f( 1.0, 1.0);
  glVertex3f(1.0,  -1.0, 0.5);

  glTexCoord2f( 0.0, 1.0);
  glVertex3f(-1.0,  -1.0, 0.5);

  glEnd ();

  glMatrixMode (GL_PROJECTION);
  glPopMatrix ();

  glDisable (GL_TEXTURE_2D);
  glBindBuffer (GL_PIXEL_UNPACK_BUFFER_ARB, 0);

  checkForOpenGLError(std::string("renderScene"));
}

//-----------------------------------------------------------------------------
void NppGLWidgetPrivate::resizeGL(int w, int h)
{
  if (!image_loaded_)   // always check if ready for processing
    return;

  this->makeCurrent();
  float zoom_w = float(w)/float(image_size_.width);
  float zoom_h = float(h)/float(image_size_.height);
  zoom_ = (zoom_w < zoom_h) ? zoom_w : zoom_h;

  checkForOpenGLError(std::string("resizeGL"));
}


//-----------------------------------------------------------------------------
void NppGLWidgetPrivate::createActions()
{
  // close event with ctrl-w
  action_close_ = new QAction(tr("Close Widget"), this);
  //action_close_->setShortcut(tr("Ctrl+w"));
  action_close_->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_W));
  connect(action_close_, SIGNAL(triggered()), this, SLOT(close()));
  this->addAction(action_close_);
  context_menu_->addAction(action_close_);

  // signal mapping for overlays
  overlay_signal_mapper_ = new QSignalMapper(this);
  connect(overlay_signal_mapper_, SIGNAL(mapped(const QString&)),
          this, SLOT(slotActivateOverlay(const QString&)));

  // zoom
  action_zoom_reset_ = new QAction(/*QIcon(":/images/new.png"),*/ tr("Zoom 1:1"), this);
  action_zoom_reset_->setShortcut(tr("Ctrl+0"));
  action_zoom_reset_->setStatusTip(tr("Reset zoom to original image size"));
  connect(action_zoom_reset_, SIGNAL(triggered()), this, SLOT(slotZoomReset()));
  this->addAction(action_zoom_reset_);
  context_menu_->addAction(action_zoom_reset_);

  context_menu_->addSeparator()->setText(tr("Grayvalue range:"));

  // slider label
  min_max_slider_label_ = new QLabel(this);
  min_max_slider_label_->setText(tr("0 - 1"));
  action_slider_range_label_ = new QWidgetAction(this);
  action_slider_range_label_->setDefaultWidget(min_max_slider_label_);
  context_menu_->addAction(action_slider_range_label_);

  // min slider
  slider_min_ = new QSlider(Qt::Horizontal, this);
  slider_min_->setMaximum(float_num_slider_values);
  slider_min_->setValue((int)(minimum_value_*float_num_slider_values));
  action_slider_min_ = new QWidgetAction(this);
  action_slider_min_->setDefaultWidget(slider_min_);
  connect(slider_min_, SIGNAL(valueChanged(int)), this, SLOT(slotMinimumValue(int)));
  context_menu_->addAction(action_slider_min_);

  // max slider
  slider_max_ = new QSlider(Qt::Horizontal, this);
  slider_max_->setMaximum(float_num_slider_values);
  slider_max_->setValue((int)(maximum_value_*float_num_slider_values));
  action_slider_max_ = new QWidgetAction(this);
  action_slider_max_->setDefaultWidget(slider_max_);
  connect(slider_max_, SIGNAL(valueChanged(int)), this, SLOT(slotMaximumValue(int)));
  context_menu_->addAction(action_slider_max_);

}

////-----------------------------------------------------------------------------
//void NppGLWidgetPrivate::addOverlay( ImageNpp_32f_C1* mask, QString name,
//                              bool active, QColor color, float mask_value)
//{
//  if(mask->region_size != image_size_)
//    qFatal("Size of rendered image and added mask do not match. Currently the region_size must be equal.");
//
//  Cuda::DeviceOverlay<float,2>* overlay = new Cuda::DeviceOverlay<float,2>(mask, name, color, active, mask_value);
//  overlay->datatype = Cuda::DeviceOverlayBase::FLOAT;
//  this->addOverlay(static_cast<Cuda::DeviceOverlayBase*>(overlay));
//}
//
////-----------------------------------------------------------------------------
//void NppGLWidgetPrivate::addOverlay( Cuda::DeviceMemory<unsigned char, 2>* mask, QString name,
//                              bool active, QColor color, unsigned char mask_value)
//{
//  if(mask->region_size != image_size_)
//    qFatal("Size of rendered image and added mask do not match. Currently the region_size must be equal.");
//
//  Cuda::DeviceOverlay<unsigned char,2>* overlay = new Cuda::DeviceOverlay<unsigned char,2>(
//      mask, name, color, active, mask_value);
//  overlay->datatype = Cuda::DeviceOverlayBase::UCHAR;
//  this->addOverlay(static_cast<Cuda::DeviceOverlayBase*>(overlay));
//}
//
////-----------------------------------------------------------------------------
//void NppGLWidgetPrivate::addOverlay(Cuda::DeviceOverlayBase* overlay)
//{
//  // if this is the first overlay -- add seperator in the context menu
//  if(overlay_list_.empty())
//    context_menu_->addSeparator()->setText(tr("Overlays:"));
//
//  overlay_list_.append(overlay);
//
//  QString name = overlay->getName(); //boost::get<2>(overlay_tuple);
//  bool active = overlay->isActive(); //boost::get<3>(overlay_tuple);
//
//  // create actions for context menu switches
//  QCheckBox *overlay_check_box = new QCheckBox(name, this);
//  overlay_check_box->setChecked(active);
//  QWidgetAction *action_overlay_widget = new QWidgetAction(this);
//  action_overlay_widget->setDefaultWidget(overlay_check_box);
//  action_list_overlays_.append(action_overlay_widget);
//  connect(overlay_check_box, SIGNAL(toggled(bool)), overlay_signal_mapper_, SLOT(map()));
//  overlay_signal_mapper_->setMapping(overlay_check_box, name);
//  context_menu_->addAction(action_overlay_widget);
//}



//-----------------------------------------------------------------------------
void NppGLWidgetPrivate::setGrayValueRange(float minimum, float maximum,
                                    float true_minimum, float true_maximum)
{
  minimum_ = minimum;
  maximum_ = maximum;
  true_minimum_ = true_minimum;
  true_maximum_ = true_maximum;

  // Update minimum slider
  slider_min_->setMinimum(minimum_*float_num_slider_values);
  slider_min_->setMaximum(maximum_*float_num_slider_values);
  slider_min_->setValue((int)(minimum_*float_num_slider_values));

  // Update maximum slider
  slider_max_->setMinimum(minimum_*float_num_slider_values);
  slider_max_->setMaximum(maximum_*float_num_slider_values);
  slider_max_->setValue((int)(maximum_*float_num_slider_values));
}

//-----------------------------------------------------------------------------
void NppGLWidgetPrivate::slotTimerCallback()
{
  updateGL();
  QCoreApplication::processEvents();
}

//-----------------------------------------------------------------------------
void NppGLWidgetPrivate::slotZoomReset()
{
  zoom_ = 1.0f;
  this->resize(image_size_.width, image_size_.height);
}

//-----------------------------------------------------------------------------
void NppGLWidgetPrivate::slotMinimumValue(int val)
{
  minimum_value_ = (float)val/(float)float_num_slider_values;

  // Update true value label
  float true_minimum_value = (minimum_value_-minimum_)/(maximum_-minimum_)*(true_maximum_-true_minimum_)+true_minimum_;
  float true_maximum_value = (maximum_value_-minimum_)/(maximum_-minimum_)*(true_maximum_-true_minimum_)+true_minimum_;
  min_max_slider_label_->setText(tr("%1 - %2").arg(true_minimum_value).arg(true_maximum_value));
}

//-----------------------------------------------------------------------------
void NppGLWidgetPrivate::slotMaximumValue(int val)
{
  maximum_value_ = (float)val/(float)float_num_slider_values;

  // Update true value label
  float true_minimum_value = (minimum_value_-minimum_)/(maximum_-minimum_)*(true_maximum_-true_minimum_)+true_minimum_;
  float true_maximum_value = (maximum_value_-minimum_)/(maximum_-minimum_)*(true_maximum_-true_minimum_)+true_minimum_;
  min_max_slider_label_->setText(tr("%1 - %2").arg(true_minimum_value).arg(true_maximum_value));
}

//-----------------------------------------------------------------------------
void NppGLWidgetPrivate::slotActivateOverlay(const QString& overlay_name)
{
//  // search for the signal sender in the overlay list
//  OverlayList::iterator it;
//  for ( it=overlay_list_.begin() ; it != overlay_list_.end(); it++ )
//  {
//    if(overlay_name == (*it)->getName())
//    {
//      // toggle overlay state if the sender was found
//      (*it)->toggleActive();
//    }
//  }
}

//-----------------------------------------------------------------------------
void NppGLWidgetPrivate::updateImage(iu::Image* image)
{
  this->setImage(image);
}

//-----------------------------------------------------------------------------
void NppGLWidgetPrivate::setImage(iu::Image* image)
{
  if (autoupdate_)
    timer_->stop();
  Q_ASSERT(image);
  this->makeCurrent();
  image_ = image;

  image_size_ = image_->size();
  if(!image_loaded_)
  {
    // set image and window sizes
    this->resize(image_size_.width, image_size_.height);
    this->createPbo(image_size_.width, image_size_.height);
    this->createTexture(image_size_.width, image_size_.height);
  }
  else // if an image was loaded beforehand the pbo and texture has to be removed.
  {
    this->deletePbo();
    this->deleteTexture();
    if (autoresize_)
      this->resize(image_size_.width, image_size_.height);
    this->createPbo(image_size_.width, image_size_.height);
    this->createTexture(image_size_.width, image_size_.height);
  }

  // start painting
  if (autoupdate_)
    timer_->start(1000.0f/fps_);
    //timer_->start(5);
  else
    updateGL();

  image_loaded_ = true;

  checkForOpenGLError(std::string("updateImage"));
  // VMLIB_CUT_CHECK_ERROR_GL();
}

//-----------------------------------------------------------------------------
void NppGLWidgetPrivate::setFps(float fps)
{
  if (fps_<0.0f)
    return;
  fps_ = fps;
  if (autoupdate_)
  {
    timer_->stop();
    timer_->start(1000.0f/fps_);
  }
}

//-----------------------------------------------------------------------------
void NppGLWidgetPrivate::mousePressEvent(QMouseEvent * event)
{
  if(!image_loaded_)
    return;

  if (event->button() == Qt::LeftButton)
  {
    // Save current Position
    int click_pos_y = static_cast<int>(this->height()/zoom_ + 0.5f) - image_size_.height;
    int offset_y = (click_pos_y > 0) ? click_pos_y : 0;
    mouse_x_old_ = floor(event->x()/zoom_);
    mouse_y_old_ = floor(event->y()/zoom_) - offset_y;

    emit mousePressed(mouse_x_old_, mouse_y_old_);
    emit mousePressed(mouse_x_old_, mouse_y_old_, event->globalX(), event->globalY());
  }
}

//-----------------------------------------------------------------------------
void NppGLWidgetPrivate::mouseReleaseEvent(QMouseEvent * event)
{
  int click_pos_y = static_cast<int>(this->height()/zoom_ + 0.5f) - image_size_.height;
  int offset_y = (click_pos_y > 0) ? click_pos_y : 0;
  emit  mouseReleased(floor(event->x()/zoom_), floor(event->y()/zoom_) - offset_y);
}

//-----------------------------------------------------------------------------
void NppGLWidgetPrivate::mouseMoveEvent(QMouseEvent * event)
{
  if(!image_loaded_)
    return;

  int click_pos_y = static_cast<int>(this->height()/zoom_ + 0.5f) - image_size_.height;
  int offset_y = (click_pos_y > 0) ? click_pos_y : 0;
  mouse_x_ = floor(event->x()/zoom_);
  mouse_y_ = floor(event->y()/zoom_) - offset_y;
  emit mouseMoved(mouse_x_old_, mouse_y_old_, mouse_x_, mouse_y_);
  mouse_x_old_ = mouse_x_;
  mouse_y_old_ = mouse_y_;

}

//-----------------------------------------------------------------------------
void NppGLWidgetPrivate::contextMenuEvent(QContextMenuEvent* event)
{
  context_menu_->exec(event->globalPos());
}

//--------------------------------------------------------------------------------
void NppGLWidgetPrivate::wheelEvent(QWheelEvent *event)
{
  int num_degrees = event->delta() / 8;
  int num_steps = num_degrees / 15;

  if (event->orientation() == Qt::Vertical && QApplication::keyboardModifiers() == Qt::ControlModifier)
  {
    float cur_zoom = zoom_ + float(num_steps)/30.0f;
    this->resize(image_size_.width*cur_zoom, image_size_.height*cur_zoom);
    event->accept();
  }
  else
    event->ignore();
}


} // namespace iuprivate


/* **************************************************************************
 *    PUBLIC INTERFACE IMPLEMENTATION
 * **************************************************************************/

namespace iu {

NppGLWidget::NppGLWidget(QWidget* parent, iu::Image* image,
                         bool autoupdate, bool autoresize) :
iuprivate::NppGLWidgetPrivate(parent, image, autoupdate, autoresize)
{
}

NppGLWidget::~NppGLWidget()
{
}

void NppGLWidget::setAutoresize(bool val) { NppGLWidgetPrivate::setAutoresize(val); }

void NppGLWidget::setFps(float fps) { NppGLWidgetPrivate::setFps(fps); }

void NppGLWidget::setGrayValueRange(float minimum, float maximum,
                                    float true_minimum, float true_maximum)
{
  NppGLWidgetPrivate::setGrayValueRange(minimum, maximum, true_minimum, true_maximum);
}

void NppGLWidget::updateImage(iu::Image *image) { NppGLWidgetPrivate::updateImage(image); }

void NppGLWidget::setImage(iu::Image* image){ NppGLWidgetPrivate::setImage(image); }
void NppGLWidget::init(){ NppGLWidgetPrivate::init(); }
void NppGLWidget::createActions(){ NppGLWidgetPrivate::createActions(); }
void NppGLWidget::closeEvent(QCloseEvent *event){ NppGLWidgetPrivate::closeEvent(event); }

void NppGLWidget::initializeGL(){ NppGLWidgetPrivate::initializeGL(); }
void NppGLWidget::createPbo(int width, int height){ NppGLWidgetPrivate::createPbo(width, height); }
void NppGLWidget::deletePbo(){ NppGLWidgetPrivate::deletePbo(); }
void NppGLWidget::createTexture(int width, int height){ NppGLWidgetPrivate::createTexture(width, height); }
void NppGLWidget::deleteTexture(){ NppGLWidgetPrivate::deleteTexture(); }
void NppGLWidget::bind(){ NppGLWidgetPrivate::bind(); }
void NppGLWidget::unbind(){ NppGLWidgetPrivate::unbind(); }
void NppGLWidget::paintGL(){ NppGLWidgetPrivate::paintGL(); }
void NppGLWidget::renderScene(){ NppGLWidgetPrivate::renderScene(); }
void NppGLWidget::resizeGL(int w, int h){ NppGLWidgetPrivate::resizeGL(w,h); }

void NppGLWidget::mousePressEvent(QMouseEvent *event){ NppGLWidgetPrivate::mousePressEvent(event); }
void NppGLWidget::mouseReleaseEvent(QMouseEvent *event){ NppGLWidgetPrivate::mouseReleaseEvent(event); }
void NppGLWidget::mouseMoveEvent(QMouseEvent *event){ NppGLWidgetPrivate::mouseMoveEvent(event); }

void NppGLWidget::contextMenuEvent(QContextMenuEvent* event){ NppGLWidgetPrivate::contextMenuEvent(event); }
void NppGLWidget::wheelEvent(QWheelEvent *event){ NppGLWidgetPrivate::wheelEvent(event); }
void NppGLWidget::checkForOpenGLError(const std::string& message){ NppGLWidgetPrivate::checkForOpenGLError(message); }


} // namespace iu

