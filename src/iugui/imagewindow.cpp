#include <QVBoxLayout>
#include <QMouseEvent>
#include <QScrollArea>

#include "qglimagegpuwidget.h"
#include "imagewindow.h"

namespace iu {

//-----------------------------------------------------------------------------
ImageWindow::ImageWindow(QWidget *parent) :
    QWidget(parent)
{
  // setup basic qglwidget
  image_gpu_widget_ = new iu::QGLImageGpuWidget;
  image_gpu_widget_->setMinimumSize(200,200);

  connect(image_gpu_widget_, SIGNAL(mousePressed(int,int)), this, SIGNAL(mousePressed(int,int)));
  connect(image_gpu_widget_, SIGNAL(mousePressed(int,int,int,int)), this, SIGNAL(mousePressed(int,int,int,int)));
  connect(image_gpu_widget_, SIGNAL(mouseMoved(int,int,int,int)), this, SIGNAL(mouseMoved(int,int,int,int)));
  connect(image_gpu_widget_, SIGNAL(mouseReleased(int,int)), this, SIGNAL(mouseReleased(int,int)));

  // just to test
  connect(this, SIGNAL(mousePressed(int,int,int,int)), this, SLOT(mouseGotPressed(int,int,int,int)));

  // setup layout
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  // TODO: add things to layout...

  image_gpu_scroll_area_ = new QScrollArea(this);
  image_gpu_scroll_area_->setWidget(image_gpu_widget_);
  image_gpu_scroll_area_->setAlignment(Qt::AlignCenter);
  image_gpu_scroll_area_->setFrameStyle(QFrame::NoFrame);
  image_gpu_scroll_area_->setMinimumSize(320, 240);
  //image_gpu_scroll_area_->setBackgroundRole(QPalette::Dark);
  //image_gpu_scroll_area_->setAutoFillBackground(true);
  main_layout->addWidget(image_gpu_scroll_area_, 1);

  this->setLayout(main_layout);
}

//-----------------------------------------------------------------------------
void ImageWindow::setImage(iu::ImageGpu_8u_C1* image, bool normalize)
{
  image_gpu_widget_->setImage(image, normalize);
  this->setupGeometry();
}

//-----------------------------------------------------------------------------
void ImageWindow::setImage(iu::ImageGpu_32f_C1* image, bool normalize)
{
  image_gpu_widget_->setImage(image, normalize);
  this->setupGeometry();
}

//-----------------------------------------------------------------------------
void ImageWindow::setImage(iu::ImageGpu_8u_C4* image, bool normalize)
{
  image_gpu_widget_->setImage(image, normalize);
  this->setupGeometry();
}

//-----------------------------------------------------------------------------
void ImageWindow::setImage(iu::ImageGpu_32f_C4* image, bool normalize)
{
  image_gpu_widget_->setImage(image, normalize);
  this->setupGeometry();
}

//-----------------------------------------------------------------------------
void ImageWindow::update()
{
  image_gpu_widget_->update();
}

//-----------------------------------------------------------------------------
void ImageWindow::setupGeometry()
{

  this->setGeometry(0,0,image_gpu_widget_->width()+15, image_gpu_widget_->height()+15);

}

//-----------------------------------------------------------------------------
void ImageWindow::addOverlay(QString name, iu::Image* constraint_image,
                             iu::LinearMemory* lut_values, iu::LinearDeviceMemory_8u_C4* lut_colors,
                             bool active)
{
  image_gpu_widget_->addOverlay(name, constraint_image, lut_values, lut_colors, active);
}


/* ****************************************************************************
     interactive
 * ***************************************************************************/

//-----------------------------------------------------------------------------
void ImageWindow::mouseGotPressed(int x, int y, int global_x, int global_y)
{
  printf("ImageWindow: mouse pressed %d/%d, %d/%d\n", x, y, global_x, global_y);
}


} // namespace iu


