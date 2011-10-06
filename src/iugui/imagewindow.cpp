#include <QVBoxLayout>
#include <QMouseEvent>
#include <QScrollArea>
#include <QScrollBar>
#include <QToolBar>
#include <QAction>
#include <QLabel>
#include <QResource>
#include <QSpinBox>
#include <QFileDialog>

#include "qglimagegpuwidget.h"
#include "imagewindow.h"
#include "iucore.h"
#include <iuio.h>

#define MINWINSIZE 50

extern int qInitResources_iu_gui_images();

namespace iu {

//-----------------------------------------------------------------------------
ImageWindow::ImageWindow(QWidget *parent) :
  QWidget(parent)
{
  // Load resources
  qInitResources_iu_gui_images();

  // setup basic qglwidget
  image_gpu_widget_ = new iu::QGLImageGpuWidget;

  connect(image_gpu_widget_, SIGNAL(mousePressed(int,int)), this, SIGNAL(mousePressed(int,int)));
  connect(image_gpu_widget_, SIGNAL(mousePressed(int,int,int,int)), this, SIGNAL(mousePressed(int,int,int,int)));
  connect(image_gpu_widget_, SIGNAL(mouseMoved(int,int,int,int)), this, SIGNAL(mouseMoved(int,int,int,int)));
  connect(image_gpu_widget_, SIGNAL(mouseReleased(int,int)), this, SIGNAL(mouseReleased(int,int)));
  connect(image_gpu_widget_, SIGNAL(zoomChanged(float)), this, SIGNAL(zoomChanged(float)));

  connect(image_gpu_widget_, SIGNAL(pan(int,int,int,int)), this, SLOT(pan(int,int,int,int)));
  connect(image_gpu_widget_, SIGNAL(zoomed(double)), this, SLOT(zoomed(double)));
  connect(image_gpu_widget_, SIGNAL(showToolbar(bool)), this, SLOT(showToolbar(bool)));
  connect(image_gpu_widget_, SIGNAL(pixelInfo(QString)), this, SLOT(updatePixelInfo(QString)));

  volume_ = NULL;
  image_ = NULL;

  file_prefix_ = "display_output";

  // setup layout
  QVBoxLayout *main_layout = new QVBoxLayout(this);

  // Toolbar
  tool_bar_ = new QToolBar(this);
  tool_bar_->setIconSize(QSize(21,21));

  action_save_ = new QAction(QIcon(":disk"), "Save image", this);
  tool_bar_->addAction(action_save_);
  connect(action_save_, SIGNAL(triggered()), this, SLOT(on_action_save__triggered()));

  slice_selector_ = new QSpinBox(this);
  slice_selector_->setToolTip("Slice selector");
  connect(slice_selector_, SIGNAL(valueChanged(int)), this, SLOT(sliceSelect(int)));
  slice_action_ = tool_bar_->addWidget(slice_selector_);
  slice_action_->setVisible(false);

  pixel_info_ = new QLabel("Pixel info", this);
  tool_bar_->addWidget(pixel_info_);

  main_layout->addWidget(tool_bar_);

  // Scroll Area
  image_gpu_scroll_area_ = new QScrollArea(this);
  image_gpu_scroll_area_->setWidget(image_gpu_widget_);
  image_gpu_scroll_area_->setAlignment(Qt::AlignCenter);
  image_gpu_scroll_area_->setFrameStyle(QFrame::NoFrame);
  main_layout->addWidget(image_gpu_scroll_area_, 1);

  this->setLayout(main_layout);
}

//-----------------------------------------------------------------------------
void ImageWindow::showToolbar(bool val)
{
  tool_bar_->setVisible(val);
}

//-----------------------------------------------------------------------------
void ImageWindow::updatePixelInfo(QString text)
{
  pixel_info_->setText(text);
}

//-----------------------------------------------------------------------------
void ImageWindow::on_action_save__triggered()
{
  QString fileName = file_prefix_;
  if (volume_)
    fileName.append(QString("_slice%1").arg(slice_selector_->value()));

  fileName.append(".png");
  fileName = QFileDialog::getSaveFileName(this, tr("Save current view to file"),fileName, tr("Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.pgm  *.pnm)"));

  if (fileName == "")
    return;

  // save image
  iu::ImageGpu_8u_C4 temp_out(image_gpu_widget_->imageWidth(), image_gpu_widget_->imageHeight());
  image_gpu_widget_->getPboOutput(&temp_out);
  iu::imsave(&temp_out, fileName.toStdString());
}

//-----------------------------------------------------------------------------
void ImageWindow::setImage(iu::ImageGpu_8u_C1* image, bool normalize)
{
  image_gpu_widget_->setImage(image, normalize);

  float minsize = IUMIN(image->width(), image->height());
  float aspectX = image->width()/minsize;
  float aspectY = image->height()/minsize;
  int sizex = MINWINSIZE*aspectX;
  int sizey = MINWINSIZE*aspectY;
  image_gpu_widget_->setMinimumSize(sizex, sizey);

  this->setupGeometry();
}

//-----------------------------------------------------------------------------
void ImageWindow::setImage(iu::ImageGpu_32f_C1* image, bool normalize)
{
  image_gpu_widget_->setImage(image, normalize);

  float minsize = IUMIN(image->width(), image->height());
  float aspectX = image->width()/minsize;
  float aspectY = image->height()/minsize;
  int sizex = MINWINSIZE*aspectX;
  int sizey = MINWINSIZE*aspectY;
  image_gpu_widget_->setMinimumSize(sizex, sizey);

  this->setupGeometry();
}

//-----------------------------------------------------------------------------
void ImageWindow::setImage(iu::ImageGpu_8u_C4* image, bool normalize)
{
  image_gpu_widget_->setImage(image, normalize);

  float minsize = IUMIN(image->width(), image->height());
  float aspectX = image->width()/minsize;
  float aspectY = image->height()/minsize;
  int sizex = MINWINSIZE*aspectX;
  int sizey = MINWINSIZE*aspectY;
  image_gpu_widget_->setMinimumSize(sizex, sizey);

  this->setupGeometry();
}

//-----------------------------------------------------------------------------
void ImageWindow::setImage(iu::ImageGpu_32f_C4* image, bool normalize)
{
  slice_selector_->hide();
  image_gpu_widget_->setImage(image, normalize);

  float minsize = IUMIN(image->width(), image->height());
  float aspectX = image->width()/minsize;
  float aspectY = image->height()/minsize;
  int sizex = MINWINSIZE*aspectX;
  int sizey = MINWINSIZE*aspectY;
  image_gpu_widget_->setMinimumSize(sizex, sizey);

  this->setupGeometry();
}

//-----------------------------------------------------------------------------
void ImageWindow::setVolume(iu::VolumeGpu_32f_C1* volume, bool normalize)
{
  volume_ = volume;
  slice_action_->setVisible(true);
  slice_selector_->setMinimum(0);
  slice_selector_->setMaximum(volume_->roi().depth-1);

  image_ = new iu::ImageGpu_32f_C1(volume_->data(slice_selector_->value()*volume_->slice_stride()),
                                   volume_->width(), volume_->height(), volume_->pitch(), true);

  image_gpu_widget_->setImage(image_, normalize);

  float minsize = IUMIN(image_->width(), image_->height());
  float aspectX = image_->width()/minsize;
  float aspectY = image_->height()/minsize;
  int sizex = MINWINSIZE*aspectX;
  int sizey = MINWINSIZE*aspectY;
  image_gpu_widget_->setMinimumSize(sizex, sizey);

  this->setupGeometry();
}

//-----------------------------------------------------------------------------
void  ImageWindow::sliceSelect(int val)
{
  if (image_)
    delete image_;

  image_ = new iu::ImageGpu_32f_C1(volume_->data(val*volume_->slice_stride()),
                                   volume_->width(), volume_->height(), volume_->pitch(), true);
}

//-----------------------------------------------------------------------------
void ImageWindow::update()
{
  if (volume_)
  {
    slice_selector_->setMaximum(volume_->roi().depth-1);
  }

  image_gpu_widget_->update();
}

//-----------------------------------------------------------------------------
void ImageWindow::setCursor(QCursor csr)
{
  image_gpu_widget_->setCursor(csr);
}

//-----------------------------------------------------------------------------
void ImageWindow::setupGeometry()
{
  this->setGeometry(0,0,image_gpu_widget_->width()+15, image_gpu_widget_->height()+15);
}

//-----------------------------------------------------------------------------
void ImageWindow::addOverlay(QString name, iu::Image* constraint_image,
                             iu::LinearMemory* lut_values, iu::LinearDeviceMemory_8u_C4* lut_colors,
                             bool active, IuComparisonOperator comp_op)
{
  image_gpu_widget_->addOverlay(name, constraint_image, lut_values, lut_colors, active, comp_op);
}


/* ****************************************************************************
     interactive
 * ***************************************************************************/

//-----------------------------------------------------------------------------
void ImageWindow::pan(int from_x, int from_y, int to_x, int to_y)
{
  int dx = to_x-from_x;
  int dy = to_y-from_y;

  panScrollBar(image_gpu_scroll_area_->horizontalScrollBar(), dx);
  panScrollBar(image_gpu_scroll_area_->verticalScrollBar(), dy);
}

//-----------------------------------------------------------------------------
void ImageWindow::zoomed(double factor)
{
  adjustScrollBar(image_gpu_scroll_area_->horizontalScrollBar(), factor);
  adjustScrollBar(image_gpu_scroll_area_->verticalScrollBar(), factor);
}


//-----------------------------------------------------------------------------
void ImageWindow::panScrollBar(QScrollBar *scrollBar, int offset)
{
  scrollBar->setValue(int(scrollBar->value() - offset));
}

//-----------------------------------------------------------------------------
void ImageWindow::adjustScrollBar(QScrollBar *scrollBar, double factor)
{
  scrollBar->setValue(int(factor * scrollBar->value()
                          + ((factor - 1) * scrollBar->pageStep()/2)));
}

} // namespace iu



