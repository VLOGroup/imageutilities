#include <QVBoxLayout>
#include <QMouseEvent>
#include <QScrollArea>
#include <QScrollBar>
#include <QToolBar>
#include <QAction>
#include <QLabel>
#include <QResource>

#include "qglimagegpuwidget.h"
#include "imagewindow.h"

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)


#define MINWINSIZE 50

extern int qInitResources_images();

namespace iu {

//-----------------------------------------------------------------------------
ImageWindow::ImageWindow(QWidget *parent) :
    QWidget(parent)
{
// bool worked = QResource::registerResource("../../lib/images.rcc");
// printf("QResource worked = %d\n", worked);

//#undef QT_NAMESPACE
//  Q_INIT_RESOURCE(images);
//#define QT_NAMESPACE

  printf("------------------------\n");

  fprintf (stderr, TOSTRING(Q_INIT_RESOURCE(images)) "\n");

  printf("------------------------\n");

  int val = qInitResources_images();

  printf("val = %d\n", val);

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


  // setup layout
  QVBoxLayout *main_layout = new QVBoxLayout(this);

  // Toolbar
  tool_bar_ = new QToolBar(this);

  action_save_ = new QAction(QIcon(":disk"), "Save image", this);
  tool_bar_->addAction(action_save_);


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
void ImageWindow::update()
{
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
                             bool active)
{
  image_gpu_widget_->addOverlay(name, constraint_image, lut_values, lut_colors, active);
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



