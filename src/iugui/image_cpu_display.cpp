#include "image_cpu_display.h"
#include "image_cpu_display_p.h"
#include <qlayout.h>
#include <stdexcept>

namespace iuprivate {

//-----------------------------------------------------------------------------
/* 8-bit; 1-channel */
QImageCpuDisplay::QImageCpuDisplay(iu::ImageCpu_8u_C1* image, const std::string& title,
                                   unsigned char minval, unsigned char maxval) :
  base_image_(0), final_image_(0),
  minval_(minval), maxval_(maxval), size_(image->size()),
  context_menu_(0), overlay_signal_mapper_(0)
{
  // create internal images
  base_image_ = new QImage(size_.width, size_.height, QImage::Format_RGB888);
  final_image_ = new QImage(size_.width, size_.height, QImage::Format_RGB888);

  this->updateImage(image, minval, maxval);

  setWindowTitle(QString::fromStdString(title));

  this->init();
}

//-----------------------------------------------------------------------------
/* 8-bit; 4-channel */
QImageCpuDisplay::QImageCpuDisplay(iu::ImageCpu_8u_C4* image, const std::string& title,
                                   unsigned char minval, unsigned char maxval) :
  base_image_(0), final_image_(0),
  minval_(minval), maxval_(maxval), size_(image->size()),
  context_menu_(0), overlay_signal_mapper_(0)
{
  // create internal images
  base_image_ = new QImage(size_.width, size_.height, QImage::Format_RGB888);
  final_image_ = new QImage(size_.width, size_.height, QImage::Format_RGB888);

  this->updateImage(image, minval, maxval);

  setWindowTitle(QString::fromStdString(title));

  this->init();
}

//-----------------------------------------------------------------------------
/* 32-bit; 1-channel */
QImageCpuDisplay::QImageCpuDisplay(iu::ImageCpu_32f_C1* image, const std::string& title,
                                   float minval, float maxval) :
  base_image_(0), final_image_(0),
  minval_(minval), maxval_(maxval), size_(image->size()),
  context_menu_(0), overlay_signal_mapper_(0)
{
  // create internal images
  base_image_ = new QImage(size_.width, size_.height, QImage::Format_RGB888);
  final_image_ = new QImage(size_.width, size_.height, QImage::Format_RGB888);

  this->updateImage(image, minval, maxval);

  setWindowTitle(QString::fromStdString(title));

  this->init();
}

//-----------------------------------------------------------------------------
/* 32-bit; 4-channel */
QImageCpuDisplay::QImageCpuDisplay(iu::ImageCpu_32f_C4* image, const std::string& title,
                                   float minval, float maxval) :
  base_image_(0), final_image_(0),
  minval_(minval), maxval_(maxval), size_(image->size()),
  context_menu_(0), overlay_signal_mapper_(0)
{
  // create internal images
  base_image_ = new QImage(size_.width, size_.height, QImage::Format_RGB888);
  final_image_ = new QImage(size_.width, size_.height, QImage::Format_RGB888);

  this->updateImage(image, minval, maxval);

  setWindowTitle(QString::fromStdString(title));
  this->init();
}


//-----------------------------------------------------------------------------
/* 8-bit; 1-channel */
void QImageCpuDisplay::updateImage(iu::ImageCpu_8u_C1* image,
                                   unsigned char minval, unsigned char maxval)
{
  minval_ = minval;
  maxval_ = maxval;

  if(size_ != image->size())
  {
    delete base_image_;
    delete final_image_;
    size_ = image->size();
    base_image_ = new QImage(size_.width, size_.height, QImage::Format_RGB888);
    final_image_ = new QImage(size_.width, size_.height, QImage::Format_RGB888);
  }

  register int r = 0;
  register unsigned int ry, rx;
  for (ry = 0; ry < size_.height; ry++)
  {
    for (rx = 0; rx < size_.width; rx++)
    {
      r = image->data(rx,ry)[0];
      base_image_->setPixel(rx, ry, qRgb(r,r,r));
    }
  }

  composeAndShow();
  adjustSize();
}

//-----------------------------------------------------------------------------
/* 8-bit; 4-channel */
void QImageCpuDisplay::updateImage(iu::ImageCpu_8u_C4* image,
                                   unsigned char minval, unsigned char maxval)
{
  minval_ = minval;
  maxval_ = maxval;

  if(size_ != image->size())
  {
    delete base_image_;
    delete final_image_;
    size_ = image->size();
    base_image_ = new QImage(size_.width, size_.height, QImage::Format_RGB888);
    final_image_ = new QImage(size_.width, size_.height, QImage::Format_RGB888);
  }

  register int r = 0, g = 0, b = 0;
  register unsigned int ry, rx;
  for (ry = 0; ry < size_.height; ry++)
  {
    for (rx = 0; rx < size_.width; rx++)
    {
      r = image->data(rx,ry)->x;
      g = image->data(rx,ry)->y;
      b = image->data(rx,ry)->z;
      base_image_->setPixel(rx, ry, qRgb(r,g,b));
    }
  }

  composeAndShow();
  adjustSize();
}

//-----------------------------------------------------------------------------
/* 32-bit; 1-channel */
void QImageCpuDisplay::updateImage(iu::ImageCpu_32f_C1* image,
                              float minval, float maxval)
{
  minval_ = minval;
  maxval_ = maxval;

  if(size_ != image->size())
  {
    delete base_image_;
    delete final_image_;
    size_ = image->size();
    base_image_ = new QImage(size_.width, size_.height, QImage::Format_RGB888);
    final_image_ = new QImage(size_.width, size_.height, QImage::Format_RGB888);
  }

  register int r = 0;
  register unsigned int ry, rx;
  for (ry = 0; ry < size_.height; ry++)
  {
    for (rx = 0; rx < size_.width; rx++)
    {
      r = scale(image->data(rx,ry)[0]);
      base_image_->setPixel(rx, ry, qRgb(r,r,r));
    }
  }

  composeAndShow();
  adjustSize();
}

//-----------------------------------------------------------------------------
/* 32-bit; 4-channel */
void QImageCpuDisplay::updateImage(iu::ImageCpu_32f_C4* image,
                                   float minval, float maxval)
{
  minval_ = minval;
  maxval_ = maxval;

  if(size_ != image->size())
  {
    delete base_image_;
    delete final_image_;
    size_ = image->size();
    base_image_ = new QImage(size_.width, size_.height, QImage::Format_RGB888);
    final_image_ = new QImage(size_.width, size_.height, QImage::Format_RGB888);
  }

  register int r = 0, g = 0, b = 0;
  register unsigned int ry, rx;
  for (ry = 0; ry < size_.height; ry++)
  {
    for (rx = 0; rx < size_.width; rx++)
    {
      r = scale(image->data(rx,ry)->x);
      g = scale(image->data(rx,ry)->y);
      b = scale(image->data(rx,ry)->z);
      base_image_->setPixel(rx, ry, qRgb(r,g,b));
    }
  }

  composeAndShow();
  adjustSize();
}

//-----------------------------------------------------------------------------
QImageCpuDisplay::~QImageCpuDisplay()
{
  deleteOverlays();
  delete base_image_;
  base_image_ = 0;
  delete final_image_;
  final_image_ = 0;
}

//-----------------------------------------------------------------------------
void QImageCpuDisplay::init()
{
  setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
  setScaledContents(true);

  context_menu_ = new QMenu("Layers", this);

  // signal mapping for overlays
  overlay_signal_mapper_ = new QSignalMapper(this);
  connect(overlay_signal_mapper_, SIGNAL(mapped(const QString&)),
          this, SLOT(slotActivateOverlay(const QString&)));
}

//-----------------------------------------------------------------------------
void QImageCpuDisplay::addOverlay(const std::string& title, const float* data)
{
  //  Overlay my_overlay;
  //  my_overlay.title = QString::fromStdString(title);
  //  my_overlay.show = true;
  //  my_overlay.image = new QImage(size_.width, size_.height, QImage::Format_ARGB32);
  //  for (size_t py = 0; py < size_.height; py++){
  //    for (size_t px = 0; px < size_.width; px++){
  //      int r = (int) (data[0 * stride_ + py * pitch_ + px] * 255.0f);
  //      int g = (int) (data[1 * stride_ + py * pitch_ + px] * 255.0f);
  //      int b = (int) (data[2 * stride_ + py * pitch_ + px] * 255.0f);
  //      int a = (int) (data[3 * stride_ + py * pitch_ + px] * 255.0f);
  //      my_overlay.image->setPixel(px, py, qRgba(r,g,b,a));
  //    }
  //  }
  //  overlays_.push_back(my_overlay);
  //  addContextMenuEntry(my_overlay.title);
  //  composeAndShow();
}

//-----------------------------------------------------------------------------
void QImageCpuDisplay::addOverlay(const std::string& title, const float* data, float alpha)
{
  //  Overlay my_overlay;
  //  my_overlay.title = QString::fromStdString(title);
  //  my_overlay.show = true;
  //  my_overlay.image = new QImage(size_.width, size_.height, QImage::Format_ARGB32);
  //  int a = (int) alpha * 255.0f;
  //  for (size_t py = 0; py < size_.height; py++){
  //    for (size_t px = 0; px < size_.width; px++){
  //      int r = (int) (data[0 * stride_ + py * pitch_ + px] * 255.0f);
  //      int g = (int) (data[1 * stride_ + py * pitch_ + px] * 255.0f);
  //      int b = (int) (data[2 * stride_ + py * pitch_ + px] * 255.0f);
  //      my_overlay.image->setPixel(px, py, qRgba(r,g,b,a));
  //    }
  //  }
  //  overlays_.push_back(my_overlay);
  //  addContextMenuEntry(my_overlay.title);
  //  composeAndShow();
}

//-----------------------------------------------------------------------------
void QImageCpuDisplay::deleteOverlays()
{
  for (std::vector<Overlay>::iterator it = overlays_.begin(); it != overlays_.end(); ++it){
    delete it->image;
  }
  overlays_.clear();
  context_menu_->clear();
}

//-----------------------------------------------------------------------------
void QImageCpuDisplay::copyCurrentImage(float *data, size_t width, size_t height, size_t depth)
{
  if (width == size_.width && height == size_.height && depth == 3){
    for (size_t py = 0; py < size_.height; py++){
      for (size_t px = 0; px < size_.width; px++){
        QRgb rgb = final_image_->pixel(px, py);
        data[0 * width * height + py * width + px] = qRed(rgb) / 255.0f;
        data[1 * width * height + py * width + px] = qGreen(rgb) / 255.0f;
        data[2 * width * height + py * width + px] = qBlue(rgb) / 255.0f;
      }
    }
  }
  else
  {
    memset(data, 0, width * height * depth * sizeof(float));
    //throw(std::runtime_error(std::string("dimensions do not fit")));
    fprintf(stderr, "QImageCpuDisplay::copyCurrentImage: dimensions do not fit\n");
  }
}

//-----------------------------------------------------------------------------
void QImageCpuDisplay::composeAndShow()
{
  *final_image_ = base_image_->copy(0,0,size_.width, size_.height);
  for (std::vector<Overlay>::iterator it = overlays_.begin(); it != overlays_.end(); ++it)
  {
    if (it->show)
    {
      for (size_t py = 0; py < size_.height; py++)
      {
        for (size_t px = 0; px < size_.width; px++)
        {
          QRgb rgb_old = final_image_->pixel(px, py);
          QRgb rgb_ovr = it->image->pixel(px, py);
          float a = qAlpha(rgb_ovr) / 255.0f;
          QRgb rgb_new = qRgb((int)((1 - a) * qRed(rgb_old) + a * qRed(rgb_ovr)),
                              (int)((1 - a) * qGreen(rgb_old) + a * qGreen(rgb_ovr)),
                              (int)((1 - a) * qBlue(rgb_old) + a * qBlue(rgb_ovr)));
          final_image_->setPixel(px, py, rgb_new);
        }
      }
    }
  }
  this->setPixmap(QPixmap::fromImage(*final_image_));
}

//-----------------------------------------------------------------------------
void QImageCpuDisplay::mousePressEvent(QMouseEvent* event)
{
  if (event->button() == Qt::LeftButton) {
    int offset_y = IUMAX((int)((int)(this->height()) - size_.height), 0);
    mouse_x_old_ = (int)(event->x());
    mouse_y_old_ = (int)(event->y()) - offset_y;
    emit mousePressed(mouse_x_old_, mouse_y_old_);
  }
}

//-----------------------------------------------------------------------------
void QImageCpuDisplay::mouseReleaseEvent(QMouseEvent* event)
{
  if (event->button() == Qt::LeftButton) {
    int offset_y = IUMAX((int)((int)(this->height()) - size_.height), 0);
    emit  mouseReleased((int)(event->x()), (int)(event->y()) - offset_y);
  }
}

//-----------------------------------------------------------------------------
void QImageCpuDisplay::mouseMoveEvent(QMouseEvent* event)
{
  int offset_y = IUMAX((int)((int)(this->height()) - size_.height), 0);
  mouse_x_ = (int)(event->x());
  mouse_y_ = (int)(event->y()) - offset_y;
  emit mouseMoved(mouse_x_old_, mouse_y_old_, mouse_x_, mouse_y_);
  mouse_x_old_ = mouse_x_;
  mouse_y_old_ = mouse_y_;
}

//-----------------------------------------------------------------------------
void QImageCpuDisplay::slotActivateOverlay(const QString& overlay_name)
{
  // search for the signal sender in the overlay list
  std::vector<Overlay>::iterator it;
  for ( it=overlays_.begin() ; it != overlays_.end(); it++ ){
    if(overlay_name == it->title) {
      if (it->show)
        it->show = false;
      else
        it->show = true;
    }
  }
  composeAndShow();
}

//-----------------------------------------------------------------------------
void QImageCpuDisplay::contextMenuEvent(QContextMenuEvent* event)
{
  context_menu_->exec(event->globalPos());
}

//-----------------------------------------------------------------------------
void QImageCpuDisplay::addContextMenuEntry(const QString& title)
{
  // add context menu entry
  QCheckBox *overlay_check_box = new QCheckBox(title, this);
  overlay_check_box->setChecked(true);
  QWidgetAction *action_overlay_widget = new QWidgetAction(this);
  action_overlay_widget->setDefaultWidget(overlay_check_box);
  action_list_overlays_.append(action_overlay_widget);
  connect(overlay_check_box, SIGNAL(toggled(bool)), overlay_signal_mapper_, SLOT(map()));
  overlay_signal_mapper_->setMapping(overlay_check_box, title);
  context_menu_->addAction(action_overlay_widget);
}

////-----------------------------------------------------------------------------
//// QMultiImageDisplay32f
////-----------------------------------------------------------------------------
//QMultiImageDisplay32f::QMultiImageDisplay32f(const float* data,
//                                             size_t width, size_t height, size_t depth,
//                                             size_t pitch, size_t stride,
//                                             const std::string& title,
//                                             float minval, float maxval)
//{
//  minval_ = minval;
//  maxval_ = maxval;
//  size_.width = width;
//  size_.height = height;
//  depth_ = depth;
//  pitch_ = pitch;
//  stride_ = stride;

//  // create images
//  for (size_t plane = 0; plane < depth; plane++){
//    QImage* tmp = new QImage(size_.width, size_.height, QImage::Format_RGB888);
//    for (size_t py = 0; py < size_.height; py++){
//      for (size_t px = 0; px < size_.width; px++){
//        int val = scale(data[plane * stride_ + py * pitch_ + px]);
//        tmp->setPixel(px, py, qRgb(val,val,val));
//      }
//    }
//    images_.push_back(tmp);
//  }
//  my_label_ = new QLabel(this);
//  my_label_->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
//  my_label_->setScaledContents(true);

//  my_slider_ = new QSlider(Qt::Horizontal, this);
//  my_slider_->setMaximum(depth_ - 1);
//  my_slider_->setMinimum(0);
//  my_slider_->setValue(0);
//  my_spinbox_ = new QSpinBox(this);
//  my_spinbox_->setMaximum(depth_ - 1);
//  my_spinbox_->setMinimum(0);
//  my_spinbox_->setValue(0);

//  QHBoxLayout* slice_layout = new QHBoxLayout;
//  slice_layout->addWidget(my_slider_);
//  slice_layout->addWidget(my_spinbox_);
//  connect(my_slider_, SIGNAL(valueChanged(int)), this, SLOT(setSlice(int)));
//  connect(my_spinbox_, SIGNAL(valueChanged(int)), this, SLOT(setSlice(int)));

//  QVBoxLayout* layout = new QVBoxLayout;
//  layout->addLayout(slice_layout);
//  layout->addWidget(my_label_);

//  this->setLayout(layout);
//  this->setWindowTitle(QString::fromStdString(title));

//  this->showSlice(0);
//  this->adjustSize();
//}

////-----------------------------------------------------------------------------
//QMultiImageDisplay32f::~QMultiImageDisplay32f()
//{
//  for (size_t plane = 0; plane < depth_; plane++)
//    delete images_[plane];
//  images_.clear();
//}

////-----------------------------------------------------------------------------
//void QMultiImageDisplay32f::setSlice(int value)
//{
//  my_slider_->setValue(value);
//  my_spinbox_->setValue(value);
//  showSlice(value);
//}

////-----------------------------------------------------------------------------
//void QMultiImageDisplay32f::showSlice(int plane)
//{
//  my_label_->setPixmap(QPixmap::fromImage(*images_[plane]));
//}


} // namespace iuprivate

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

namespace iu {

//-----------------------------------------------------------------------------
QImageCpuDisplay::QImageCpuDisplay(iu::ImageCpu_8u_C1 *image, const std::string &title,
                                   unsigned char minval, unsigned char maxval) :
  iuprivate::QImageCpuDisplay(image, title, minval, maxval)
{ /* */ }

QImageCpuDisplay::QImageCpuDisplay(iu::ImageCpu_8u_C4 *image, const std::string &title,
                                   unsigned char minval, unsigned char maxval) :
  iuprivate::QImageCpuDisplay(image, title, minval, maxval)
{ /* */ }

QImageCpuDisplay::QImageCpuDisplay(iu::ImageCpu_32f_C1 *image, const std::string &title,
                                   float minval, float maxval) :
  iuprivate::QImageCpuDisplay(image, title, minval, maxval)
{ /* */ }

QImageCpuDisplay::QImageCpuDisplay(iu::ImageCpu_32f_C4 *image, const std::string &title,
                                   float minval, float maxval) :
  iuprivate::QImageCpuDisplay(image, title, minval, maxval)
{ /* */ }

QImageCpuDisplay::~QImageCpuDisplay()
{ /* */ }

} // namespace iu

