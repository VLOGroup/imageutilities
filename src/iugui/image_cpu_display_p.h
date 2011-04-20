//
//  W A R N I N G
//  -------------
//
// This file is not part of the IU API.  It exists purely as an
// implementation detail.  This header file may change from version to
// version without notice, or even be removed.
//

#ifndef IUPRIVATE_IMAGE_CPU_DISPLAY_H
#define IUPRIVATE_IMAGE_CPU_DISPLAY_H

#include <QObject>
#include <QScrollArea>
#include <QLabel>
#include <QString>
#include <QPixmap>
#include <QImage>
#include <QRgb>
#include <QMouseEvent>
#include <QMenu>
#include <QSignalMapper>
#include <QList>
#include <QCheckBox>
#include <QWidgetAction>
#include <QSlider>
#include <QSpinBox>
#include "iudefs.h"

namespace iuprivate {

struct Overlay{
  QImage* image;
  QString title;
  bool show;
};

//-----------------------------------------------------------------------------
// QImageDisplay32f
//-----------------------------------------------------------------------------
class QImageCpuDisplay : public QLabel
{
  Q_OBJECT

public:
  QImageCpuDisplay(iu::ImageCpu_8u_C1* image, const std::string& title,
                   unsigned char minval=0x00, unsigned char maxval=0xff);
  QImageCpuDisplay(iu::ImageCpu_8u_C4* image, const std::string& title,
                   unsigned char minval=0x00, unsigned char maxval=0xff);
  QImageCpuDisplay(iu::ImageCpu_32f_C1* image, const std::string& title,
                   float minval=0.0f, float maxval=1.0f);
  QImageCpuDisplay(iu::ImageCpu_32f_C4* image, const std::string& title,
                   float minval=0.0f, float maxval=1.0f);

  ~QImageCpuDisplay();

  void init();

  //! Updates the current image
  void updateImage(iu::ImageCpu_8u_C1* image,
                   unsigned char minval=0x00, unsigned char maxval=0xff);

  //! Updates the current image
  void updateImage(iu::ImageCpu_8u_C4* image,
                   unsigned char minval=0x00, unsigned char maxval=0xff);

  //! Updates the current image
  void updateImage(iu::ImageCpu_32f_C1* image,
                   float minval=0.0f, float maxval=1.0f);

  //! Updates the current image
  void updateImage(iu::ImageCpu_32f_C4* image,
                   float minval=0.0f, float maxval=1.0f);

  //! adds a grayscale overlay with the given color values when the 8bit value is greater 128
  void addOverlay(iu::ImageCpu_8u_C1* image, const std::string& title);

  //! adds a color overlay with the same dimensions as the input and 4 planes (rgba)
  void addOverlay(const std::string& title, const float* data);
  //! adds a color overlay with the same dimensions as the input and 3 planes (rgb)
  void addOverlay(const std::string& title, const float* data, float alpha);
  void deleteOverlays();

  //! Copies the current buffer to the data pointer passed.
  void copyCurrentImage(float* data, size_t width, size_t height, size_t depth);

signals:
  void mouseMoved(int from_x, int from_y, int to_x, int to_y);
  void mousePressed(int x, int y);
  void mouseReleased(int x, int y);

protected slots:
  /** Activates the corresponing overlay to be displayed. */
  void slotActivateOverlay(const QString& overlay_name);

protected:
  void mousePressEvent(QMouseEvent* event);
  void mouseReleaseEvent(QMouseEvent* event);
  void mouseMoveEvent(QMouseEvent* event);
  void contextMenuEvent(QContextMenuEvent* event);

  inline int scale(float value) {
    return (int) (255.0f / (maxval_ - minval_) * (value - minval_));
  };

  inline bool isInside(unsigned int x, unsigned int y)
  {
    if (x >= 0 && x < size_.width && y >= 0 && y < size_.height)
      return true;
    return false;
  };

  void composeAndShow();

  QImage* base_image_;
  QImage* final_image_;
  float minval_, maxval_;
  IuSize size_;

  std::vector<Overlay> overlays_;

  // mouse interaction
  int  mouse_x_old_;
  int  mouse_y_old_;
  int  mouse_x_;
  int  mouse_y_;

  // context menu
  QMenu *context_menu_;
  QSignalMapper *overlay_signal_mapper_;
  QList<QWidgetAction*> action_list_overlays_;

  void addContextMenuEntry(const QString& title);

  QImageCpuDisplay(const QImageCpuDisplay&);
  QImageCpuDisplay();
};


////-----------------------------------------------------------------------------
//// QMultiImageDisplay32f
////-----------------------------------------------------------------------------
//class QMultiImageDisplay32f: public QWidget
//{
//  Q_OBJECT

//public:
//  QMultiImageDisplay32f(const float* data,
//    size_t width, size_t height, size_t depth,
//    size_t pitch, size_t stride,
//    const std::string& title = "", float minval = 0.0f, float maxval = 1.0f);

//  ~QMultiImageDisplay32f();

//public slots:
//  void setSlice(int);

//private:
//  inline int scale(float value) {
//    return (int) (255.0f / (maxval_ - minval_) * (value - minval_));
//  };

//  inline bool isInside(int x, int y){
//    if (x >= 0 && x < (int)width_ && y >= 0 && y < (int)height_)
//      return true;
//    return false;
//  };

//  void showSlice(int plane);

//  float minval_, maxval_;
//  size_t width_, height_, depth_, pitch_, stride_;

//  std::vector<QImage*> images_;

//  QLabel* my_label_;
//  QSlider* my_slider_;
//  QSpinBox* my_spinbox_;

//  QMultiImageDisplay32f(const QMultiImageDisplay32f&);
//  QMultiImageDisplay32f();
//};

} // namespace iuprivate

#endif // IUPRIVATE_IMAGE_CPU_DISPLAY_H
