#ifndef IMAGE_CPU_DISPLAY_H
#define IMAGE_CPU_DISPLAY_H

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

#include "image_cpu_display_p.h"

namespace iu {

//-----------------------------------------------------------------------------
// QImageDisplay32f
//-----------------------------------------------------------------------------
class IU_DLLAPI QImageCpuDisplay : public iuprivate::QImageCpuDisplay
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

  virtual ~QImageCpuDisplay();

  void init();

  // adds a color overlay with the same dimensions as the input and 4 planes (rgba)
  void addOverlay(const std::string& title, const float* data);
  // adds a color overlay with the same dimensions as the input and 3 planes (rgb)
  void addOverlay(const std::string& title, const float* data, float alpha);
  void deleteOverlays();

  // copies the current buffer to the data pointer passed. width, height need
  // to be the same size as the input, depth has to be 3
  void copyCurrentImage(float* data, size_t width, size_t height, size_t depth);

signals:
  void mouseMoved(int from_x, int from_y, int to_x, int to_y);
  void mousePressed(int x, int y);
  void mouseReleased(int x, int y);

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

#endif // IMAGE_CPU_DISPLAY_H
