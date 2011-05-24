#ifndef IMAGEWINDOW_H
#define IMAGEWINDOW_H

#include <QWidget>
//#include <QGraphicsView>
#include "iudefs.h"

class QScrollArea;


namespace iu {

// forward declarations in iu namespace
class QGLImageGpuWidget;

/** class documentation todo
 */
class ImageWindow : public QWidget
{
  Q_OBJECT
public:
  explicit ImageWindow(QWidget *parent = 0);

  void setImage(iu::ImageGpu_8u_C1* image, bool normalize = false);
  void setImage(iu::ImageGpu_32f_C1* image, bool normalize = false);
  void setImage(iu::ImageGpu_8u_C4* image, bool normalize = false);
  void setImage(iu::ImageGpu_32f_C4* image, bool normalize = false);

  void addOverlay(QString name, iu::Image* constraint_image,
                  iu::LinearMemory* lut_values, iu::LinearDeviceMemory_8u_C4* lut_colors,
                  bool active = true);

signals:
  void mouseMoved(int from_x, int from_y, int to_x, int to_y);
  void mousePressed(int x, int y);
  void mousePressed(int x, int y, int global_x, int global_y);
  void mouseReleased(int x, int y);

public slots:
  /* this slot is just for testing purpose */
  void mouseGotPressed(int x, int y, int global_x, int global_y);

protected:
  void setupGeometry();

  QGLImageGpuWidget* image_gpu_widget_;
  QScrollArea* image_gpu_scroll_area_;

};

} // namespace iuprivate

#endif // IMAGEWINDOW_H
