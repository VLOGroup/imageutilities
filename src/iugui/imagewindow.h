#ifndef IMAGEWINDOW_H
#define IMAGEWINDOW_H

#include <QWidget>
//#include <QGraphicsView>
#include "iudefs.h"

class QScrollArea;
class QScrollBar;
class QToolBar;
class QLabel;
class QSpinBox;

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

  void setVolume(iu::VolumeGpu_32f_C1* volume, bool normalize = false);

  void addOverlay(QString name, iu::Image* constraint_image,
                  iu::LinearMemory* lut_values, iu::LinearDeviceMemory_8u_C4* lut_colors,
                  bool active = true, IuComparisonOperator comp_op = IU_EQUAL);
  void update();
  void setCursor(QCursor csr);

  void setFilePrefix(QString prefix) {file_prefix_ = prefix;}

  QGLImageGpuWidget* getWidget() {return image_gpu_widget_;}

signals:
  void mouseMoved(int from_x, int from_y, int to_x, int to_y);
  void mousePressed(int x, int y);
  void mousePressed(int x, int y, int global_x, int global_y);
  void mouseReleased(int x, int y);

  void zoomChanged(float val);

public slots:
  void pan(int from_x, int from_y, int to_x, int to_y);
  void zoomed(double factor);

  void showToolbar(bool val=true);
  void updatePixelInfo(QString text);

  void sliceSelect(int val);
  void on_action_save__triggered();

protected:
  void setupGeometry();

private:
  iu::VolumeGpu_32f_C1* volume_;
  iu::ImageGpu_32f_C1* image_;

  void panScrollBar(QScrollBar *scrollBar, int offset);
  void adjustScrollBar(QScrollBar *scrollBar, double factor);

  QGLImageGpuWidget* image_gpu_widget_;
  QScrollArea* image_gpu_scroll_area_;

  QToolBar* tool_bar_;
  QAction* action_save_;

  QLabel* pixel_info_;

  QSpinBox* slice_selector_;
  QAction* slice_action_;

  QString file_prefix_;

};

} // namespace iuprivate

#endif // IMAGEWINDOW_H
