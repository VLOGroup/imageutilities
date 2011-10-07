#ifndef PLOTWIDGET_H
#define PLOTWIDGET_H

// QT
#include <QObject>
#include <QWidget>
#include <QDoubleSpinBox>
#include <QCheckBox>


#include "QScienceSpinBox.h"


#include "iudefs.h"

// System
#include <list>

class QwtPlot;

namespace iu {

class IUGUI_DLLAPI PlotGapWidget : public QWidget
{
  Q_OBJECT

public:
  PlotGapWidget(QWidget* parent, bool x_log_scale = true, bool y_log_scale = false);
  virtual ~PlotGapWidget();

  void addCurve(std::list<int> x_values, std::list<double> y_values, QString name, QColor color);
  void addCurve(std::list<double> x_values, std::list<double> y_values, QString name, QColor color);

  void addCurve(double* x_values_array, double* y_values_array,
                               int elements_list, QString name, QColor color);
public slots:

private slots:
  void updateXMin(double value);
  void updateXMax(double value);
  void updateXLog(bool value);
  void updateYMin(double value);
  void updateYMax(double value);
  void updateYLog(bool value);

signals:

protected:

private:
  /** Private Constructor. */
  PlotGapWidget();
  /** Copy Constructor. Copying a widget not allowed. */
  PlotGapWidget(const PlotGapWidget&);
  /** Assignment Operator. Private asignment operator prohibits implicit copying. */
  PlotGapWidget& operator= (const PlotGapWidget&);


  QwtPlot* myPlot_;
  QScienceSpinBox* xMin_;
  QScienceSpinBox* xMax_;
  QScienceSpinBox* yMin_;
  QScienceSpinBox* yMax_;
  QCheckBox* xLog_;
  QCheckBox* yLog_;
};

} // namespace iu

#endif // PLOTWIDGET_H
