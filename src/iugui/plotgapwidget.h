#ifndef PLOTWIDGET_H
#define PLOTWIDGET_H

// QT
#include <QObject>
#include <QWidget>
#include <QDoubleSpinBox>

#include "iudefs.h"

// System
#include <list>

class QwtPlot;

namespace iu {

class IU_DLLAPI PlotGapWidget : public QWidget
{
  Q_OBJECT

public:
  PlotGapWidget(QWidget* parent, bool x_log_scale = true, bool y_log_scale = false);
  virtual ~PlotGapWidget();

  void addCurve(std::list<double> x_values, std::list<double> y_values, QString name, QColor color);

public slots:

private slots:
  void updateXMin(double value);
  void updateXMax(double value);
  void updateYMin(double value);
  void updateYMax(double value);

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
  QDoubleSpinBox* xMin_;
  QDoubleSpinBox* xMax_;
  QDoubleSpinBox* yMin_;
  QDoubleSpinBox* yMax_;
};

} // namespace iu

#endif // PLOTWIDGET_H
