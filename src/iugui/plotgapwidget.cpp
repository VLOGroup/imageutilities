// includes, Qwt
#include <qwt_plot.h>
#include <qwt_plot_curve.h>
#include <qwt_scale_engine.h>

// System Includes
#include <iostream>

// include local
#include "plotgapwidget.h"

// QT
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>

namespace iu {

//-----------------------------------------------------------------------------
PlotGapWidget::PlotGapWidget(QWidget* parent, bool x_log_scale, bool y_log_scale)
  :QWidget(parent)
{
  myPlot_ = new QwtPlot(0);
  int numDecimals = 4;

  myPlot_->setCanvasBackground(QColor(255,255,255));
  if (x_log_scale)
    myPlot_->setAxisScaleEngine(QwtPlot::xBottom, new QwtLog10ScaleEngine);
  if (y_log_scale)
    myPlot_->setAxisScaleEngine(QwtPlot::yLeft, new QwtLog10ScaleEngine);

  QHBoxLayout* controlLayout = new QHBoxLayout(NULL);
  QLabel* xMinLabel = new QLabel;
  xMinLabel->setText("X-Axis Min");
  controlLayout->addWidget(xMinLabel);
  xMin_ = new QDoubleSpinBox;
  xMin_->setDecimals(numDecimals);
  controlLayout->addWidget(xMin_);
  QLabel* xMaxLabel = new QLabel;
  xMaxLabel->setText("X-Axis Max");
  controlLayout->addWidget(xMaxLabel);
  xMax_ = new QDoubleSpinBox;
  xMax_->setDecimals(numDecimals);
  controlLayout->addWidget(xMax_);
  QLabel* yMinLabel = new QLabel;
  yMinLabel->setText("Y-Axis Min");
  controlLayout->addWidget(yMinLabel);
  yMin_ = new QDoubleSpinBox;
  yMin_->setDecimals(numDecimals);
  controlLayout->addWidget(yMin_);
  QLabel* yMaxLabel = new QLabel;
  yMaxLabel->setText("Y-Axis Max");
  controlLayout->addWidget(yMaxLabel);
  yMax_ = new QDoubleSpinBox;
  yMax_->setDecimals(numDecimals);
  controlLayout->addWidget(yMax_);
  QVBoxLayout* mainLayout = new QVBoxLayout(this);
  mainLayout->addLayout(controlLayout);
  mainLayout->addWidget(myPlot_);
  this->setLayout(mainLayout);

  myPlot_->replot();

  // make some connections
  connect(xMin_, SIGNAL(valueChanged(double)), this, SLOT(updateXMin(double)));
  connect(xMax_, SIGNAL(valueChanged(double)), this, SLOT(updateXMax(double)));
  connect(yMin_, SIGNAL(valueChanged(double)), this, SLOT(updateYMin(double)));
  connect(yMax_, SIGNAL(valueChanged(double)), this, SLOT(updateYMax(double)));
}


//-----------------------------------------------------------------------------
PlotGapWidget::~PlotGapWidget()
{

}

//-----------------------------------------------------------------------------
void PlotGapWidget::addCurve(std::list<double> x_values, std::list<double> y_values,
                             QString name, QColor color)
{
  int elements_list = x_values.size();
  double* x_values_array = new double[elements_list];
  double* y_values_array = new double[elements_list];

  std::list<double>::iterator it;
  int count = 1;
  for ( it=x_values.begin() ; it != x_values.end(); it++ )
  {
    x_values_array[elements_list-count] = *it;
    count++;
  }
  count = 1;

  for ( it=y_values.begin() ; it != y_values.end(); it++ )
  {
    y_values_array[elements_list-count] = *it;
    count++;
  }

  QwtPlotCurve *curveDual = new QwtPlotCurve(name);
  curveDual->setPen(QPen(color));
  curveDual->setData(x_values_array, y_values_array, elements_list);
  curveDual->attach(myPlot_);

  delete x_values_array;
  delete y_values_array;

  myPlot_->setAxisAutoScale(QwtPlot::xBottom);
  myPlot_->setAxisAutoScale(QwtPlot::yLeft);

  myPlot_->replot();

#if (QWT_VERSION >= 0x050200)
  double xmin = myPlot_->axisScaleDiv(QwtPlot::xBottom)->lowerBound();
  double xmax = myPlot_->axisScaleDiv(QwtPlot::xBottom)->upperBound();
  double ymin = myPlot_->axisScaleDiv(QwtPlot::yLeft)->lowerBound();
  double ymax = myPlot_->axisScaleDiv(QwtPlot::yLeft)->upperBound();
#else
  double xmin = myPlot_->axisScaleDiv(QwtPlot::xBottom)->lBound();
  double xmax = myPlot_->axisScaleDiv(QwtPlot::xBottom)->hBound();
  double ymin = myPlot_->axisScaleDiv(QwtPlot::yLeft)->lBound();
  double ymax = myPlot_->axisScaleDiv(QwtPlot::yLeft)->hBound();
#endif
  xMin_->setMinimum(xmin/10.0);
  xMin_->setMaximum(xmax*10.0);
  xMax_->setMinimum(xmin/10.0);
  xMax_->setMaximum(xmax*10.0);
  yMin_->setMinimum(ymin-abs(ymax-ymin));
  yMin_->setMaximum(ymax+abs(ymax-ymin));
  yMax_->setMinimum(ymin-abs(ymax-ymin));
  yMax_->setMaximum(ymax+abs(ymax-ymin));
  xMin_->setValue(xmin);
  xMax_->setValue(xmax);
  yMin_->setValue(ymin);
  yMax_->setValue(ymax);

}

//-----------------------------------------------------------------------------
void PlotGapWidget::updateXMin(double value)
{
  myPlot_->setAxisScale(QwtPlot::xBottom, xMin_->value(), xMax_->value(), 0);
  myPlot_->replot();
}

//-----------------------------------------------------------------------------
void PlotGapWidget::updateXMax(double value)
{
  myPlot_->setAxisScale(QwtPlot::xBottom, xMin_->value(), xMax_->value(), 0);
  myPlot_->replot();
}

//-----------------------------------------------------------------------------
void PlotGapWidget::updateYMin(double value)
{
  myPlot_->setAxisScale(QwtPlot::yLeft, yMin_->value(), yMax_->value(), 0);
  myPlot_->replot();
}

//-----------------------------------------------------------------------------
void PlotGapWidget::updateYMax(double value)
{
  myPlot_->setAxisScale(QwtPlot::yLeft, yMin_->value(), yMax_->value(), 0);
  myPlot_->replot();
}

} // namespace iu

