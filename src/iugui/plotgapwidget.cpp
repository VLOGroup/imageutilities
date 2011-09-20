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
#include <QSpacerItem>

namespace iu {

//-----------------------------------------------------------------------------
PlotGapWidget::PlotGapWidget(QWidget* parent, bool x_log_scale, bool y_log_scale)
  :QWidget(parent)
{
  myPlot_ = new QwtPlot(0);
  int numDecimals = 2;

  myPlot_->setCanvasBackground(QColor(255,255,255));
  if (x_log_scale)
    myPlot_->setAxisScaleEngine(QwtPlot::xBottom, new QwtLog10ScaleEngine);
  if (y_log_scale)
    myPlot_->setAxisScaleEngine(QwtPlot::yLeft, new QwtLog10ScaleEngine);

  QHBoxLayout* controlLayout = new QHBoxLayout(NULL);

  QLabel* yMinLabel = new QLabel;
  yMinLabel->setText("Y-Axis");
  controlLayout->addWidget(yMinLabel);
  yMin_ = new QScienceSpinBox;
  yMin_->setDecimals(numDecimals);
  controlLayout->addWidget(yMin_);

  yMax_ = new QScienceSpinBox;
  yMax_->setDecimals(numDecimals);
  controlLayout->addWidget(yMax_);

  yLog_ = new QCheckBox();
  yLog_->setText("Log scale");
  if (y_log_scale)
    yLog_->setChecked(true);
  else
    yLog_->setChecked(false);
  controlLayout->addWidget(yLog_);

  controlLayout->insertStretch(4);

  QLabel* xMinLabel = new QLabel;
  xMinLabel->setText("X-Axis");
  controlLayout->addWidget(xMinLabel);
  xMin_ = new QScienceSpinBox;
  xMin_->setDecimals(numDecimals);
  controlLayout->addWidget(xMin_);

  xMax_ = new QScienceSpinBox;
  xMax_->setDecimals(numDecimals);
  controlLayout->addWidget(xMax_);

  xLog_ = new QCheckBox;
  xLog_->setText("Log scale");
  if (x_log_scale)
    xLog_->setChecked(true);
  else
    xLog_->setChecked(false);
  controlLayout->addWidget(xLog_);

  QVBoxLayout* mainLayout = new QVBoxLayout(this);
  mainLayout->addLayout(controlLayout);
  mainLayout->addWidget(myPlot_);
  this->setLayout(mainLayout);

  myPlot_->replot();

  // make some connections
  connect(xMin_, SIGNAL(valueChanged(double)), this, SLOT(updateXMin(double)));
  connect(xMax_, SIGNAL(valueChanged(double)), this, SLOT(updateXMax(double)));
  connect(xLog_, SIGNAL(toggled(bool)), this, SLOT(updateXLog(bool)));
  connect(yMin_, SIGNAL(valueChanged(double)), this, SLOT(updateYMin(double)));
  connect(yMax_, SIGNAL(valueChanged(double)), this, SLOT(updateYMax(double)));
  connect(yLog_, SIGNAL(toggled(bool)), this, SLOT(updateYLog(bool)));
}


//-----------------------------------------------------------------------------
PlotGapWidget::~PlotGapWidget()
{

}

//-----------------------------------------------------------------------------
void PlotGapWidget::addCurve(std::list<int> x_values, std::list<double> y_values,
                             QString name, QColor color)
{
  int elements_list = x_values.size();
  double* x_values_array = new double[elements_list];
  double* y_values_array = new double[elements_list];

//  // skip first element
//  if (*(x_values.begin()) == 0)
//  {
//    x_values.pop_front();
//    y_values.pop_front();
//  }

  // copy
  std::list<int>::iterator itx;
  int count = 1;
  for ( itx=x_values.begin() ; itx != x_values.end(); itx++ )
  {
    x_values_array[elements_list-count] = *itx;
    count++;
  }
  count = 1;

 std::list<double>::iterator ity;
  for ( ity=y_values.begin() ; ity != y_values.end(); ity++ )
  {
    y_values_array[elements_list-count] = *ity;
    count++;
  }

  printf("OOOOOOOOOOOOOOOOOOOOOOOOO\n");
  for (int i=0; i<count-1; i++)
  {
    printf("(x,y): %f, %f\n", x_values_array[i], y_values_array[i]);
  }
  printf("OOOOOOOOOOOOOOOOOOOOOOOOO\n");


  addCurve(x_values_array, y_values_array, elements_list, name, color);

  delete x_values_array;
  delete y_values_array;
}


//-----------------------------------------------------------------------------
void PlotGapWidget::addCurve(std::list<double> x_values, std::list<double> y_values,
                             QString name, QColor color)
{
  int elements_list = x_values.size();
  double* x_values_array = new double[elements_list];
  double* y_values_array = new double[elements_list];

//  // skip first element
//  if (*(x_values.begin()) == 0)
//  {
//    x_values.pop_front();
//    y_values.pop_front();
//  }


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


  printf("########################\n");
  for (int i=0; i<count-1; i++)
  {
    printf("(x,y): %f, %f\n", x_values_array[i], y_values_array[i]);
  }
  printf("########################\n");

  addCurve(x_values_array, y_values_array, elements_list, name, color);

  delete x_values_array;
  delete y_values_array;
}

//-----------------------------------------------------------------------------
void PlotGapWidget::addCurve(double* x_values_array, double* y_values_array,
                             int elements_list, QString name, QColor color)
{
  QwtPlotCurve *curveDual = new QwtPlotCurve(name);
  curveDual->setPen(QPen(color));

#if QWT_VERSION < 0x060000
    curveDual->setData(x_values_array, y_values_array, elements_list);
#else
    curveDual->setSamples(x_values_array, y_values_array, elements_list);
#endif

  curveDual->attach(myPlot_);

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
//  xMin_->setMinimum(xmin/10.0);
//  xMin_->setMaximum(xmax*10.0);
//  xMax_->setMinimum(xmin/10.0);
//  xMax_->setMaximum(xmax*10.0);
//  yMin_->setMinimum(ymin-abs(ymax-ymin));
//  yMin_->setMaximum(ymax+abs(ymax-ymin));
//  yMax_->setMinimum(ymin-abs(ymax-ymin));
//  yMax_->setMaximum(ymax+abs(ymax-ymin));
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
void PlotGapWidget::updateXLog(bool value)
{
  if (value)
    myPlot_->setAxisScaleEngine(QwtPlot::xBottom, new QwtLog10ScaleEngine);
  else
    myPlot_->setAxisScaleEngine(QwtPlot::xBottom, new QwtLinearScaleEngine);
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

//-----------------------------------------------------------------------------
void PlotGapWidget::updateYLog(bool value)
{
  if (value)
    myPlot_->setAxisScaleEngine(QwtPlot::yLeft, new QwtLog10ScaleEngine);
  else
    myPlot_->setAxisScaleEngine(QwtPlot::yLeft, new QwtLinearScaleEngine);
  myPlot_->replot();
}

} // namespace iu

