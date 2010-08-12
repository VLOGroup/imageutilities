/*
 * Copyright (c) ICG. All rights reserved.
 *
 * Institute for Computer Graphics and Vision
 * Graz University of Technology / Austria
 *
 *
 * This software is distributed WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 * PURPOSE.  See the above copyright notices for more information.
 *
 *
 * Project     : vmgpu
 * Module      : Tools
 * Language    : C++
 * Description : Definition/Implementation of a thread responsible to process and provide images from a camera
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */


#ifndef PGRCAMERAIMAGECONSUMER_H
#define PGRCAMERAIMAGECONSUMER_H

//#include <QtCore>
#include <QThread>
#include <QImage>
class PGRCameraData;

class PGRCameraImageConsumer : public QThread
{
  Q_OBJECT;

public:
  PGRCameraImageConsumer(PGRCameraData* data);
  virtual ~PGRCameraImageConsumer();

  virtual void run();

signals:
  void signalPushNewQImage(const QImage &image, unsigned int& stride,
                           unsigned int& bits_per_pixel);

public slots:
  void quit();

private:
  PGRCameraData* data_;
  bool thread_running_;
};

#endif // PGRCAMERAIMAGECONSUMER_H

