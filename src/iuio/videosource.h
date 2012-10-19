#ifndef VIDEOSOURCE_H
#define VIDEOSOURCE_H

#include <QObject>
#include <opencv2/opencv.hpp>


/** Abstract base class for Video Input */
class VideoSource : public QObject
{
  Q_OBJECT
public:

  /** Get image data. Has to be implemented in a derived class */
  virtual cv ::Mat getImage() = 0;

  /** Get image width. Has to be implemented in a derived class */
  virtual unsigned int getWidth() = 0;

  /** Get image height. Has to be implemented in a derived class */
  virtual unsigned int getHeight() = 0;

  /** Get current frame number. Has to be implemented in a derived class */
  virtual unsigned int getCurrentFrameNr() = 0;

protected:
  unsigned int width_;
  unsigned int height_;
  unsigned int frameNr_;
};

#endif // VIDEOSOURCE_H
