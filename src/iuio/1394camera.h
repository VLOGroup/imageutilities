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
 * Module      : OpticalFlow GUI
 * Class       : $RCSfile$
 * Language    : C++
 * Description : Interface of an IEEE camera with libdc1394v2
 *
 * Author     : 
 * EMail      : 
 *
 */

#ifndef CAMERA_H
#define CAMERA_H

#include <vector>

#include <QMutex>
#include <QThread>
#include <QWaitCondition>

#include <dc1394/dc1394.h>
#include <dc1394/control.h>
#include <dc1394/conversions.h>
#include <cvd/image.h>
#include <cvd/byte.h>
#include <cvd/rgb.h>

using namespace std;

struct RGB8
{
	unsigned char r;
	unsigned char g;
	unsigned char b;
};

struct RGB16
{
	unsigned short r;
	unsigned short g;
	unsigned short b;
};

struct HSV8
{
	unsigned char h;
	unsigned char s;
	unsigned char v;
};

struct HSV16
{
	unsigned short h;
	unsigned short s;
	unsigned short v;
};


class Camera: public QThread
{
public:
	////////////////////////////////////////////////////////////////////////////////
	//	Constructors and Destructor
	////////////////////////////////////////////////////////////////////////////////

	Camera(QObject *parent);
	~Camera();

	
	////////////////////////////////////////////////////////////////////////////////
	//	Methods inherited from QThread
	////////////////////////////////////////////////////////////////////////////////
	void start();
	void quit();

	////////////////////////////////////////////////////////////////////////////////
	//	Misc methods
	////////////////////////////////////////////////////////////////////////////////
	void initialize();
	void waitForNextImage();
	void resetCamera();

	////////////////////////////////////////////////////////////////////////////////
	//	Getters and setters for properties
	////////////////////////////////////////////////////////////////////////////////
	bool isRunning();
	int currentFrame();
	int getWidth();
	int getHeight();
	int getSourceWidth();
	int getSourceHeight();
	int getPixelCount();
	int getSourcePixelCount();
	QString getDeviceFilename();
	void setDeviceFilename(QString filename);
	
	dc1394framerate_t getFramerate();
	void setFramerate(dc1394framerate_t framerate);
	dc1394video_mode_t getMode();
	void setMode(dc1394video_mode_t mode);
	dc1394operation_mode_t getOperationMode();
	void setOperationMode(dc1394operation_mode_t operationMode);
	dc1394speed_t getIsoSpeed();
	void setIsoSpeed(dc1394speed_t speed);

	//latest error encountered
	dc1394error_t getLastError();
	
	dc1394bayer_method_t getBayerMethod();
	void setBayerMethod(dc1394bayer_method_t method);
	dc1394color_filter_t getBayerPattern();
	void setBayerPattern(dc1394color_filter_t filter);

	bool getBayerDecoding();
	void setBayerDecoding(bool val);
	bool getOutputMono();
	void setOutputMono(bool val);
	bool getOutputRGB();
	void setOutputRGB(bool val);
	bool getOutputYUV();
	void setOutputYUV(bool val);
	bool getHalfYUV();
	void setHalfYUV(bool val);

	bool isMonoMode();
	int getBitDepth();

	////////////////////////////////////////////////////////////////////////////////
	//	Methods for retrieving the images
	////////////////////////////////////////////////////////////////////////////////
	int getImageRaw(unsigned char* image);
	int getImageFloat(float* image);
	int getImageMono8(unsigned char* image);
	int getImageRGB8(RGB8* image);
	int getImageRGB8(CVD::Rgb<unsigned char>* image);
	int getImageMono16(unsigned short* image);
	int getImageRGB16(RGB16* image);
	int getImageYUV8(unsigned char* y,unsigned char* u,unsigned char* v);
	int getImageYUV16(unsigned short* y,unsigned short* u,unsigned short* v);

protected:
	////////////////////////////////////////////////////////////////////////////////
	//	Methods inherited from QThread
	////////////////////////////////////////////////////////////////////////////////
	void run();

	////////////////////////////////////////////////////////////////////////////////
	//	Internal methods
	////////////////////////////////////////////////////////////////////////////////
        void storeImage(unsigned char* camera_buffer);
	void cleanUp();


private:


	//port not used yet
	unsigned int _port;
	//node not used yet
	unsigned int _node;
	QString _deviceFilename;
	
	dc1394framerate_t _framerate;
	dc1394video_mode_t _mode;
	dc1394operation_mode_t _operationMode;	
	dc1394speed_t _isoSpeed;

	//initialization for the library
	dc1394_t* _libdc1394;
	//all ieee1394 cameras connected to the pc
	dc1394camera_list_t* _cameras;
	//we select one of the available cameras
	dc1394camera_id_t _cameraId;
	//the pointer to the camera we selected
	dc1394camera_t* _camera;
	
	//latest error encountered
	dc1394error_t _lastError;

	bool _initialized;
	bool _stopped;
	bool _running;
	unsigned int _frameNumber;
	unsigned char* _buffer;

	unsigned char* _outMono8;
	unsigned short* _outMono16;
	RGB8* _outRGB8;
	RGB16* _outRGB16;
	unsigned char* _outY8;
	unsigned char* _outU8;
	unsigned char* _outV8;
	unsigned short* _outY16;
	unsigned short* _outU16;
	unsigned short* _outV16;
	

	//prevents two thread to access the properties concurrently
	QMutex* _propertyMutex;
	//only one thread at a time is allowed to read/write the buffer
	QMutex* _bufferMutex;
	QMutex* _waitMutex;
	QWaitCondition _waitCondition;

	bool _bayerDecoding;
	dc1394bayer_method_t _bayerMethod;
	dc1394color_filter_t _bayerPattern;
	bool _outputMono;
	bool _outputRGB;
	bool _outputYUV;
	bool _halfYUV;
};

#endif
