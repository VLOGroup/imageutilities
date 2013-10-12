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
 * Description : Implementation of an IEEE camera with libdc1394v2
 *
 * Author     : 
 * EMail      : 
 *
 */

#include <cassert>
#include <iostream>

#include "1394camera.h"
#include <dc1394/types.h>
#include <dc1394/video.h>
#include <dc1394/dc1394.h>

using namespace std;

/*dc1394video_mode_t
DC1394_VIDEO_MODE_640x480_RGB8
DC1394_VIDEO_MODE_640x480_MONO8
DC1394_VIDEO_MODE_640x480_MONO16
DC1394_VIDEO_MODE_800x600_RGB8
DC1394_VIDEO_MODE_800x600_MONO8
DC1394_VIDEO_MODE_1024x768_RGB8
DC1394_VIDEO_MODE_1024x768_MONO8
DC1394_VIDEO_MODE_800x600_MONO16
DC1394_VIDEO_MODE_1024x768_MONO16
DC1394_VIDEO_MODE_1280x960_RGB8
DC1394_VIDEO_MODE_1280x960_MONO8
DC1394_VIDEO_MODE_1600x1200_RGB8
DC1394_VIDEO_MODE_1600x1200_MONO8
DC1394_VIDEO_MODE_1280x960_MONO16
DC1394_VIDEO_MODE_1600x1200_MONO16
*/

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////

std::string getVideoModeAsString(const dc1394video_mode_t mode);

bool isValidMode(dc1394video_mode_t mode)
{
	if (
		(mode==DC1394_VIDEO_MODE_640x480_RGB8)||
		(mode==DC1394_VIDEO_MODE_640x480_MONO8)||
		(mode==DC1394_VIDEO_MODE_640x480_MONO16)||
		(mode==DC1394_VIDEO_MODE_800x600_RGB8)||
		(mode==DC1394_VIDEO_MODE_800x600_MONO8)||
		(mode==DC1394_VIDEO_MODE_800x600_MONO16)||
		(mode==DC1394_VIDEO_MODE_1024x768_RGB8)||
		(mode==DC1394_VIDEO_MODE_1024x768_MONO8)||
		(mode==DC1394_VIDEO_MODE_1024x768_MONO16)||
		(mode==DC1394_VIDEO_MODE_1280x960_RGB8)||
		(mode==DC1394_VIDEO_MODE_1280x960_MONO8)||
		(mode==DC1394_VIDEO_MODE_1280x960_MONO16)||
		(mode==DC1394_VIDEO_MODE_1600x1200_RGB8)||
		(mode==DC1394_VIDEO_MODE_1600x1200_MONO8)||
                (mode==DC1394_VIDEO_MODE_1600x1200_MONO16) ||
                (mode == DC1394_VIDEO_MODE_FORMAT7_1))
		return true;
	else
		return false;
}

int widthOfMode(dc1394video_mode_t mode)
{
  int result = -1;
  switch (mode)
  {
  case (DC1394_VIDEO_MODE_640x480_RGB8):
  case (DC1394_VIDEO_MODE_640x480_MONO8):
  case (DC1394_VIDEO_MODE_640x480_MONO16):
    result = 640;
    break;
  case (DC1394_VIDEO_MODE_800x600_RGB8):
  case (DC1394_VIDEO_MODE_800x600_MONO8):
  case (DC1394_VIDEO_MODE_800x600_MONO16):
    result = 800;
    break;
  case (DC1394_VIDEO_MODE_1024x768_RGB8):
  case (DC1394_VIDEO_MODE_1024x768_MONO8):
  case (DC1394_VIDEO_MODE_1024x768_MONO16):
    result = 1024;
    break;
  case (DC1394_VIDEO_MODE_1280x960_RGB8):
  case (DC1394_VIDEO_MODE_1280x960_MONO8):
  case (DC1394_VIDEO_MODE_1280x960_MONO16):
    result = 1280;
    break;
  case (DC1394_VIDEO_MODE_1600x1200_RGB8):
  case (DC1394_VIDEO_MODE_1600x1200_MONO8):
  case (DC1394_VIDEO_MODE_1600x1200_MONO16):
    result = 1600;
    break;
  case (DC1394_VIDEO_MODE_FORMAT7_1):
    return 512;
    break;
  default:
    break;
  }
  return result;
}

int heightOfMode(dc1394video_mode_t mode)
{
	int result = -1;
	switch (mode)
	{
		case (DC1394_VIDEO_MODE_640x480_RGB8):
		case (DC1394_VIDEO_MODE_640x480_MONO8):
		case (DC1394_VIDEO_MODE_640x480_MONO16):
			result = 480;
			break;
		case (DC1394_VIDEO_MODE_800x600_RGB8):
		case (DC1394_VIDEO_MODE_800x600_MONO8):
		case (DC1394_VIDEO_MODE_800x600_MONO16):
			result = 600;
			break;
		case (DC1394_VIDEO_MODE_1024x768_RGB8):
		case (DC1394_VIDEO_MODE_1024x768_MONO8):
		case (DC1394_VIDEO_MODE_1024x768_MONO16):
			result = 768;
			break;
		case (DC1394_VIDEO_MODE_1280x960_RGB8):
		case (DC1394_VIDEO_MODE_1280x960_MONO8):
		case (DC1394_VIDEO_MODE_1280x960_MONO16):
			result = 960;
			break;
		case (DC1394_VIDEO_MODE_1600x1200_RGB8):
		case (DC1394_VIDEO_MODE_1600x1200_MONO8):
		case (DC1394_VIDEO_MODE_1600x1200_MONO16):
			result = 1200;
			break;
        case (DC1394_VIDEO_MODE_FORMAT7_1):
          return 384;
          break;
		default:
			break;
	}
	return result;
}

int bitdepthOfMode(dc1394video_mode_t mode)
{
	int result = -1;
	switch (mode)
	{
		case (DC1394_VIDEO_MODE_640x480_RGB8):
		case (DC1394_VIDEO_MODE_640x480_MONO8):
		case (DC1394_VIDEO_MODE_800x600_RGB8):
		case (DC1394_VIDEO_MODE_800x600_MONO8):
		case (DC1394_VIDEO_MODE_1024x768_RGB8):
		case (DC1394_VIDEO_MODE_1024x768_MONO8):
		case (DC1394_VIDEO_MODE_1280x960_RGB8):
		case (DC1394_VIDEO_MODE_1280x960_MONO8):
		case (DC1394_VIDEO_MODE_1600x1200_RGB8):
		case (DC1394_VIDEO_MODE_1600x1200_MONO8):
        case (DC1394_VIDEO_MODE_FORMAT7_1):
			result = 8;
			break;
		case (DC1394_VIDEO_MODE_1600x1200_MONO16):
		case (DC1394_VIDEO_MODE_640x480_MONO16):
		case (DC1394_VIDEO_MODE_800x600_MONO16):
		case (DC1394_VIDEO_MODE_1024x768_MONO16):
		case (DC1394_VIDEO_MODE_1280x960_MONO16):
			result = 16;
			break;
		default:
			break;
	}
	return result;
}

int pixelSizeOfMode(dc1394video_mode_t mode)
{
	int result = -1;
	switch (mode)
	{

        case (DC1394_VIDEO_MODE_640x480_MONO8):
        case (DC1394_VIDEO_MODE_800x600_MONO8):
        case (DC1394_VIDEO_MODE_1024x768_MONO8):
        case (DC1394_VIDEO_MODE_1280x960_MONO8):
        case (DC1394_VIDEO_MODE_1600x1200_MONO8):
    case (DC1394_VIDEO_MODE_FORMAT7_1):
          result = 1;
          break;
        case (DC1394_VIDEO_MODE_1600x1200_MONO16):
        case (DC1394_VIDEO_MODE_640x480_MONO16):
        case (DC1394_VIDEO_MODE_800x600_MONO16):
        case (DC1394_VIDEO_MODE_1024x768_MONO16):
        case (DC1394_VIDEO_MODE_1280x960_MONO16):
          result = 2;
          break;
        case (DC1394_VIDEO_MODE_640x480_RGB8):
        case (DC1394_VIDEO_MODE_800x600_RGB8):
        case (DC1394_VIDEO_MODE_1024x768_RGB8):
        case (DC1394_VIDEO_MODE_1280x960_RGB8):
        case (DC1394_VIDEO_MODE_1600x1200_RGB8):

          result = 3;
          break;
        default:
          break;
	}
	return result;
}

bool isMonoMode(dc1394video_mode_t mode)
{
	bool result = -1;
	switch (mode)
	{
		case (DC1394_VIDEO_MODE_640x480_RGB8):
		case (DC1394_VIDEO_MODE_800x600_RGB8):
		case (DC1394_VIDEO_MODE_1024x768_RGB8):
		case (DC1394_VIDEO_MODE_1280x960_RGB8):
		case (DC1394_VIDEO_MODE_1600x1200_RGB8):
			result = false;
			break;
		case (DC1394_VIDEO_MODE_1280x960_MONO8):
		case (DC1394_VIDEO_MODE_1280x960_MONO16):
		case (DC1394_VIDEO_MODE_1024x768_MONO8):
		case (DC1394_VIDEO_MODE_1024x768_MONO16):
		case (DC1394_VIDEO_MODE_800x600_MONO8):
		case (DC1394_VIDEO_MODE_800x600_MONO16):
		case (DC1394_VIDEO_MODE_640x480_MONO8):
		case (DC1394_VIDEO_MODE_640x480_MONO16):
		case (DC1394_VIDEO_MODE_1600x1200_MONO8):
		case (DC1394_VIDEO_MODE_1600x1200_MONO16):
			result = true;
			break;
		default:
			break;
	}
	return result;
}

std::string getVideoModeAsString(const dc1394video_mode_t mode)
{
  switch (mode)
  {
  case DC1394_VIDEO_MODE_160x120_YUV444:
    return "160x120 YUV444";
    break;
  case DC1394_VIDEO_MODE_320x240_YUV422:
    return "320x240 YUV422";
    break;
  case DC1394_VIDEO_MODE_640x480_YUV411:
    return "640x480 YUV411";
    break;
  case DC1394_VIDEO_MODE_640x480_YUV422:
    return "640x480 YUV422";
    break;
  case DC1394_VIDEO_MODE_640x480_RGB8:
    return "640x480 RGB8";
    break;
  case DC1394_VIDEO_MODE_640x480_MONO8:
    return "640x480 MONO8";
    break;
  case DC1394_VIDEO_MODE_640x480_MONO16:
    return "640x480 MONO16";
    break;
  case DC1394_VIDEO_MODE_800x600_YUV422:
    return "800x600 YUV422";
    break;
  case DC1394_VIDEO_MODE_800x600_RGB8:
    return "800x600 RGB8";
    break;
  case DC1394_VIDEO_MODE_800x600_MONO8:
    return "800x600 MONO8";
    break;
  case DC1394_VIDEO_MODE_1024x768_YUV422:
    return "1024x768 YUV422";
    break;
  case DC1394_VIDEO_MODE_1024x768_RGB8:
    return "1024x768 RGB8";
    break;
  case DC1394_VIDEO_MODE_1024x768_MONO8:
    return "1024x768 MONO8";
    break;
  case DC1394_VIDEO_MODE_800x600_MONO16:
    return "800x600 MONO16";
    break;
  case DC1394_VIDEO_MODE_1024x768_MONO16:
    return "1024x768 MONO16";
    break;
  case DC1394_VIDEO_MODE_1280x960_YUV422:
    return "1280x960 YUV422";
    break;
  case DC1394_VIDEO_MODE_1280x960_RGB8:
    return "1280x960 RGB8";
    break;
  case DC1394_VIDEO_MODE_1280x960_MONO8:
    return "1280x960 MONO8";
    break;
  case DC1394_VIDEO_MODE_1600x1200_YUV422:
    return "1600x1200 YUV422";
    break;
  case DC1394_VIDEO_MODE_1600x1200_RGB8:
    return "1600x1200 RGB8";
    break;
  case DC1394_VIDEO_MODE_1600x1200_MONO8:
    return "1600x1200 MONO8";
    break;
  case DC1394_VIDEO_MODE_1280x960_MONO16:
    return "1280x960 MONO16";
    break;
  case DC1394_VIDEO_MODE_1600x1200_MONO16:
    return "1600x1200 MONO16";
    break;
  case DC1394_VIDEO_MODE_EXIF:
    return "EXIF";
    break;
  case DC1394_VIDEO_MODE_FORMAT7_0:
    return "Format 7_0";
    break;
  case DC1394_VIDEO_MODE_FORMAT7_1:
    return "Format 7_1";
    break;
  case DC1394_VIDEO_MODE_FORMAT7_2:
    return "Format 7_2";
    break;
  case DC1394_VIDEO_MODE_FORMAT7_3:
    return "Format 7_3";
    break;
  case DC1394_VIDEO_MODE_FORMAT7_4:
    return "Format 7_4";
    break;
  case DC1394_VIDEO_MODE_FORMAT7_5:
    return "Format 7_5";
    break;
  case DC1394_VIDEO_MODE_FORMAT7_6:
    return "Format 7_6";
    break;
  case DC1394_VIDEO_MODE_FORMAT7_7:
    return "Format 7_7";
    break;
  default:
    return "Unrecognized format";
    break;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Constructors and Destructor
////////////////////////////////////////////////////////////////////////////////

/*!Constructor
 *
*/
Camera::Camera(QObject *parent)
		: QThread( parent )
{
	_propertyMutex = new QMutex(QMutex::Recursive);
	_bufferMutex = new QMutex(QMutex::Recursive);
	_waitMutex = new QMutex();
	_libdc1394 = dc1394_new();

	_initialized = false;
	_stopped = true;
	_frameNumber = 0;

	_port = 0;
	_node = 0;
	_deviceFilename = "";
	
	_framerate = DC1394_FRAMERATE_60;
	_operationMode = DC1394_OPERATION_MODE_LEGACY;
	_isoSpeed = DC1394_ISO_SPEED_400;
        _mode = DC1394_VIDEO_MODE_640x480_MONO8;

	_cameras = 0;
	_camera = 0;
	
	//latest error encountered
	_lastError = DC1394_SUCCESS;

	_buffer = 0;

	_outMono8 = 0;
	_outRGB8 = 0;
	_outMono16 = 0;
	_outRGB16 = 0;
	_outY8 = 0;
	_outU8 = 0;
	_outV8 = 0;
	_outY16 = 0;
	_outU16 = 0;
	_outV16 = 0;

	_bayerDecoding = false;
	_bayerMethod = DC1394_BAYER_METHOD_NEAREST;
	_bayerPattern = DC1394_COLOR_FILTER_GBRG;
	_outputMono = true;
	_outputRGB = false;
	_outputYUV = false;
	_halfYUV = false;

        this->initialize();
}

/*!Destructor
 *
*/
Camera::~Camera()
{
	cleanUp();
	dc1394_free(_libdc1394);
	delete _propertyMutex;
	delete _bufferMutex;
	delete _waitMutex;
}

////////////////////////////////////////////////////////////////////////////////
//	Methods inherited from QThread
////////////////////////////////////////////////////////////////////////////////
/*! run()
 *  This is what the thread does.
*/
void Camera::run()
{
  //cout << "CAMERA THREAD : capture setup" << endl;
  _lastError = dc1394_capture_setup(_camera, 4, DC1394_CAPTURE_FLAGS_DEFAULT);
  DC1394_WRN(_lastError,"CAMERA THREAD : error in capture setup");
  if (_lastError!=DC1394_SUCCESS)
  {
    return;
  }
  /* Start transmission */
  //cout << "CAMERA THREAD : set transmission" << endl;
  _lastError = dc1394_video_set_transmission(_camera, DC1394_ON);
  DC1394_WRN(_lastError,"CAMERA THREAD : error in set transmission");
  if (_lastError!=DC1394_SUCCESS)
  {
    return;
  }

  dc1394video_frame_t* frame;
  while (!_stopped)
  {
    //cout << "CAMERA THREAD : capture dequeue" << endl;
    _lastError = dc1394_capture_dequeue(_camera, DC1394_CAPTURE_POLICY_WAIT, &frame);
    DC1394_WRN(_lastError,"CAMERA THREAD : error in capture dequeue");
    if (_lastError!=DC1394_SUCCESS)
    {
      _running = false;
      return;
    }


    //cout << "CAMERA THREAD : _bufferMutex.lock" << endl;
    _bufferMutex->lock();
    //cout << "CAMERA THREAD : store image" << endl;
    storeImage(frame->image);

//    switch (frame->color_coding)
//    {
//    case DC1394_COLOR_CODING_MONO8:
//      std::cout << "mono 8" << endl;
//      break;
//    case DC1394_COLOR_CODING_YUV411:
//      std::cout << "yuv 411" << endl;
//      break;
//    case DC1394_COLOR_CODING_YUV422:
//      std::cout << "yuv 422" << endl;
//      break;
//    case DC1394_COLOR_CODING_YUV444:
//      std::cout << "yuv 444" << endl;
//      break;
//    case DC1394_COLOR_CODING_RGB8:
//      std::cout << "rgb 8" << endl;
//      break;
//    case DC1394_COLOR_CODING_MONO16:
//      std::cout << "mono 16" << endl;
//      break;
//    case DC1394_COLOR_CODING_RGB16:
//      std::cout << "rgb 16" << endl;
//      break;
//    case DC1394_COLOR_CODING_MONO16S:
//      std::cout << "mono 16s" << endl;
//      break;
//    case DC1394_COLOR_CODING_RGB16S:
//      std::cout << "rgb 16s" << endl;
//      break;
//    case DC1394_COLOR_CODING_RAW8:
//      std::cout << "raw 8" << endl;
//      break;
//    case DC1394_COLOR_CODING_RAW16:
//      std::cout << "raw 16" << endl;
//      break;
//    }



    _bufferMutex->unlock();

    _running = true;
    //cout << "CAMERA THREAD : capture enqueue" << endl;
    _lastError = dc1394_capture_enqueue(_camera, frame);
    DC1394_WRN(_lastError,"CAMERA THREAD : error in capture enqueue");
    if (_lastError!=DC1394_SUCCESS)
    {
      _running = false;
      return;
    }

    ++_frameNumber;

    //cout << "CAMERA THREAD : wait all" << endl;
    _waitCondition.wakeAll();
  }
  _running = false;
  /* Stop transmission */
  _lastError = dc1394_video_set_transmission(_camera, DC1394_OFF);
  DC1394_WRN(_lastError,"CAMERA THREAD : error in set transmission OFF");
  if (_lastError!=DC1394_SUCCESS)
  {
    _running = false;
    return;
  }

  _lastError = dc1394_capture_stop(_camera);
  DC1394_WRN(_lastError,"CAMERA THREAD : error in capture stop");
  if (_lastError!=DC1394_SUCCESS)
  {
    _running = false;
    return;
  }
}

/*! start()
   *  starts a new thread.
  */
void Camera::start()
{
  _stopped = false;
  QThread::start();
}

/*! Signals the thread to stop.
 * 
*/
void Camera::quit()
{
	_stopped = true;
	QThread::quit();
}
////////////////////////////////////////////////////////////////////////////////
//	Misc methods
////////////////////////////////////////////////////////////////////////////////

void Camera::initialize()
{
  std::cout << "initializing libdc1394 camera" << std::endl;
  QMutexLocker locker(_propertyMutex);
  if (_stopped)
  {
    if (_initialized)
      cleanUp();

    _lastError = dc1394_camera_enumerate(_libdc1394,&_cameras);
    DC1394_ERR(_lastError,"Failed to enumerate cameras");

    if (_cameras->num == 0) {
      /* Verify that we have at least one camera */
      dc1394_log_error("No cameras found");
      exit(1);
    }
    _cameraId = _cameras->ids[0];
    _camera = dc1394_camera_new(_libdc1394,_cameraId.guid);
    if (!_camera) {
      dc1394_log_error("Failed to initialize camera with guid %llx", _cameraId.guid);
      exit(1);
    }


    int pc = getSourcePixelCount();
    // buffer to store the camera frame
    _buffer = new unsigned char[pc*pixelSizeOfMode(_mode)];

    _lastError = dc1394_video_set_framerate(_camera,_framerate);
    DC1394_ERR(_lastError,"Failed to set framerate");
    _lastError = dc1394_video_set_mode(_camera,_mode);
    DC1394_ERR(_lastError,"Failed to set mode");
    _lastError = dc1394_video_set_operation_mode(_camera,_operationMode);
    DC1394_ERR(_lastError,"Failed to set operation_mode");
    _lastError = dc1394_video_set_iso_speed(_camera,_isoSpeed);
    DC1394_ERR(_lastError,"Failed to set iso_speed");

    _outMono8 = new unsigned char[pc];
    _outRGB8 = new RGB8[pc];
    _outMono16 = new unsigned short[pc];
    _outRGB16 = new RGB16[pc];
    _outY8 = new unsigned char[pc];
    _outU8 = new unsigned char[pc];
    _outV8 = new unsigned char[pc];
    _outY16 = new unsigned short[pc];
    _outU16 = new unsigned short[pc];
    _outV16 = new unsigned short[pc];

    dc1394video_mode_t mmode;
    dc1394_video_get_mode(_camera, &mmode);
    std::cout << "[1394camera] video mode " << getVideoModeAsString(mmode) << std::endl;

    _frameNumber = 0;

    _running = false;

    _initialized = true;
  }
}

void Camera::waitForNextImage()
{
	if (_running)
	{
		_waitMutex->lock();
		_waitCondition.wait(_waitMutex);
		_waitMutex->unlock();
	}
}

void Camera::resetCamera()
{
	_lastError = dc1394_reset_bus(_camera);
	DC1394_WRN(_lastError,"Failed to set framerate");
}
////////////////////////////////////////////////////////////////////////////////
//	Internal methods
////////////////////////////////////////////////////////////////////////////////

/*! helper function for run()
 *
*/
void Camera::storeImage(unsigned char* camera_buffer)
{
  if (_mode == DC1394_VIDEO_MODE_640x480_RGB8 ||
      _mode == DC1394_VIDEO_MODE_800x600_RGB8 ||
      _mode == DC1394_VIDEO_MODE_1024x768_RGB8/* ||
      _mode == DC1394_VIDEO_MODE_FORMAT7_1*/)
    dc1394_convert_to_MONO8(camera_buffer, _buffer, getSourceWidth(), getSourceHeight(),
                            0, DC1394_COLOR_CODING_RGB8, 8);
  else
    memcpy(_buffer, camera_buffer, getSourcePixelCount()*pixelSizeOfMode(_mode));

  if (_mode == DC1394_VIDEO_MODE_640x480_MONO8 ||
      _mode == DC1394_VIDEO_MODE_800x600_MONO8 ||
      _mode == DC1394_VIDEO_MODE_1024x768_MONO8 ||
      _mode == DC1394_VIDEO_MODE_FORMAT7_1)
    dc1394_convert_to_RGB8(_buffer, (unsigned char*)_outRGB8, getSourceWidth(), getSourceHeight(),
                           0, DC1394_COLOR_CODING_MONO8, 8);
  else /*if (_mode == DC1394_VIDEO_MODE_FORMAT7_1)*/
    memcpy(_outRGB8, camera_buffer, getSourcePixelCount()*pixelSizeOfMode(_mode));
  //          for (int i = 0; i < 10; i++)
  //            std::cout << (int)_outRGB8->r << " " << (int)_outRGB8->g << " " << (int)_outRGB8->b << std::endl;
  //          std::cout << std::endl;



  if (bitdepthOfMode(_mode)==8)
  {
    if (_bayerDecoding)
    {
      dc1394_bayer_decoding_8bit(_buffer,(unsigned char*)_outRGB8,getSourceWidth(),getSourceHeight(),_bayerPattern,_bayerMethod);


      int bpc = getPixelCount();
      if (_outputMono)
      {
        for (int i=0;i<bpc;i++)
        {
          _outMono8[i] = (_outRGB8[i].r+_outRGB8[i].g+_outRGB8[i].b)/3;
        }
      }
      if (_outputYUV)
      {
        for (int i=0;i<bpc;i++)
        {
          RGB2YUV(_outRGB8[i].r,_outRGB8[i].g,_outRGB8[i].b,_outY8[i],_outU8[i],_outV8[i]);
        }
      }
    }else{
      if (_outputYUV)
      {
        int bpc = getPixelCount();
        for (int i=0;i<bpc;i++)
        {
          RGB2YUV(_buffer[i],_buffer[i],_buffer[i],_outY8[i],_outU8[i],_outV8[i]);
        }
      }
    }
  }
  else if (bitdepthOfMode(_mode)==16)
  {
    if (_bayerDecoding)
    {
      dc1394_bayer_decoding_16bit((unsigned short*)_buffer,(unsigned short*)_outRGB16,getSourceWidth(),getSourceHeight(),_bayerPattern,_bayerMethod,16);
      int bpc = getPixelCount();
      if (_outputMono)
      {
        for (int i=0;i<bpc;i++)
        {
          _outMono16[i] = (_outRGB16[i].r+_outRGB16[i].g+_outRGB16[i].b)/3;
        }
      }
      if (_outputYUV)
      {
        for (int i=0;i<bpc;i++)
        {
          RGB2YUV(_outRGB16[i].r,_outRGB16[i].g,_outRGB16[i].b,_outY16[i],_outU16[i],_outV16[i]);
        }
      }
    }
  }else{
    dc1394_log_error("invalid bit depth");
  }
}

void Camera::cleanUp()
{
	if (_initialized)
	{
		dc1394_camera_free(_camera);
		dc1394_camera_free_list(_cameras);
	}
	delete [] _buffer;
	delete [] _outMono8;
	delete [] _outRGB8;
	delete [] _outMono16;
	delete [] _outRGB16;
	delete [] _outY8;
	delete [] _outU8;
	delete [] _outV8;
	delete [] _outY16;
	delete [] _outU16;
	delete [] _outV16;
	_initialized = false;
}

////////////////////////////////////////////////////////////////////////////////
//			Getters and setters for instance properties
////////////////////////////////////////////////////////////////////////////////
bool Camera::isRunning()
{
	return _running;
}

/*! return the number of the current frame
*/
int Camera::currentFrame()
{
	return _frameNumber;
}

int Camera::getWidth()
{
	QMutexLocker locker(_propertyMutex);
        if (_bayerDecoding&&(_bayerMethod==DC1394_BAYER_METHOD_DOWNSAMPLE))
		return widthOfMode(_mode)/2;
	else
		return widthOfMode(_mode);
}

int Camera::getHeight()
{
	QMutexLocker locker(_propertyMutex);
	if (_bayerDecoding&&(_bayerMethod==DC1394_BAYER_METHOD_DOWNSAMPLE))
		return heightOfMode(_mode)/2;
	else
		return heightOfMode(_mode);
}

int Camera::getSourceWidth()
{
	QMutexLocker locker(_propertyMutex);
	return widthOfMode(_mode);
}

int Camera::getSourceHeight()
{
	QMutexLocker locker(_propertyMutex);
	return heightOfMode(_mode);
}

int Camera::getPixelCount()
{
	QMutexLocker locker(_propertyMutex);
	if (_bayerDecoding&&(_bayerMethod==DC1394_BAYER_METHOD_DOWNSAMPLE))
		return (widthOfMode(_mode)*heightOfMode(_mode))/4;
	else
		return widthOfMode(_mode)*heightOfMode(_mode);
}

int Camera::getSourcePixelCount()
{
	QMutexLocker locker(_propertyMutex);
	return widthOfMode(_mode)*heightOfMode(_mode);
}

QString Camera::getDeviceFilename()
{
	QMutexLocker locker(_propertyMutex);
	return _deviceFilename;
}

void Camera::setDeviceFilename(QString filename)
{
	QMutexLocker locker(_propertyMutex);
	if (_stopped)
	{
		_deviceFilename = filename;
		if (_initialized)
			initialize();
	}
}

dc1394framerate_t Camera::getFramerate()
{
	QMutexLocker locker(_propertyMutex);
	return _framerate;
}

void Camera::setFramerate(dc1394framerate_t framerate)
{
	QMutexLocker locker(_propertyMutex);
	if (_stopped)
	{
		_framerate = framerate;
		if (_initialized)
		{
			_lastError = dc1394_video_set_framerate(_camera,_framerate);
			DC1394_ERR(_lastError,"Failed to set framerate");
		}
	}
}

dc1394video_mode_t Camera::getMode()
{
	QMutexLocker locker(_propertyMutex);
	return _mode;
}

void Camera::setMode(dc1394video_mode_t mode)
{
  QMutexLocker locker(_propertyMutex);
  if (_stopped)
  {
    if (isValidMode(mode))
    {
      if (_initialized)
      {
        dc1394video_modes_t availableModes;
        _lastError = dc1394_video_get_supported_modes(_camera, &availableModes);

        for (unsigned int i=0; i < availableModes.num; i++)
        {
          if (mode == availableModes.modes[i])
          {
            _mode = mode;
            std::cout << "[1394camera] setting mode " << getVideoModeAsString(mode) << std::endl;
            _lastError = dc1394_video_set_mode(_camera,_mode);
            DC1394_ERR(_lastError,"Failed to set mode");
            return;
          }
        }
        std::cout << "WARNING [1394camera] setMode(): mode " << getVideoModeAsString(mode) <<
            " not supported by camera, falling back to " << getVideoModeAsString(_mode) << std::endl;
        return;
      }
      std::cout << "[1394camera] setMode(): camera not initialized" << std::endl;
      return;
    }
    else
    {
      cout << "[1394camera] mode " << getVideoModeAsString(mode) << " is invalid" << endl;
    }
  }
  else
  {
    std::cout << "[1394camera] setMode(): camera running, not setting mode" << std::endl;
    return;
  }
}

dc1394operation_mode_t Camera::getOperationMode()
{
	QMutexLocker locker(_propertyMutex);
	return _operationMode;
}

void Camera::setOperationMode(dc1394operation_mode_t operationMode)
{
	QMutexLocker locker(_propertyMutex);
	if (_stopped)
	{
		_operationMode = operationMode;
		if (_initialized)
		{
			_lastError = dc1394_video_set_operation_mode(_camera,_operationMode);
			DC1394_ERR(_lastError,"Failed to set operation_mode");
		}
	}
}

dc1394speed_t Camera::getIsoSpeed()
{
	QMutexLocker locker(_propertyMutex);
	return _isoSpeed;
}

void Camera::setIsoSpeed(dc1394speed_t speed)
{
	QMutexLocker locker(_propertyMutex);
	if (_stopped)
	{
		_isoSpeed = speed;
		if (_initialized)
		{
			_lastError = dc1394_video_set_iso_speed(_camera,_isoSpeed);
			DC1394_ERR(_lastError,"Failed to set iso_speed");
		}
	}
}

//latest error encountered
dc1394error_t Camera::getLastError()
{
	QMutexLocker locker(_propertyMutex);
	return _lastError;
}

dc1394bayer_method_t Camera::getBayerMethod()
{
	QMutexLocker locker(_propertyMutex);
	return _bayerMethod;
}

void Camera::setBayerMethod(dc1394bayer_method_t method)
{
	QMutexLocker locker(_propertyMutex);
	if (_stopped)
	{
		_bayerMethod = method;
	}
}

dc1394color_filter_t Camera::getBayerPattern()
{
	QMutexLocker locker(_propertyMutex);
	return _bayerPattern;
}

void Camera::setBayerPattern(dc1394color_filter_t filter)
{
	QMutexLocker locker(_propertyMutex);
	if (_stopped)
	{
		_bayerPattern = filter;
	}
}


bool Camera::getBayerDecoding()
{
	QMutexLocker locker(_propertyMutex);
	return _bayerDecoding;
}

void Camera::setBayerDecoding(bool val)

{
	QMutexLocker locker(_propertyMutex);
	if (_stopped)
	{
		_bayerDecoding = val;
	}
}

bool Camera::getOutputMono()
{
	QMutexLocker locker(_propertyMutex);
	return _outputMono;
}

void Camera::setOutputMono(bool val)
{
	QMutexLocker locker(_propertyMutex);
	if (_stopped)
	{
		_outputMono = val;
	}
}

bool Camera::getOutputRGB()
{
	QMutexLocker locker(_propertyMutex);
	return _outputRGB;
}

void Camera::setOutputRGB(bool val)
{
	QMutexLocker locker(_propertyMutex);
	if (_stopped)
	{
		_outputRGB = val;
	}
}

bool Camera::getOutputYUV()
{
	QMutexLocker locker(_propertyMutex);
	return _outputYUV;
}

void Camera::setOutputYUV(bool val)
{
	QMutexLocker locker(_propertyMutex);
	if (_stopped)
	{
		_outputYUV = val;
	}
}

bool Camera::getHalfYUV()
{
	QMutexLocker locker(_propertyMutex);
	return _halfYUV;
}

void Camera::setHalfYUV(bool val)
{
	QMutexLocker locker(_propertyMutex);
	if (_stopped)
	{
		_halfYUV = val;
	}
}


bool Camera::isMonoMode()
{
	return ::isMonoMode(_mode);
}

int Camera::getBitDepth()
{
	return bitdepthOfMode(_mode);
}

////////////////////////////////////////////////////////////////////////////////
//! Get an unsigned char image
////////////////////////////////////////////////////////////////////////////////
int Camera::getImageRaw(unsigned char* image)
{
	QMutexLocker locker(_bufferMutex);

	int res = _frameNumber;

	memcpy(image, _buffer, getSourcePixelCount()*pixelSizeOfMode(_mode));

	return res;
}

////////////////////////////////////////////////////////////////////////////////
//! Get a float image
////////////////////////////////////////////////////////////////////////////////
int Camera::getImageFloat(float* image)
{
	QMutexLocker locker(_bufferMutex);

	int res = _frameNumber;
	int pc = getPixelCount();

	if (bitdepthOfMode(_mode)==8)
	{
		if (_bayerDecoding)
		{
			for (int i=0; i< pc; i++)
			{
				image[i] = (float)_outMono8[i]/255.0f;
			}
		}else{
			for (int i=0; i< pc; i++)
			{
				image[i] = (float)_buffer[i]/255.0f;
			}
		}
	}
	else if (bitdepthOfMode(_mode)==16)
	{
		float m = 65536;
		if (_bayerDecoding)
		{
			for (int i=0; i< pc; i++)
			{
				image[i] = ((float)_outMono16[i])/m;
			}
		}else{
			for (int i=0; i< pc; i++)
			{
				unsigned short* pt = (unsigned short*)_buffer;
				image[i] = ((float)pt[i])/m;
			}
		}
	}else{
	}

	return res;
}

////////////////////////////////////////////////////////////////////////////////
//! Get an unsigned char image
////////////////////////////////////////////////////////////////////////////////
int Camera::getImageMono8(unsigned char* image)
{
	QMutexLocker locker(_bufferMutex);

	int res = _frameNumber;
	int pc = getPixelCount();

	if (_bayerDecoding)
	{
		memcpy(image, _outMono8, pc);
	}else{
		memcpy(image, _buffer, pc);
	}

	return res;
}

int Camera::getImageMono16(unsigned short* image)
{
	QMutexLocker locker(_bufferMutex);

	int res = _frameNumber;
	int pc = getPixelCount();

	if (_bayerDecoding)
	{
		memcpy(image, _outMono16, pc);
	}else{
		memcpy(image, _buffer, pc*2);
	}

	return res;
}

////////////////////////////////////////////////////////////////////////////////
//! Get an RGB image
////////////////////////////////////////////////////////////////////////////////
int Camera::getImageRGB8(RGB8* image)
{
	QMutexLocker locker(_bufferMutex);

	int res = _frameNumber;
	int pc = getPixelCount();

	memcpy(image, _outRGB8, pc*sizeof(RGB8));

	return res;
}

int Camera::getImageRGB8(CVD::Rgb<unsigned char>* image)
{
  QMutexLocker locker(_bufferMutex);

  int res = _frameNumber;
  int pc = getPixelCount();

  memcpy(image, _outRGB8, pc*sizeof(RGB8));
//  for (int i = 0; i < 10; i++)
//    std::cout << (int)_outRGB8->r << " " << (int)_outRGB8->g << " " << (int)_outRGB8->b << std::endl;
//  std::cout << std::endl;

  return res;
}

int Camera::getImageRGB16(RGB16* image)
{
	QMutexLocker locker(_bufferMutex);

	int res = _frameNumber;
	int pc = getPixelCount();

	memcpy(image, _outRGB16, pc*sizeof(RGB16));

	return res;
}

////////////////////////////////////////////////////////////////////////////////
//! Get an YUV image
////////////////////////////////////////////////////////////////////////////////
int Camera::getImageYUV8(unsigned char* y,unsigned char* u,unsigned char* v)
{
	QMutexLocker locker(_bufferMutex);

	int res = _frameNumber;
	int pc = getPixelCount();

	memcpy(y, _outY8, pc);
	if (_halfYUV)
	{
		int w = getWidth();
		int w_2 = getWidth()/2;
		int h_2 = getHeight()/2;
		for (int y=0;y<h_2;y++)
			for (int x=0;x<w_2;x++)
			{
				u[x+y*w_2] = (_outU8[(x*2)+(y*2+1)*w] +_outU8[(x*2)+1+y*2*w] +_outU8[(x*2)+y*2*w] +_outU8[(x*2)+1+(y*2+1)*w])/4;
				v[x+y*w_2] = (_outV8[(x*2)+(y*2+1)*w] +_outV8[(x*2)+1+y*2*w] +_outV8[(x*2)+y*2*w] +_outV8[(x*2)+1+(y*2+1)*w])/4;
			}
	}
	else
	{
		memcpy(u, _outU8, pc);
		memcpy(v, _outV8, pc);
	}
	return res;
}

int Camera::getImageYUV16(unsigned short* y,unsigned short* u,unsigned short* v)
{
	QMutexLocker locker(_bufferMutex);

	int res = _frameNumber;
	int pc = getPixelCount();

	memcpy(y, _outY16, pc*2);
	if (_halfYUV)
	{
		int w = getWidth();
		int w_2 = getWidth()/2;
		int h_2 = getHeight()/2;
		for (int y=0;y<h_2;y++)
			for (int x=0;x<w_2;x++)
		{
			u[x+y*w_2] = (_outU16[(x*2)+(y*2+1)*w] +_outU16[(x*2)+1+y*2*w] +_outU16[(x*2)+y*2*w] +_outU16[(x*2)+1+(y*2+1)*w])/4;
			v[x+y*w_2] = (_outV16[(x*2)+(y*2+1)*w] +_outV16[(x*2)+1+y*2*w] +_outV16[(x*2)+y*2*w] +_outV16[(x*2)+1+(y*2+1)*w])/4;
		}
	}
	else
	{
		memcpy(u, _outU16, pc*2);
		memcpy(v, _outV16, pc*2);
	}

	return res;
}

