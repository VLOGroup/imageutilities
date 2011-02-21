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
 * Project     : ImageUtilities
 * Module      : Core
 * Class       : IuSize, IuRect
 * Language    : C++
 * Description : Typedefinitions and Macros for ImageUtilities core module.
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IU_COREDEFS_H
#define IU_COREDEFS_H

#include <stdio.h>
#include <assert.h>

/** Basic assert macro
 * This macro should be used to enforce any kind of pre or post conditions.
 * Unlike the C assertion this assert also prints an error/warning as output in release mode.
 * \note The macro is written in such a way that omitting a semicolon after its usage
 * causes a compiler error. The correct way to invoke this macro is:
 * IU_ASSERT(small_value < big_value);
 */
#define IU_ASSERT(C) \
  do { \
  if (!(C)) \
  { \
    assert(C); \
    fprintf(stderr, "\n\n ImageUtilities assertion faild: \n "); \
    fprintf(stderr,"  file:       %s\n",__FILE__); \
    fprintf(stderr,"  function:   %s\n",__FUNCTION__); \
    fprintf(stderr,"  line:       %d\n\n",__LINE__); \
  } \
  } while(false)


#ifdef __CUDACC__ // only include this error check in cuda files (seen by nvcc)


//-----------------------------------------------------------------------------
#define __IU_CHECK_FOR_CUDA_ERRORS_ENABLED__ // enables checking for cuda errors

/** CUDA ERROR HANDLING (CHECK FOR CUDA ERRORS)
 */
#ifdef __IU_CHECK_FOR_CUDA_ERRORS_ENABLED__
#define IU_CHECK_CUDA_ERRORS() \
{ \
  cudaThreadSynchronize(); \
  if (cudaError_t err = cudaGetLastError()) \
  { \
    fprintf(stderr,"\n\nCUDA Error: %s\n",cudaGetErrorString(err)); \
    fprintf(stderr,"  file:       %s\n",__FILE__); \
    fprintf(stderr,"  function:   %s\n",__FUNCTION__); \
    fprintf(stderr,"  line:       %d\n\n",__LINE__); \
    return IU_ERROR; \
  } \
}
#else // __IU_CHECK_FOR_CUDA_ERRORS_ENABLED__
#define IU_CHECK_CUDA_ERRORS() {}
#endif // __IU_CHECK_FOR_CUDA_ERRORS_ENABLED__

#endif // __CUDACC__



/** Error status codes.
 * Negative error codes represent an error.
 * Zero means that everything is ok.
 * Positive error codes represent warnings.
 */
typedef enum
{
  // error
  IU_MEM_COPY_ERROR = -11,
  IU_MEM_ALLOC_ERROR = -10,
  IU_CUDA_ERROR = -3,
  IU_NOT_SUPPORTED_ERROR = -2,
  IU_ERROR = -1,

  // success
  IU_NO_ERROR = 0,
  IU_SUCCESS = 0,

  // warnings
  IU_WARNING = 1

} IuStatus;

/** Interpolation types. */
typedef enum
{
  IU_INTERPOLATE_NEAREST, /**< nearest neighbour interpolation. */
  IU_INTERPOLATE_LINEAR, /**< linear interpolation. */
  IU_INTERPOLATE_CUBIC, /**< cubic interpolation. */
  IU_INTERPOLATE_CUBIC_SPLINE /**< cubic spline interpolation. */
} IuInterpolationType;

/** 2D Size
 * This struct contains width, height and some helper functions to define a 2D size.
 */
struct IuSize
{
  unsigned int width;
  unsigned int height;
  unsigned int depth;

  IuSize() :
      width(0), height(0), depth(0)
  {
  }

  IuSize(unsigned int _width, unsigned int _height, unsigned int _depth = 1) :
      width(_width), height(_height), depth(_depth)
  {
  }

  IuSize(const IuSize& from) :
      width(from.width), height(from.height), depth(from.depth)
  {
  }

  IuSize& operator= (const IuSize& from)
  {
//    if(from == *this)
//      return *this;

    this->width = from.width;
    this->height = from.height;
    this->depth = from.depth;
    return *this;
  }

};

inline bool operator==(const IuSize& lhs, const IuSize& rhs)
{
  return ((lhs.width == rhs.width) && (lhs.height == rhs.height) && (lhs.depth == rhs.depth));
}

inline bool operator!=(const IuSize& lhs, const IuSize& rhs)
{
  return ((lhs.width != rhs.width) || (lhs.height != rhs.height) || (lhs.depth != rhs.depth));
}


/** 2D Rectangle
 * This struct contains cordinates of upper left corner and its size in pixels.
 */
struct IuRect
{
  int x;       //!< x-coord of the upper left corner
  int y;       //!< x-coord of the upper left corner
  unsigned int width;   //!< width of the rectangle
  unsigned int height;  //!< width of the rectangle

  IuRect() :
      x(0), y(0), width(0), height(0)
  {
  }

  IuRect(int _x, int _y, unsigned int _width, unsigned int _height) :
      x(_x), y(_y), width(_width), height(_height)
  {
  }

  IuRect(const IuRect& from) :
      x(from.x), y(from.y), width(from.width), height(from.height)
  {
  }

  IuRect& operator= (const IuRect& from)
  {
//    if (from == *this)
//      return *this;

    this->x = from.x;
    this->y = from.y;
    this->width = from.width;
    this->height = from.height;

    return *this;
  }

  IuRect(const IuSize& from) :
      x(0), y(0), width(from.width), height(from.height)
  {
  }

  IuRect& operator= (const IuSize& from)
  {
    this->x = 0;
    this->y = 0;
    this->width = from.width;
    this->height = from.height;

    return *this;
  }

  IuSize size()
  {
    return IuSize(this->width, this->height);
  }

};

inline bool operator==(const IuRect& a, const IuRect& b)
{
  return ((a.x == b.x) && (a.y == b.y) &&
          (a.width == b.width) && (a.height == b.height));
}

inline bool operator!=(const IuRect& a, const IuRect& b)
{
  return ((a.x != b.x) || (a.y != b.y) ||
          (a.width != b.width) || (a.height != b.height));
}

/** 3D Cube
 * This struct contains cordinates of upper left corner and its size in pixels.
 */
struct IuCube
{
  int x;       //!< x-coord of the upper left corner
  int y;       //!< y-coord of the upper left corner
  int z;       //!< z-coord of the upper left corner
  unsigned int width;   //!< width of the rectangle
  unsigned int height;  //!< height of the rectangle
  unsigned int depth;  //!< depth of the rectangle

  IuCube() :
      x(0), y(0), z(0), width(0), height(0), depth(0)
  {
  }

  IuCube(int _x, int _y, int _z, unsigned int _width, unsigned int _height, unsigned int _depth) :
      x(_x), y(_y), z(_z), width(_width), height(_height), depth(_depth)
  {
  }

  IuCube(const IuCube& from) :
      x(from.x), y(from.y), z(from.z), width(from.width), height(from.height), depth(from.depth)
  {
  }

  IuCube& operator= (const IuCube& from)
  {
//    if (from == *this)
//      return *this;

    this->x = from.x;
    this->y = from.y;
    this->z = from.z;
    this->width = from.width;
    this->height = from.height;
    this->depth = from.depth;

    return *this;
  }

  IuCube(const IuSize& from) :
      x(0), y(0), z(0), width(from.width), height(from.height), depth(from.depth)
  {
  }

  IuCube& operator= (const IuSize& from)
  {
    this->x = 0;
    this->y = 0;
    this->z = 0;
    this->width = from.width;
    this->height = from.height;
    this->depth = from.depth;

    return *this;
  }

};

#endif // IU_COREDEFS_H

