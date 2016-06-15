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
#include <iostream>
#include <sstream>
//#include "globaldefs.h"

 /** Basic assert macro
 * This macro should be used to enforce any kind of pre or post conditions.
 * Unlike the C assertion this assert also prints an error/warning as output in release mode.
 * \note The macro is written in such a way that omitting a semicolon after its usage
 * causes a compiler error. The correct way to invoke this macro is:
 * IU_ASSERT(small_value < big_value);
 */
#ifdef DEBUG
#define IU_ASSERT(C) \
  do { \
    if (!(C)) \
    { \
      fprintf(stderr, "%s(%d) : assertion '%s' failed!\n", \
    __FILE__, __LINE__, #C ); \
    abort(); \
    } \
  } while(false)
#else //DEBUG
#define IU_ASSERT(C)
#endif //DEBUG

/** Assertion with additional error information
 */
class IuException : public std::exception
{
public:
  IuException(const std::string& msg, const char* file=NULL, const char* function=NULL, int line=0) throw():
    msg_(msg),
    file_(file),
    function_(function),
    line_(line)
  {
    std::ostringstream out_msg;

    out_msg << "IuException: ";
    out_msg << (msg_.empty() ? "unknown error" : msg_) << "\n";
    out_msg << "      where: ";
    out_msg << (file_.empty() ? "no filename available" : file_) << " | ";
    out_msg << (function_.empty() ? "unknown function" : function_) << ":" << line_;
    msg_ = out_msg.str();
  }

  virtual ~IuException() throw()
  { }

  virtual const char* what() const throw()
  {
    return msg_.c_str();
  }

  std::string msg_;
  std::string file_;
  std::string function_;
  int line_;
}; // class

///** Error status codes.
// * Negative error codes represent an error.
// * Zero means that everything is ok.
// * Positive error codes represent warnings.
// */
//typedef enum
//{
//  // error
//  IU_MEM_COPY_ERROR = -11,
//  IU_MEM_ALLOC_ERROR = -10,
//  IU_CUDA_ERROR = -3,
//  IU_NOT_SUPPORTED_ERROR = -2,
//  IU_ERROR = -1,

//  // success
//  IU_NO_ERROR = 0,
//  IU_SUCCESS = 0,

//  // warnings
//  IU_WARNING = 1

//} IuStatus;

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

  IuSize(unsigned int _width, unsigned int _height, unsigned int _depth = 0) :
      width(_width), height(_height), depth(_depth)
  {
  }

  IuSize(const IuSize& from) :
      width(from.width), height(from.height), depth(from.depth)
  {
  }

  IuSize& operator= (const IuSize& from)
  {
    this->width = from.width;
    this->height = from.height;
    this->depth = from.depth;
    return *this;
  }
  
  
  IuSize operator* (const double factor) const
  {
    return IuSize(static_cast<int>(this->width * factor + 0.5f), static_cast<int>(this->height * factor + 0.5f), static_cast<int>(this->depth * factor + 0.5f));
  }
  
  IuSize operator/ (const double factor) const
  {
    IU_ASSERT(factor != 0);
    double invFactor = 1 / factor;
    return IuSize(this->width, this->height, this->depth) * invFactor;
  }

  friend std::ostream& operator<<(std::ostream & out, IuSize const& size)
  {
    out << "size=[" << size.width << ", " << size.height << ", " << size.depth << "]";
    return out;
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

namespace iu {
/** Round a / b to nearest higher integer value.
 * @param[in] a Numerator
 * @param[in] b Denominator
 * @return a / b rounded up
 */
static inline /*__device__ __host__*/ unsigned int divUp(unsigned int a, unsigned int b)
{
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

} // namespace iu

///** 2D Rectangle
// * This struct contains cordinates of upper left corner and its size in pixels.
// */
//struct IuRect
//{
//  int x;       //!< x-coord of the upper left corner
//  int y;       //!< x-coord of the upper left corner
//  unsigned int width;   //!< width of the rectangle
//  unsigned int height;  //!< width of the rectangle

//  IuRect() :
//      x(0), y(0), width(0), height(0)
//  {
//  }

//  IuRect(int _x, int _y, unsigned int _width, unsigned int _height) :
//      x(_x), y(_y), width(_width), height(_height)
//  {
//  }

//  IuRect(const IuRect& from) :
//      x(from.x), y(from.y), width(from.width), height(from.height)
//  {
//  }

//  IuRect& operator= (const IuRect& from)
//  {
//    this->x = from.x;
//    this->y = from.y;
//    this->width = from.width;
//    this->height = from.height;

//    return *this;
//  }

//  IuRect(const IuSize& from) :
//      x(0), y(0), width(from.width), height(from.height)
//  {
//  }

//  IuRect& operator= (const IuSize& from)
//  {
//    this->x = 0;
//    this->y = 0;
//    this->width = from.width;
//    this->height = from.height;

//    return *this;
//  }

//  IuSize size()
//  {
//    return IuSize(this->width, this->height);
//  }

//  void reset()
//  {
//    this->x = 0;
//    this->y = 0;
//    this->width = 0;
//    this->height = 0;
//  }

//};

//inline bool operator==(const IuRect& a, const IuRect& b)
//{
//  return ((a.x == b.x) && (a.y == b.y) &&
//          (a.width == b.width) && (a.height == b.height));
//}

//inline bool operator!=(const IuRect& a, const IuRect& b)
//{
//  return ((a.x != b.x) || (a.y != b.y) ||
//          (a.width != b.width) || (a.height != b.height));
//}

///** 3D Cube
// * This struct contains cordinates of upper left corner and its size in pixels.
// */
//struct IuCube
//{
//  int x;       //!< x-coord of the upper left corner
//  int y;       //!< y-coord of the upper left corner
//  int z;       //!< z-coord of the upper left corner
//  unsigned int width;   //!< width of the rectangle
//  unsigned int height;  //!< height of the rectangle
//  unsigned int depth;  //!< depth of the rectangle

//  IuCube() :
//      x(0), y(0), z(0), width(0), height(0), depth(0)
//  {
//  }

//  IuCube(int _x, int _y, int _z, unsigned int _width, unsigned int _height, unsigned int _depth) :
//      x(_x), y(_y), z(_z), width(_width), height(_height), depth(_depth)
//  {
//  }

//  IuCube(const IuCube& from) :
//      x(from.x), y(from.y), z(from.z), width(from.width), height(from.height), depth(from.depth)
//  {
//  }

//  IuCube& operator= (const IuCube& from)
//  {
////    if (from == *this)
////      return *this;

//    this->x = from.x;
//    this->y = from.y;
//    this->z = from.z;
//    this->width = from.width;
//    this->height = from.height;
//    this->depth = from.depth;

//    return *this;
//  }

//  IuCube(const IuSize& from) :
//      x(0), y(0), z(0), width(from.width), height(from.height), depth(from.depth)
//  {
//  }

//  IuCube& operator= (const IuSize& from)
//  {
//    this->x = 0;
//    this->y = 0;
//    this->z = 0;
//    this->width = from.width;
//    this->height = from.height;
//    this->depth = from.depth;

//    return *this;
//  }

//};

#endif // IU_COREDEFS_H

