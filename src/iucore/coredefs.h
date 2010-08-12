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

#include <nppdefs.h>
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

/** 2D Size
 * This struct contains width, height and some helper functions to define a 2D size.
 */
struct IuSize
{
  unsigned int width;
  unsigned int height;

  IuSize() :
      width(0), height(0)
  {
  }

  IuSize(unsigned int _width, unsigned int _height) :
      width(_width), height(_height)
  {
  }

  IuSize(const IuSize& from) :
      width(from.width), height(from.height)
  {
  }

  IuSize& operator= (const IuSize& from)
  {
//    if(from == *this)
//      return *this;

    this->width = from.width;
    this->height = from.height;
    return *this;
  }

  // NppiSize wrappers

  IuSize(const NppiSize& from) :
      width(from.width), height(from.height)
  {
  }

  IuSize& operator= (const NppiSize& from)
  {
    this->width = from.width;
    this->height = from.height;
    return *this;
  }

  NppiSize nppiSize()
  {
    NppiSize sz = {this->width, this->height};
    return sz;
  }

};

//bool operator==(const IuSize& lhs, const IuSize& rhs)
//{
//  return ((lhs.width == rhs.width) && (lhs.height == rhs.height));
//}
//
//bool operator!=(const IuSize& lhs, const IuSize& rhs)
//{
//  return ((lhs.width != rhs.width) || (lhs.height != rhs.height));
//}

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


};

//bool operator==(const IuRect& a, const IuRect& b)
//{
//  return ((a.x == b.x) && (a.y == b.y) &&
//          (a.width == b.width) && (a.height == b.height));
//}
//
//bool operator!=(const IuRect& a, const IuRect& b)
//{
//  return ((a.x != b.x) || (a.y != b.y) ||
//          (a.width != b.width) || (a.height != b.height));
//}

#endif // IU_COREDEFS_H
