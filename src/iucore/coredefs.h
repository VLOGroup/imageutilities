
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

/** @brief Exceptions with additional error information
* @ingroup UTILS
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

/** Interpolation types. */
typedef enum
{
  IU_INTERPOLATE_NEAREST, /**< nearest neighbour interpolation. */
  IU_INTERPOLATE_LINEAR /**< linear interpolation. */
} IuInterpolationType;

/** @brief 3D size information
 * This struct contains width, height, depth and some helper functions to define a 3D size.
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

#endif // IU_COREDEFS_H

