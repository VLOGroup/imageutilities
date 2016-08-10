#pragma once

#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <sstream>

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



