#pragma once

#include "../iuhelpermath.h"
#include "iumathapi.h"

namespace iu {
/** Wrapper class for vector types like float2, double2 to allow
 * more efficient templating especially for fft/ifft methods.
 */
template<typename PixelType>
struct IUMATH_DLLAPI type_trait {};

/** Wrapper class for float2 to allow more efficient templating. */
template<> struct IUMATH_DLLAPI type_trait<float2>
{
  /** Make float2 from a single float value. */
  static inline __host__ __device__ float2 make_complex(float x)
  {
    return make_float2(x, x);
  }
  /** Make float2 from two float values. */
  static inline __host__ __device__ float2 make_complex(float x, float y)
  {
    return make_float2(x, y);
  }

  /** Make float2 from a single float value. */
  static inline __host__ __device__ float2 make(float x)
  {
    return make_float2(x, x);
  }

  /** Compute absolute value of complex number (float2). */
  static inline __host__ __device__  float abs(float2 x)
  {
    return length(x);
  }

  /** Define basic real type (float) */
  typedef float real_type;

  /** Define basic complex type (float2) */
  typedef float2 complex_type;

  /** Return if PixelType is complex. */
  struct is_complex { static const bool value = true; };

  /** Return type name. */
  static const char* name()
  {
      return "float2";
  }
};

/** Wrapper class for double2 to allow more efficient templating. */
template<> struct IUMATH_DLLAPI type_trait<double2>
{
  /** Make double2 from a single double value. */
  static inline __host__ __device__ double2 make_complex(double x)
  {
    return make_double2(x, x);
  }
  /** Make double2 from two double values. */
  static inline __host__ __device__ double2 make_complex(double x, double y)
  {
    return make_double2(x, y);
  }

  /** Make double2 from a single double value. */
  static inline __host__ __device__ double2 make(double x)
  {
    return make_double2(x, x);
  }

  /** Compute absolute value of complex number (double2). */
  static inline __host__ __device__ double abs(double2 x)
  {
    return length(x);
  }

  /** Define basic real type (double) */
  typedef double real_type;

  /** Define basic complex type (double2) */
  typedef double2 complex_type;

  /** Return if PixelType is complex. */
  struct is_complex { static const bool value = true; };

  /** Return type name. */
  static inline const char* name()
  {
      return "double2";
  }
};

/** Wrapper class for float to allow more efficient templating. */
template<> struct IUMATH_DLLAPI type_trait<float>
{
  /** Make float2 from a single float value. */
  static inline __host__ __device__ float2 make_complex(float x)
  {
    return make_float2(x, x);
  }
  /** Make float2 from two float values. */
  static inline __host__ __device__ float2 make_complex(float x, float y)
  {
    return make_float2(x, y);
  }

  /** Make float from a single float value. */
  static inline __host__ __device__ float make(float x)
  {
    return x;
  }

  /** Compute absolute value. */
  static inline __host__ __device__ float abs(float x)
  {
    return fabs(x);
  }

  /** Compute maximum value. */
  static inline __host__ __device__ float max(float x, float y)
  {
      return fmaxf(x, y);
  }

  /** Define basic real type (float) */
  typedef float real_type;

  /** Define basic complex type (float2) */
  typedef float2 complex_type;

  /** Return if PixelType is complex. */
  struct is_complex { static const bool value = false; };

  /** Return type name. */
  static const char* name()
  {
      return "float";
  }
};

/** Wrapper class for double to allow more efficient templating. */
template<> struct IUMATH_DLLAPI type_trait<double>
{
  /** Make float2 from a single float value. */
  static inline __host__ __device__ double2 make_complex(double x)
  {
    return make_double2(x, x);
  }

  /** Make float2 from two float values. */
  static inline __host__ __device__ double2 make_complex(double x, double y)
  {
    return make_double2(x, y);
  }

  /** Make double from a single double value. */
  static inline __host__ __device__ double make(double x)
  {
    return x;
  }

  /** Compute absolute value. */
  static inline __host__ __device__ double abs(double x)
  {
    return fabs(x);
  }

  /** Compute maximum value. */
  static inline __host__ __device__ double max(double x, double y)
  {
      return dmaxd(x, y);
  }

  /** Define basic real type (double) */
  typedef double real_type;

  /** Define basic complex type (double2) */
  typedef double2 complex_type;

  /** Return if PixelType is complex. */
  struct is_complex { static const bool value = true; };

  /** Return type name. */
  static inline __host__ char* name()
  {
      return "double";
  }
};

}

/** Complex multiplication. */
template<typename PixelType>
IUMATH_DLLAPI inline __host__ __device__ typename iu::type_trait<PixelType>::complex_type complex_multiply(
    const typename iu::type_trait<PixelType>::complex_type& src1,
    const typename iu::type_trait<PixelType>::complex_type& src2)
{
  typename iu::type_trait<PixelType>::complex_type dst;
  dst.x = src1.x * src2.x - src1.y * src2.y;
  dst.y = src1.x * src2.y + src1.y * src2.x;
  return dst;
}

/** Complex multiplication with conjugate. */
template<typename PixelType>
IUMATH_DLLAPI inline __host__ __device__ typename iu::type_trait<PixelType>::complex_type complex_multiply_conjugate(
    const typename iu::type_trait<PixelType>::complex_type& src1,
    const typename iu::type_trait<PixelType>::complex_type& src2)
{
  typename iu::type_trait<PixelType>::complex_type dst;
  dst.x = src1.x * src2.x + src1.y * src2.y;
  dst.y = -src1.x * src2.y + src1.y * src2.x;
  return dst;
}
