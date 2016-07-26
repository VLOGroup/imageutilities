///@file complex.cuh
///@brief Functions for complex numbers.
///@author Kerstin Hammernik <hammernik@icg.tugraz.at>

#ifndef COMPLEX_CUH
#define COMPLEX_CUH

#include "iucore.h"
#include <thrust/transform.h>

float2 __host__ __device__ make_complex(float x, float y)
{
  return make_float2(x, y);
}

double2 __host__ __device__ make_complex(double x, double y)
{
  return make_double2(x, y);
}

template<typename ComplexType, typename RealType>
struct abs_functor: public thrust::unary_function<ComplexType, RealType>
{
  __host__   __device__ RealType operator()(const ComplexType& t) const
  {
    return sqrt(t.x * t.x + t.y * t.y);
  }
};

template<typename ComplexType, typename RealType>
struct real_functor: public thrust::unary_function<ComplexType, RealType>
{
  __host__   __device__ RealType operator()(const ComplexType& t) const
  {
    return t.x;
  }
};

template<typename ComplexType, typename RealType>
struct imag_functor: public thrust::unary_function<ComplexType, RealType>
{
  __host__   __device__ RealType operator()(const ComplexType& t) const
  {
    return t.y;
  }
};

template<typename ComplexType, typename RealType>
struct phase_functor: public thrust::unary_function<ComplexType, RealType>
{
  __host__   __device__ RealType operator()(const ComplexType& t) const
  {
    return atan(t.y / t.x);
  }
};

template<typename ComplexType, typename RealType>
struct complex_scale_functor: public thrust::unary_function<
    thrust::tuple<ComplexType, RealType>, ComplexType>
{
  typedef thrust::tuple<ComplexType, RealType> InputTuple;
  __host__   __device__ ComplexType operator()(const InputTuple& t) const
  {
    return thrust::get<0>(t) * thrust::get<1>(t);
  }
};

template<typename ComplexType>
struct complex_multiply_functor: public thrust::unary_function<thrust::tuple<ComplexType, ComplexType>,
    ComplexType>
{
  typedef thrust::tuple<ComplexType, ComplexType> InputTuple;
  __host__   __device__ ComplexType operator()(const InputTuple& t) const
  {
    ComplexType a = thrust::get<0>(t);
    ComplexType b = thrust::get<1>(t);
    return make_complex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
  }
};

template<typename ComplexType>
struct complex_multiply_conjugate_functor: public thrust::unary_function<
thrust::tuple<ComplexType, ComplexType>, ComplexType>
{
  typedef thrust::tuple<ComplexType, ComplexType> InputTuple;
  __host__   __device__ ComplexType operator()(const InputTuple& t) const
  {
    ComplexType a = thrust::get<0>(t);
    ComplexType b = thrust::get<1>(t);
    return make_complex(a.x * b.x + a.y * b.y, - a.x * b.y + a.y * b.x);
  }
};

namespace iuprivate {
namespace math {
namespace complex {

/** Compute the absolute image of a complex (two channel) image
 * \param[in] complex_img Complex source image
 * \param[out] abs_img Absolute image
 *
 */
template<template<typename, typename > class PitchedMemoryType, template<
    typename > class Allocator, typename ComplexPixelType,
    typename RealPixelType>
void abs(
    PitchedMemoryType<ComplexPixelType, Allocator<ComplexPixelType> >& complex_img,
    PitchedMemoryType<RealPixelType, Allocator<RealPixelType> >& abs_img)
{
  thrust::transform(complex_img.begin(), complex_img.end(), abs_img.begin(),
                    abs_functor<ComplexPixelType, RealPixelType>());
}

template<template<typename, int> class LinearMemoryType, typename ComplexPixelType,
    typename RealPixelType, int Ndim>
void abs(
    LinearMemoryType<ComplexPixelType, Ndim>& complex_img,
    LinearMemoryType<RealPixelType, Ndim>& abs_img)
{
  thrust::transform(complex_img.begin(), complex_img.end(), abs_img.begin(),
                    abs_functor<ComplexPixelType, RealPixelType>());
}

/** Compute the real image of a complex (two channel) image
 * \param[in] complex_img Complex source image
 * \param[out] real_img Real image
 *
 */
template<template<typename, typename > class PitchedMemoryType, template<
    typename > class Allocator, typename ComplexPixelType,
    typename RealPixelType>
void real(
    PitchedMemoryType<ComplexPixelType, Allocator<ComplexPixelType> >& complex_img,
    PitchedMemoryType<RealPixelType, Allocator<RealPixelType> >& real_img)
{
  thrust::transform(complex_img.begin(), complex_img.end(), real_img.begin(),
                    real_functor<ComplexPixelType, RealPixelType>());
}

template<template<typename, int> class LinearMemoryType, typename ComplexPixelType,
    typename RealPixelType, int Ndim>
void real(
    LinearMemoryType<ComplexPixelType, Ndim>& complex_img,
    LinearMemoryType<RealPixelType, Ndim>& real_img)
{
  thrust::transform(complex_img.begin(), complex_img.end(), real_img.begin(),
                    real_functor<ComplexPixelType, RealPixelType>());
}

/** Compute the imaginary image of a complex (two channel) image
 * \param[in] complex_img Complex source image
 * \param[out] imag_img Imaginary image
 *
 */
template<template<typename, typename > class PitchedMemoryType, template<
    typename > class Allocator, typename ComplexPixelType,
    typename RealPixelType>
void imag(
    PitchedMemoryType<ComplexPixelType, Allocator<ComplexPixelType> >& complex_img,
    PitchedMemoryType<RealPixelType, Allocator<RealPixelType> >& imag_img)
{
  thrust::transform(complex_img.begin(), complex_img.end(), imag_img.begin(),
                    imag_functor<ComplexPixelType, RealPixelType>());
}

template<template<typename, int> class LinearMemoryType, typename ComplexPixelType,
    typename RealPixelType, int Ndim>
void imag(
    LinearMemoryType<ComplexPixelType, Ndim>& complex_img,
    LinearMemoryType<RealPixelType, Ndim>& imag_img)
{
  thrust::transform(complex_img.begin(), complex_img.end(), imag_img.begin(),
                    imag_functor<ComplexPixelType, RealPixelType>());
}

/** Compute the phase of a complex (two channel) image
 * \param[in] complex_img Complex source image
 * \param[out] phase_img Phase image
 *
 */
template<template<typename, typename > class PitchedMemoryType, template<
    typename > class Allocator, typename ComplexPixelType,
    typename RealPixelType>
void phase(
    PitchedMemoryType<ComplexPixelType, Allocator<ComplexPixelType> >& complex_img,
    PitchedMemoryType<RealPixelType, Allocator<RealPixelType> >& phase_img)
{
  thrust::transform(complex_img.begin(), complex_img.end(), phase_img.begin(),
                    phase_functor<ComplexPixelType, RealPixelType>());
}

template<template<typename, int> class LinearMemoryType, typename ComplexPixelType,
    typename RealPixelType, int Ndim>
void phase(
    LinearMemoryType<ComplexPixelType, Ndim>& complex_img,
    LinearMemoryType<RealPixelType, Ndim>& phase_img)
{
  thrust::transform(complex_img.begin(), complex_img.end(), phase_img.begin(),
                    phase_functor<ComplexPixelType, RealPixelType>());
}

/** Multiply two complex (two channel) images
 * \param[in] complex_src1 First complex source image
 * \param[in] complex_src2 Second complex source image
 * \param[out] complex_dst Complex result image
 *
 */
template<template<typename, typename > class PitchedMemoryType, template<
    typename > class Allocator, typename ComplexPixelType>
void multiply(
    PitchedMemoryType<ComplexPixelType, Allocator<ComplexPixelType> >& complex_src1,
    PitchedMemoryType<ComplexPixelType, Allocator<ComplexPixelType> >& complex_src2,
    PitchedMemoryType<ComplexPixelType, Allocator<ComplexPixelType> >& complex_dst)
{
  thrust::transform(
      thrust::make_zip_iterator(
          thrust::make_tuple(complex_src1.begin(), complex_src2.begin())),
      thrust::make_zip_iterator(
          thrust::make_tuple(complex_src1.end(), complex_src2.end())), complex_dst.begin(),
                    complex_multiply_functor<ComplexPixelType>());
}

template<template<typename, int> class LinearMemoryType, typename ComplexPixelType, int Ndim>
void multiply(
    LinearMemoryType<ComplexPixelType, Ndim>& complex_src1,
    LinearMemoryType<ComplexPixelType, Ndim>& complex_src2,
    LinearMemoryType<ComplexPixelType, Ndim>& complex_dst)
{
  thrust::transform(
      thrust::make_zip_iterator(
          thrust::make_tuple(complex_src1.begin(), complex_src2.begin())),
      thrust::make_zip_iterator(
          thrust::make_tuple(complex_src1.end(), complex_src2.end())), complex_dst.begin(),
                    complex_multiply_functor<ComplexPixelType>());
}


/** Multiply one complex (two channel) image with the complex conjugate of a second complex image
 * \param[in] complex_src1 First complex source image
 * \param[in] complex_src2 Second complex source image
 * \param[out] complex_dst Complex result image
 *
 */
template<template<typename, typename > class PitchedMemoryType, template<
    typename > class Allocator, typename ComplexPixelType>
void multiplyConjugate(
    PitchedMemoryType<ComplexPixelType, Allocator<ComplexPixelType> >& complex_src1,
    PitchedMemoryType<ComplexPixelType, Allocator<ComplexPixelType> >& complex_src2,
    PitchedMemoryType<ComplexPixelType, Allocator<ComplexPixelType> >& complex_dst)
{
  thrust::transform(
      thrust::make_zip_iterator(
          thrust::make_tuple(complex_src1.begin(), complex_src2.begin())),
      thrust::make_zip_iterator(
          thrust::make_tuple(complex_src1.end(), complex_src2.end())), complex_dst.begin(),
                    complex_multiply_conjugate_functor<ComplexPixelType>());
}

template<template<typename, int> class LinearMemoryType, typename ComplexPixelType, int Ndim>
void multiplyConjugate(
    LinearMemoryType<ComplexPixelType, Ndim>& complex_src1,
    LinearMemoryType<ComplexPixelType, Ndim>& complex_src2,
    LinearMemoryType<ComplexPixelType, Ndim>& complex_dst)
{
  thrust::transform(
      thrust::make_zip_iterator(
          thrust::make_tuple(complex_src1.begin(), complex_src2.begin())),
      thrust::make_zip_iterator(
          thrust::make_tuple(complex_src1.end(), complex_src2.end())), complex_dst.begin(),
                    complex_multiply_conjugate_functor<ComplexPixelType>());
}

}  // namespace complex
}  //namespace math
}  // namespace iuprivate

#endif //COMPLEX_CUH
