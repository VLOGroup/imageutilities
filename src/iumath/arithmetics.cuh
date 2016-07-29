#ifndef ARITHMETICS_CUH
#define ARITHMETICS_CUH

#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>

#include "thrust_kernels.cuh"
#include "iucore.h"

namespace iuprivate{
namespace math {
/** Addition to every pixel of a constant value. (can be called in-place)
 * \param[in] src Source image.
 * \param[in] val Value to be added.
 * \param[out] dst Destination image.
 */
template<template<typename, typename > class PitchedMemoryType,
    template<typename> class Allocator, typename PixelType>
void addC(PitchedMemoryType<PixelType, Allocator<PixelType> >& src, const PixelType& val,
          PitchedMemoryType<PixelType,Allocator<PixelType>  >& dst)
{
    thrust::transform(src.begin(),src.end(),thrust::constant_iterator<PixelType>(val),dst.begin(),thrust::plus<PixelType>());
}

template<template<typename> class LinearMemoryType, typename PixelType>
void addC(LinearMemoryType<PixelType>& src, const PixelType& val,
          LinearMemoryType<PixelType>& dst)
{
    thrust::transform(src.begin(),src.end(),thrust::constant_iterator<PixelType>(val),dst.begin(),thrust::plus<PixelType>());
}

/** Multiplication to every pixel of a constant value. (can be called in-place)
 * \param[in] src Source image.
 * \param[in] val Value to be added.
 * \param[out] dst Destination image.
 */
template<template<typename, typename > class PitchedMemoryType,
    template<typename> class Allocator, typename PixelType>
void mulC(PitchedMemoryType<PixelType, Allocator<PixelType> >& src, const PixelType& val,
          PitchedMemoryType<PixelType, Allocator<PixelType> >& dst)
{
    thrust::transform(src.begin(),src.end(),thrust::constant_iterator<PixelType>(val),dst.begin(),thrust::multiplies<PixelType>());
}

template<template<typename> class LinearMemoryType, typename PixelType>
void mulC(LinearMemoryType<PixelType>& src, const PixelType& val,
          LinearMemoryType<PixelType>& dst)
{
    thrust::transform(src.begin(),src.end(),thrust::constant_iterator<PixelType>(val),dst.begin(),thrust::multiplies<PixelType>());
}

/** Adding an image with an additional weighting factor to another.
 * \param[in] src1 Source image 1.
 * \param[in] src2 Source image 2.
 * \param[in] weight Weighting of image 2 before its added to image 1.
 * \param[out] dst Result image dst=weight1*src1 + weight1*src2.
 */
template<template<typename, typename > class PitchedMemoryType,
    template<typename> class Allocator, typename PixelType>
void addWeighted(PitchedMemoryType<PixelType, Allocator<PixelType> >& src1, const PixelType& weight1,
                 PitchedMemoryType<PixelType, Allocator<PixelType> >& src2, const PixelType& weight2,
                 PitchedMemoryType<PixelType, Allocator<PixelType> >& dst)
{
    weightedsum_transform_tuple<PixelType> unary_op(weight1,weight2); // transformation operator
    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(src1.begin(), src2.begin())),
                      thrust::make_zip_iterator(thrust::make_tuple(src1.end(), src2.end())),
                      dst.begin(),
                      unary_op);
}

template<template<typename> class LinearMemoryType, typename PixelType>
void addWeighted(LinearMemoryType<PixelType>& src1, const PixelType& weight1,
                 LinearMemoryType<PixelType>& src2, const PixelType& weight2,
                 LinearMemoryType<PixelType>& dst)
{
    weightedsum_transform_tuple<PixelType> unary_op(weight1,weight2); // transformation operator
    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(src1.begin(), src2.begin())),
                      thrust::make_zip_iterator(thrust::make_tuple(src1.end(), src2.end())),
                      dst.begin(),
                      unary_op);
}

/** Multiplying an image to another.
 * \param[in] src1 Source image 1.
 * \param[in] src2 Source image 2.
 * \param[out] dst Result image dst=src1*src2.
 */
template<template<typename, typename > class PitchedMemoryType,
    template<typename> class Allocator, typename PixelType>
void mul(PitchedMemoryType<PixelType, Allocator<PixelType> >& src1,
         PitchedMemoryType<PixelType, Allocator<PixelType> >& src2,
         PitchedMemoryType<PixelType, Allocator<PixelType> >& dst)
{
    thrust::transform(src1.begin(), src1.end(), src2.begin(), dst.begin(), thrust::multiplies<PixelType>());
}

template<template<typename> class LinearMemoryType, typename PixelType>
void mul(LinearMemoryType<PixelType>& src1,
         LinearMemoryType<PixelType>& src2,
         LinearMemoryType<PixelType>& dst)
{
    thrust::transform(src1.begin(), src1.end(), src2.begin(), dst.begin(), thrust::multiplies<PixelType>());
}

/** Fill memory with a given value.
 * \param dst Destination memory.
 * \param val Value to set
 */

template<template<typename, typename > class PitchedMemoryType,
    template<typename> class Allocator, typename PixelType>
void fill(PitchedMemoryType<PixelType, Allocator<PixelType> >& dst,PixelType& val)
{
    thrust::fill(dst.begin(),dst.end(),val);
}

template<template<typename> class LinearMemoryType, typename PixelType>
void fill(LinearMemoryType<PixelType>& dst,PixelType& val)
{
    thrust::fill(dst.begin(),dst.end(),val);
}

/** Split planes of a two channel image (e.g. complex image)
 * \param[in] src  Combined image (e.g. complex image)
 * \param[out] dst1 First channel (e.g. real part)
 * \param[out] dst2 Second channel (e.g. imaginary part)
 *
 */
template<template<typename, typename > class PitchedMemoryType, template<
    typename > class Allocator, typename CombinedPixelType,
    typename PlanePixelType>
void splitPlanes(
    PitchedMemoryType<CombinedPixelType, Allocator<CombinedPixelType> >& src,
    PitchedMemoryType<PlanePixelType, Allocator<PlanePixelType> >& dst1,
    PitchedMemoryType<PlanePixelType, Allocator<PlanePixelType> >& dst2)
{
  thrust::transform(
      src.begin(),
      src.end(),
      thrust::make_zip_iterator(
          thrust::make_tuple(dst1.begin(), dst2.begin())),
      split_planes2_functor<CombinedPixelType, PlanePixelType>());
}

template<template<typename> class LinearMemoryType, typename CombinedPixelType,
    typename PlanePixelType>
void splitPlanes(
    LinearMemoryType<CombinedPixelType>& src,
    LinearMemoryType<PlanePixelType>& dst1,
    LinearMemoryType<PlanePixelType>& dst2)
{
  thrust::transform(
      src.begin(),
      src.end(),
      thrust::make_zip_iterator(
          thrust::make_tuple(dst1.begin(), dst2.begin())),
      split_planes2_functor<CombinedPixelType, PlanePixelType>());
}

/** Split planes of a three channel image (e.g. rgb image)
 * \param[in] src  Combined image (e.g. rgb image)
 * \param[out] dst1 First channel (e.g. r channel)
 * \param[out] dst2 Second channel (e.g. g channel)
 * \param[out] dst3 Third channel (e.g. b channel)
 *
 */
template<template<typename, typename > class PitchedMemoryType, template<
    typename > class Allocator, typename CombinedPixelType,
    typename PlanePixelType>
void splitPlanes(
    PitchedMemoryType<CombinedPixelType, Allocator<CombinedPixelType> >& src,
    PitchedMemoryType<PlanePixelType, Allocator<PlanePixelType> >& dst1,
    PitchedMemoryType<PlanePixelType, Allocator<PlanePixelType> >& dst2,
    PitchedMemoryType<PlanePixelType, Allocator<PlanePixelType> >& dst3)
{
  thrust::transform(
      src.begin(),
      src.end(),
      thrust::make_zip_iterator(
          thrust::make_tuple(dst1.begin(), dst2.begin(), dst3.begin())),
      split_planes3_functor<CombinedPixelType, PlanePixelType>());
}

template<template<typename> class LinearMemoryType, typename CombinedPixelType,
    typename PlanePixelType>
void splitPlanes(
    LinearMemoryType<CombinedPixelType>& src,
    LinearMemoryType<PlanePixelType>& dst1,
    LinearMemoryType<PlanePixelType>& dst2,
    LinearMemoryType<PlanePixelType>& dst3)
{
  thrust::transform(
      src.begin(),
      src.end(),
      thrust::make_zip_iterator(
          thrust::make_tuple(dst1.begin(), dst2.begin(), dst3.begin())),
      split_planes3_functor<CombinedPixelType, PlanePixelType>());
}

/** Combine planes to a two channel image (e.g. complex image)
 * \param[in] src1 First channel (e.g. real part)
 * \param[in] src2 Second channel (e.g. imaginary part)
 * \param[out] dst Combined image (e.g. complex image)
 *
 */
template<template<typename, typename > class PitchedMemoryType, template<
    typename > class Allocator, typename CombinedPixelType,
    typename PlanePixelType>
void combinePlanes(
    PitchedMemoryType<PlanePixelType, Allocator<PlanePixelType> >& src1,
    PitchedMemoryType<PlanePixelType, Allocator<PlanePixelType> >& src2,
    PitchedMemoryType<CombinedPixelType, Allocator<CombinedPixelType> >& dst)
{
  thrust::transform(
      thrust::make_zip_iterator(
          thrust::make_tuple(src1.begin(), src2.begin())),
      thrust::make_zip_iterator(
          thrust::make_tuple(src1.end(), src2.end())),
      dst.begin(),
      combine_planes2_functor<CombinedPixelType, PlanePixelType>());
}

template<template<typename> class LinearMemoryType, typename CombinedPixelType,
    typename PlanePixelType>
void combinePlanes(
    LinearMemoryType<PlanePixelType>& src1,
    LinearMemoryType<PlanePixelType>& src2,
    LinearMemoryType<CombinedPixelType>& dst)
{
  thrust::transform(
      thrust::make_zip_iterator(
          thrust::make_tuple(src1.begin(), src2.begin())),
      thrust::make_zip_iterator(
          thrust::make_tuple(src1.end(), src2.end())),
      dst.begin(),
      combine_planes2_functor<CombinedPixelType, PlanePixelType>());
}

/** Combine planes to a three channel image (e.g. rgb image)
 * \param[in] src1 First channel (e.g. r channel)
 * \param[in] src2 Second channel (e.g. g channel)
 * \param[in] src3 Third channel (e.g. b channel)
 * \param[out] dst Combined image (e.g. rgb image)
 *
 */
template<template<typename, typename > class PitchedMemoryType, template<
    typename > class Allocator, typename CombinedPixelType,
    typename PlanePixelType>
void combinePlanes(
    PitchedMemoryType<PlanePixelType, Allocator<PlanePixelType> >& src1,
    PitchedMemoryType<PlanePixelType, Allocator<PlanePixelType> >& src2,
    PitchedMemoryType<PlanePixelType, Allocator<PlanePixelType> >& src3,
    PitchedMemoryType<CombinedPixelType, Allocator<CombinedPixelType> >& dst)
{
  thrust::transform(
      thrust::make_zip_iterator(
          thrust::make_tuple(src1.begin(), src2.begin(), src3.begin())),
      thrust::make_zip_iterator(
          thrust::make_tuple(src1.end(), src2.end(), src3.end())),
      dst.begin(),
      combine_planes3_functor<CombinedPixelType, PlanePixelType>());
}

template<template<typename> class LinearMemoryType, typename CombinedPixelType,
    typename PlanePixelType>
void combinePlanes(
    LinearMemoryType<PlanePixelType>& src1,
    LinearMemoryType<PlanePixelType>& src2,
    LinearMemoryType<PlanePixelType>& src3,
    LinearMemoryType<CombinedPixelType>& dst)
{
  thrust::transform(
      thrust::make_zip_iterator(
          thrust::make_tuple(src1.begin(), src2.begin(), src3.begin())),
      thrust::make_zip_iterator(
          thrust::make_tuple(src1.end(), src2.end(), src3.end())),
      dst.begin(),
      combine_planes3_functor<CombinedPixelType, PlanePixelType>());
}


}//namespace math
}//namespace iuprivate

#endif // ARITHMETICS_CUH
