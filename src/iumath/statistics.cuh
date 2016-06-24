///@file statistics.cuh
///@brief statistics functions for CUDA code
///@author Christian Reinbacher <reinbacher@icg.tugraz.at>

#ifndef STATISTICS_CUH
#define STATISTICS_CUH

#include "iucore.h"
#include "thrust_kernels.cuh"
#include <thrust/extrema.h>

namespace iuprivate {
namespace math {

/** Finds the minimum and maximum value of an image.
 * \param[in] src Source image [device]
 * \param[out] min Minium value found in the source image.
 * \param[out] max Maximum value found in the source image.
 *
 */
template<template<typename, typename > class PitchedMemoryType, template<
    typename > class Allocator, typename PixelType>
void minMax(PitchedMemoryType<PixelType, Allocator<PixelType> >& d_img,
            PixelType& minval, PixelType& maxval)
{
  typedef thrust::tuple<bool, PixelType, PixelType> result_type;
  result_type init(true, 10e10f, -10e10f);  // initial value
  minmax_transform_tuple<PixelType, int> unary_op(d_img.width(),
                                                  d_img.stride());  // transformation operator
  minmax_reduce_tuple<PixelType, int> binary_op;  // reduction operator
  result_type result = thrust::transform_reduce(
      thrust::make_zip_iterator(
          thrust::make_tuple(d_img.begin(),thrust::counting_iterator<int>(0))),
      thrust::make_zip_iterator(
          thrust::make_tuple(d_img.end(),thrust::counting_iterator<int>(0))),
      unary_op, init, binary_op);
  minval = thrust::get<1>(result);
  maxval = thrust::get<2>(result);
}

template<typename PixelType>
void minMax(iu::LinearDeviceMemory<PixelType>& in, PixelType& minval,
            PixelType& maxval, unsigned int& minidx, unsigned int& maxidx)
{
  typedef thrust::pair<thrust::device_ptr<PixelType>,
      thrust::device_ptr<PixelType> > result_type;
  result_type result = thrust::minmax_element(in.begin(), in.end());
  minval = *result.first;
  maxval = *result.second;
  minidx = result.first - in.begin();
  maxidx = result.second - in.begin();
}

template<typename PixelType>
void minMax(iu::LinearHostMemory<PixelType>& in, PixelType& minval,
            PixelType& maxval, unsigned int& minidx, unsigned int& maxidx)
{
  typedef thrust::pair<thrust::pointer<PixelType, thrust::host_system_tag>,
      thrust::pointer<PixelType, thrust::host_system_tag> > result_type;
  result_type result = thrust::minmax_element(in.begin(), in.end());
  minval = *result.first;
  maxval = *result.second;
  minidx = result.first - in.begin();
  maxidx = result.second - in.begin();
}
/** Returns the sum of the values of an image.
 * \param[in] src Source image [device]
 * \param[out] min Minium value found in the source image.
 * \param[out] max Maximum value found in the source image.
 *
 */
template<template<typename, typename > class PitchedMemoryType, template<
    typename > class Allocator, typename PixelType>
void summation(PitchedMemoryType<PixelType, Allocator<PixelType> >& d_img,
               PixelType initval, PixelType& sum)
{
  typedef thrust::tuple<bool, PixelType> result_type;
  result_type init(true, initval);  // initial value
  sum_transform_tuple<PixelType, int> unary_op(d_img.width(), d_img.stride());  // transformation operator
  sum_reduce_tuple<PixelType, int> binary_op;  // reduction operator
  result_type result = thrust::transform_reduce(
      thrust::make_zip_iterator(
          thrust::make_tuple(d_img.begin(),thrust::counting_iterator<int>(0))),
      thrust::make_zip_iterator(
          thrust::make_tuple(d_img.end(),thrust::counting_iterator<int>(0))),
      unary_op, init, binary_op);
  sum = thrust::get<1>(result);
}

template<template<typename > class LinearMemoryType, typename PixelType>
void summation(LinearMemoryType<PixelType>& d_img, PixelType init,
               PixelType& sum)
{
  sum = thrust::reduce(d_img.begin(), d_img.end(), init,
                       thrust::plus<PixelType>());
}

/** Calculate the L1-Norm \f$ \sum\limits_{i=1}^N \vert x_i - y_i \vert \f$
 *  where \f$ N \f$ is the total number of pixels.
 * \param[in] src Source array \f$ x \f$
 * \param[in] ref Reference array \f$ y \f$
 * \param[out] norm Resulting norm
 */
template<template<typename, typename > class PitchedMemoryType, template<
    typename > class Allocator, typename PixelType>
void normDiffL1(PitchedMemoryType<PixelType, Allocator<PixelType> >& src,
                PitchedMemoryType<PixelType, Allocator<PixelType> >& ref,
                PixelType& norm)
{
  typedef thrust::tuple<bool, PixelType> result_type;
  result_type init(true, 0);  // initial value
  diffabs_transform_tuple<PixelType, int> unary_op(src.width(), src.stride());  // transformation operator
  sum_reduce_tuple<PixelType, int> binary_op;  // reduction operator
  result_type result = thrust::transform_reduce(
      thrust::make_zip_iterator(
          thrust::make_tuple(src.begin(),
                             ref.begin(),
                             thrust::counting_iterator<int>(0))),
      thrust::make_zip_iterator(
          thrust::make_tuple(src.end(),
                             ref.end(),
                             thrust::counting_iterator<int>(0))),
      unary_op, init, binary_op);
  norm = thrust::get<1>(result);
}

/** Calculate the L1-Norm \f$ \sum\limits_{i=1}^N \vert x_i - y \vert \f$
 *  where \f$ N \f$ is the total number of pixels.
 * \param[in] src Source array \f$ x \f$
 * \param[in] ref Reference value \f$ y \f$
 * \param[out] norm Resulting norm
 */
template<template<typename, typename > class PitchedMemoryType, template<
    typename > class Allocator, typename PixelType>
void normDiffL1(PitchedMemoryType<PixelType, Allocator<PixelType> >& src,
                PixelType& ref, PixelType& norm)
{
  typedef thrust::tuple<bool, PixelType> result_type;
  result_type init(true, 0);  // initial value
  diffabs_transform_tuple<PixelType, int> unary_op(src.width(), src.stride());  // transformation operator
  sum_reduce_tuple<PixelType, int> binary_op;  // reduction operator
  result_type result = thrust::transform_reduce(
      thrust::make_zip_iterator(
          thrust::make_tuple(src.begin(),
                             thrust::constant_iterator<PixelType>(ref),
                             thrust::counting_iterator<int>(0))),
      thrust::make_zip_iterator(
          thrust::make_tuple(src.end(),
                             thrust::constant_iterator<PixelType>(ref),
                             thrust::counting_iterator<int>(0))),
      unary_op, init, binary_op);
  norm = thrust::get<1>(result);
}

/** Calculate the L2-Norm \f$ \sqrt{\sum\limits_{i=1}^N ( x_i - y_i )^2}\f$
 *  where \f$ N \f$ is the total number of pixels.
 * \param[in] src Source array \f$ x \f$
 * \param[in] ref Reference array \f$ y \f$
 * \param[out] norm Resulting norm
 */
template<template<typename, typename > class PitchedMemoryType, template<
    typename > class Allocator, typename PixelType>
void normDiffL2(PitchedMemoryType<PixelType, Allocator<PixelType> >& src,
                PitchedMemoryType<PixelType, Allocator<PixelType> >& ref,
                PixelType& norm)
{
  typedef thrust::tuple<bool, PixelType> result_type;
  result_type init(true, 0);  // initial value
  diffsqr_transform_tuple<PixelType, int> unary_op(src.width(), src.stride());  // transformation operator
  sum_reduce_tuple<PixelType, int> binary_op;  // reduction operator
  result_type result = thrust::transform_reduce(
      thrust::make_zip_iterator(
          thrust::make_tuple(src.begin(),
                             ref.begin(),
                             thrust::counting_iterator<int>(0))),
      thrust::make_zip_iterator(
          thrust::make_tuple(src.end(),
                             ref.end(),
                             thrust::counting_iterator<int>(0))),
      unary_op, init, binary_op);
  norm = sqrt(thrust::get<1>(result));
}

/** Calculate the L2-Norm \f$ \sqrt{\sum\limits_{i=1}^N ( x_i - y )^2}\f$
 *  where \f$ N \f$ is the total number of pixels.
 * \param[in] src Source array \f$ x \f$
 * \param[in] ref Reference value \f$ y \f$
 * \param[out] norm Resulting norm
 */
template<template<typename, typename > class PitchedMemoryType, template<
    typename > class Allocator, typename PixelType>
void normDiffL2(PitchedMemoryType<PixelType, Allocator<PixelType> >& src,
                PixelType& ref, PixelType& norm)
{
  typedef thrust::tuple<bool, PixelType> result_type;
  result_type init(true, 0);  // initial value
  diffsqr_transform_tuple<PixelType, int> unary_op(src.width(), src.stride());  // transformation operator
  sum_reduce_tuple<PixelType, int> binary_op;  // reduction operator
  result_type result = thrust::transform_reduce(
      thrust::make_zip_iterator(
          thrust::make_tuple(src.begin(),
                             thrust::constant_iterator<PixelType>(ref),
                             thrust::counting_iterator<int>(0))),
      thrust::make_zip_iterator(
          thrust::make_tuple(src.end(),
                             thrust::constant_iterator<PixelType>(ref),
                             thrust::counting_iterator<int>(0))),
      unary_op, init, binary_op);
  norm = sqrt(thrust::get<1>(result));
}

/** Calculate the mean-squared error (MSE) \f$ \frac{\sum\limits_{i=1}^N ( x_i - y_i )^2}{N}\f$
 *  where \f$ N \f$ is the total number of pixels.
 * \param[in] src Source array \f$ x \f$
 * \param[in] ref Reference array  \f$ y \f$
 * \param[out] mse mean-squared error
 */
template<template<typename, typename > class PitchedMemoryType, template<
    typename > class Allocator, typename PixelType>
void mse(PitchedMemoryType<PixelType, Allocator<PixelType> >& src,
         PitchedMemoryType<PixelType, Allocator<PixelType> >& ref,
         PixelType& mse)
{
  normDiffL2(src, ref, mse);
  mse = (mse * mse) / (src.numel());
}

}  //namespace math
}  // namespace iuprivate

#endif //STATISTICS_CUH
