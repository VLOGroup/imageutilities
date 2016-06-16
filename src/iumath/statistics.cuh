///@file statistics.cuh
///@brief statistics functions for CUDA code
///@author Christian Reinbacher <reinbacher@icg.tugraz.at>

#ifndef STATISTICS_CUH
#define STATISTICS_CUH

#ifdef NVCC

#include "iucore.h"
#include "thrust_kernels.cuh"

namespace iuprivate {
namespace math {

/** Finds the minimum and maximum value of an image.
 * \param src Source image [device]
 * \param[out] min Minium value found in the source image.
 * \param[out] max Maximum value found in the source image.
 *
 */
template<typename PixelType, class Allocator >
void minMax(iu::ImageGpu<PixelType, iuprivate::ImageAllocatorGpu<Allocator> >& d_img, PixelType& minval, PixelType& maxval)
{
    typedef thrust::tuple<bool, PixelType, PixelType> result_type;
    result_type init(true, 10e10f, -10e10f); // initial value
    minmax_transform_tuple<int,PixelType> unary_op(d_img.width(), d_img.stride()); // transformation operator
    minmax_reduce_tuple<int,PixelType> binary_op; // reduction operator
    result_type result =
            thrust::transform_reduce(
                thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<int>(0), d_img.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<int>(0), d_img.begin())) + d_img.stride()*d_img.height(),
                unary_op,
                init,
                binary_op);
    minval = thrust::get<1>(result);
    maxval = thrust::get<2>(result);
}

/** Returns the sum of the values of an image.
 * \param src Source image [device]
 * \param[out] min Minium value found in the source image.
 * \param[out] max Maximum value found in the source image.
 *
 */
template<typename PixelType, class Allocator >
void summation(iu::ImageGpu<PixelType, iuprivate::ImageAllocatorGpu<Allocator> >& d_img,PixelType& sum)
{
    typedef thrust::tuple<bool, PixelType> result_type;
    result_type init(true, 0); // initial value
    sum_transform_tuple<int,PixelType> unary_op(d_img.width(), d_img.stride()); // transformation operator
    sum_reduce_tuple<int,PixelType> binary_op; // reduction operator
    result_type result =
            thrust::transform_reduce(
                thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<int>(0), d_img.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<int>(0), d_img.begin())) + d_img.stride()*d_img.height(),
                unary_op,
                init,
                binary_op);
    sum = thrust::get<1>(result);
}

/** Returns the norm of the L1-difference between the image and a reference.
 * \param src Source image [device]
 * \param ref Reference image [device]
 * \param[out] mse MSE value (sum((src-ref)^2)).
 *
 */
template<typename PixelType, class Allocator >
void normDiffL1(iu::ImageGpu<PixelType, iuprivate::ImageAllocatorGpu<Allocator> >& src,
                iu::ImageGpu<PixelType, iuprivate::ImageAllocatorGpu<Allocator> >& ref,PixelType& norm)
{
    typedef thrust::tuple<bool, PixelType> result_type;
    result_type init(true, 0); // initial value
    diffabs_transform_tuple<int,PixelType> unary_op(src.width(), src.stride()); // transformation operator
    sum_reduce_tuple<int,PixelType> binary_op; // reduction operator
    result_type result =
            thrust::transform_reduce(
                thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<int>(0), src.begin(), ref.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<int>(0), src.begin(), ref.begin())) + src.stride()*src.height(),
                unary_op,
                init,
                binary_op);
    norm = sqrt(thrust::get<1>(result));
}

/** Returns the norm of the L1-difference between the image and a reference value.
 * \param src Source image [device]
 * \param ref Reference value
 * \param[out] mse MSE value (sum((src-ref)^2)).
 *
 */
template<typename PixelType, class Allocator >
void normDiffL1(iu::ImageGpu<PixelType, iuprivate::ImageAllocatorGpu<Allocator> >& src,PixelType& ref,PixelType& norm)
{
    typedef thrust::tuple<bool, PixelType> result_type;
    result_type init(true, 0); // initial value
    diffabs_transform_tuple<int,PixelType> unary_op(src.width(), src.stride()); // transformation operator
    sum_reduce_tuple<int,PixelType> binary_op; // reduction operator
    result_type result =
            thrust::transform_reduce(
                thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<int>(0), src.begin(), thrust::constant_iterator<PixelType>(ref))),
                thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<int>(0), src.begin(), thrust::constant_iterator<PixelType>(ref))) + src.stride()*src.height(),
                unary_op,
                init,
                binary_op);
    norm = sqrt(thrust::get<1>(result));
}

/** Returns the norm of the L2-difference between the image and a reference.
 * \param src Source image [device]
 * \param ref Reference image [device]
 * \param[out] mse MSE value (sum((src-ref)^2)).
 *
 */
template<typename PixelType, class Allocator >
void normDiffL2(iu::ImageGpu<PixelType, iuprivate::ImageAllocatorGpu<Allocator> >& src,
                iu::ImageGpu<PixelType, iuprivate::ImageAllocatorGpu<Allocator> >& ref,PixelType& norm)
{
    typedef thrust::tuple<bool, PixelType> result_type;
    result_type init(true, 0); // initial value
    diffsqr_transform_tuple<int,PixelType> unary_op(src.width(), src.stride()); // transformation operator
    sum_reduce_tuple<int,PixelType> binary_op; // reduction operator
    result_type result =
            thrust::transform_reduce(
                thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<int>(0), src.begin(), ref.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<int>(0), src.begin(), ref.begin())) + src.stride()*src.height(),
                unary_op,
                init,
                binary_op);
    norm = sqrt(thrust::get<1>(result));
}

/** Returns the norm of the L2-difference between the image and a reference value.
 * \param src Source image [device]
 * \param ref Reference value
 * \param[out] mse MSE value (sum((src-ref)^2)).
 *
 */
template<typename PixelType, class Allocator >
void normDiffL2(iu::ImageGpu<PixelType, iuprivate::ImageAllocatorGpu<Allocator> >& src,PixelType& ref,PixelType& norm)
{
    typedef thrust::tuple<bool, PixelType> result_type;
    result_type init(true, 0); // initial value
    diffsqr_transform_tuple<int,PixelType> unary_op(src.width(), src.stride()); // transformation operator
    sum_reduce_tuple<int,PixelType> binary_op; // reduction operator
    result_type result =
            thrust::transform_reduce(
                thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<int>(0), src.begin(), thrust::constant_iterator<PixelType>(ref))),
                thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<int>(0), src.begin(), thrust::constant_iterator<PixelType>(ref))) + src.stride()*src.height(),
                unary_op,
                init,
                binary_op);
    norm = sqrt(thrust::get<1>(result));
}

/** Returns the MSE between the image and a reference.
 * \param src Source image [device]
 * \param ref Reference image [device]
 * \param[out] mse MSE value (sum((src-ref)^2)).
 *
 */
template<typename PixelType, class Allocator >
void mse(iu::ImageGpu<PixelType, iuprivate::ImageAllocatorGpu<Allocator> >& src,
         iu::ImageGpu<PixelType, iuprivate::ImageAllocatorGpu<Allocator> >& ref,PixelType& mse)
{
    normDiffL2(src,ref,mse);
    mse=(mse*mse)/(src.width()*src.height());
}

} //namespace math
} // namespace iuprivate
#endif //NVCC
#endif //STATISTICS_CUH
