#ifndef ARITHMETICS_CUH
#define ARITHMETICS_CUH

#ifdef NVCC

#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include "thrust_kernels.cuh"
#include "iucore.h"

namespace iuprivate{
namespace math {
/** Addition to every pixel of a constant value. (can be called in-place)
 * \param src Source image.
 * \param val Value to be added.
 * \param dst Destination image.
 */
template<typename PixelType, class Allocator >
void addC(iu::ImageGpu<PixelType, iuprivate::ImageAllocatorGpu<Allocator> >& src, const PixelType& val,
                        iu::ImageGpu<PixelType, iuprivate::ImageAllocatorGpu<Allocator> >& dst)
{
    thrust::transform(src.begin(),src.end(),thrust::constant_iterator<PixelType>(val),dst.begin(),thrust::plus<PixelType>());
}

template<typename PixelType>
void addC(iu::LinearDeviceMemory<PixelType>& src, const PixelType& val,
          iu::LinearDeviceMemory<PixelType>& dst)
{
    thrust::transform(src.begin(),src.end(),thrust::constant_iterator<PixelType>(val),dst.begin(),thrust::plus<PixelType>());
}

/** Multiplication to every pixel of a constant value. (can be called in-place)
 * \param src Source image.
 * \param val Value to be added.
 * \param[out] dst Destination image.
 */
template<typename PixelType, class Allocator >
void mulC(iu::ImageGpu<PixelType, iuprivate::ImageAllocatorGpu<Allocator> >& src, const PixelType& val,
                        iu::ImageGpu<PixelType, iuprivate::ImageAllocatorGpu<Allocator> >& dst)
{
    thrust::transform(src.begin(),src.end(),thrust::constant_iterator<PixelType>(val),dst.begin(),thrust::multiplies<PixelType>());
}

template<typename PixelType>
void mulC(iu::LinearDeviceMemory<PixelType>& src, const PixelType& val,
                        iu::LinearDeviceMemory<PixelType>& dst)
{
    thrust::transform(src.begin(),src.end(),thrust::constant_iterator<PixelType>(val),dst.begin(),thrust::multiplies<PixelType>());
}

/** Adding an image with an additional weighting factor to another.
 * \param src1 Source image 1.
 * \param src2 Source image 2.
 * \param weight Weighting of image 2 before its added to image 1.
 * \param dst Result image dst=weight1*src1 + weight1*src2.
 */
template<typename PixelType, class Allocator >
void addWeighted(iu::ImageGpu<PixelType, iuprivate::ImageAllocatorGpu<Allocator> >& src1, const PixelType& weight1,
                 iu::ImageGpu<PixelType, iuprivate::ImageAllocatorGpu<Allocator> >& src2, const PixelType& weight2,
                 iu::ImageGpu<PixelType, iuprivate::ImageAllocatorGpu<Allocator> >& dst)
{
    weightedsum_transform_tuple<PixelType> unary_op(weight1,weight2); // transformation operator
    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(src1.begin(), src2.begin())),
                      thrust::make_zip_iterator(thrust::make_tuple(src1.begin(), src2.begin())) + src1.stride()*src1.height(),
                      dst.begin(),
                      unary_op);
}

template<typename PixelType>
void addWeighted(iu::LinearDeviceMemory<PixelType>& src1, const PixelType& weight1,
                 iu::LinearDeviceMemory<PixelType>& src2, const PixelType& weight2,
                 iu::LinearDeviceMemory<PixelType>& dst)
{
    weightedsum_transform_tuple<PixelType> unary_op(weight1,weight2); // transformation operator
    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(src1.begin(), src2.begin())),
                      thrust::make_zip_iterator(thrust::make_tuple(src1.begin(), src2.begin())) + src1.length(),
                      dst.begin(),
                      unary_op);
}

/** Multiplying an image to another.
 * \param src1 Source image 1.
 * \param src2 Source image 2.
 * \param dst Result image dst=src1*src2.
 */
template<typename PixelType, class Allocator >
void mul(iu::ImageGpu<PixelType, iuprivate::ImageAllocatorGpu<Allocator> >& src1,
         iu::ImageGpu<PixelType, iuprivate::ImageAllocatorGpu<Allocator> >& src2,
         iu::ImageGpu<PixelType, iuprivate::ImageAllocatorGpu<Allocator> >& dst)
{
    thrust::transform(src1.begin(), src1.end(), src2.begin(), dst.begin(), thrust::multiplies<PixelType>());
}

template<typename PixelType>
void mul(iu::LinearDeviceMemory<PixelType>& src1,
         iu::LinearDeviceMemory<PixelType>& src2,
         iu::LinearDeviceMemory<PixelType>& dst)
{
    thrust::transform(src1.begin(), src1.end(), src2.begin(), dst.begin(), thrust::multiplies<PixelType>());
}

}//namespace math
}//namespace iuprivate
#endif //NVCC
#endif // ARITHMETICS_CUH
