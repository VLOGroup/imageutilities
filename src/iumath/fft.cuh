#pragma once

#include "../iudefs.h"
#include "../iucutil.h"
#include "fft_kernels.cuh"
#include "fftplan.h"
#include "complex.cuh"
#include "arithmetics.cuh"

#include <type_traits>

#define COMMON_BLOCK_SIZE_X 16
#define COMMON_BLOCK_SIZE_Y 16
#define COMMON_BLOCK_SIZE_Z 4

namespace iuprivate {
namespace math {
namespace fft {

/** \brief 2d fftshift for ImageGpu.
 *
 *  This function cannot be called in-place!
 *  @param src Source image
 *  @param dst Destination image
 */
template<typename PixelType, class Allocator>
void fftshift2(const iu::ImageGpu<PixelType, Allocator>& src,
               iu::ImageGpu<PixelType, Allocator>& dst)
{
  dim3 dimBlock(COMMON_BLOCK_SIZE_X, COMMON_BLOCK_SIZE_Y, 1);
  dim3 dimGrid(iu::divUp(src.size()[0], dimBlock.x),
               iu::divUp(src.size()[1], dimBlock.y));

  fftshift2_kernel<PixelType, Allocator> <<<dimGrid, dimBlock>>>(src, dst);
  IU_CUDA_CHECK;
}

/** \brief Slice-wise 2d fftshift for VolumeGpu.
 *
 *  This function cannot be called in-place!
 *  @param src Source volume
 *  @param dst Destination volume
 */
template<typename PixelType, class Allocator>
void fftshift2(const iu::VolumeGpu<PixelType, Allocator>& src,
               iu::VolumeGpu<PixelType, Allocator>& dst)
{
  dim3 dimBlock(COMMON_BLOCK_SIZE_X, COMMON_BLOCK_SIZE_Y, COMMON_BLOCK_SIZE_Z);
  dim3 dimGrid(iu::divUp(src.size()[0], dimBlock.x),
               iu::divUp(src.size()[1], dimBlock.y),
               iu::divUp(src.size()[2], dimBlock.z));

  fftshift2_kernel<PixelType, Allocator> <<<dimGrid, dimBlock>>>(src, dst);
  IU_CUDA_CHECK;
}

/** \brief 2d fftshift for LinearDeviceMemory with dimension 2.
 *
 *  This function cannot be called in-place!
 *  @param src Source linear memory
 *  @param dst Destination linear memory
 */
template<typename PixelType, unsigned int Ndim, typename ResultType = void>
typename std::enable_if<(Ndim == 2), ResultType>::type fftshift2(
    const iu::LinearDeviceMemory<PixelType, Ndim>& src,
    iu::LinearDeviceMemory<PixelType, Ndim>& dst)
{
  dim3 dimBlock(COMMON_BLOCK_SIZE_X, COMMON_BLOCK_SIZE_Y, 1);
  dim3 dimGrid(iu::divUp(src.size()[0], dimBlock.x),
               iu::divUp(src.size()[1], dimBlock.y));

  fftshift2_kernel<PixelType> <<<dimGrid, dimBlock>>>(src, dst);
  IU_CUDA_CHECK;
}

/** \brief Slice-wise 2d fftshift for LinearDeviceMemory with dimension > 2.
 *
 *  This function cannot be called in-place!
 *  @param src Source linear memory
 *  @param dst Destination linear memory
 */
template<typename PixelType, unsigned int Ndim, typename ResultType = void>
typename std::enable_if<(Ndim > 2), ResultType>::type fftshift2(
    const iu::LinearDeviceMemory<PixelType, Ndim>& src,
    iu::LinearDeviceMemory<PixelType, Ndim>& dst)
{
  dim3 dimBlock(COMMON_BLOCK_SIZE_X, COMMON_BLOCK_SIZE_Y, COMMON_BLOCK_SIZE_Z);
  dim3 dimGrid(
      iu::divUp(src.size()[0], dimBlock.x),
      iu::divUp(src.size()[1], dimBlock.y),
      iu::divUp(src.numel() / (src.size()[0] * src.size()[1]), dimBlock.z));

  fftshift2_kernel<PixelType, Ndim> <<<dimGrid, dimBlock>>>(src, dst);
  IU_CUDA_CHECK;
}

/** \brief 2d ifftshift for ImageGpu.
 *
 *  This function cannot be called in-place!
 *  @param src Source image
 *  @param dst Destination image
 */
template<typename PixelType, class Allocator>
void ifftshift2(const iu::ImageGpu<PixelType, Allocator>& src,
                iu::ImageGpu<PixelType, Allocator>& dst)
{
  dim3 dimBlock(COMMON_BLOCK_SIZE_X, COMMON_BLOCK_SIZE_Y, 1);
  dim3 dimGrid(iu::divUp(src.size()[0], dimBlock.x),
               iu::divUp(src.size()[1], dimBlock.y));

  ifftshift2_kernel<PixelType, Allocator> <<<dimGrid, dimBlock>>>(src, dst);
  IU_CUDA_CHECK;
}

/** \brief Slice-wise 2d ifftshift for VolumeGpu.
 *
 *  This function cannot be called in-place!
 *  @param src Source volume
 *  @param dst Destination volume
 */
template<typename PixelType, class Allocator>
void ifftshift2(const iu::VolumeGpu<PixelType, Allocator>& src,
                iu::VolumeGpu<PixelType, Allocator>& dst)
{
  dim3 dimBlock(COMMON_BLOCK_SIZE_X, COMMON_BLOCK_SIZE_Y, COMMON_BLOCK_SIZE_Z);
  dim3 dimGrid(iu::divUp(src.size()[0], dimBlock.x),
               iu::divUp(src.size()[1], dimBlock.y),
               iu::divUp(src.size()[2], dimBlock.z));

  ifftshift2_kernel<PixelType, Allocator> <<<dimGrid, dimBlock>>>(src, dst);
  IU_CUDA_CHECK;
}

/** \brief 2d ifftshift for LinearDeviceMemory with dimension 2.
 *
 *  This function cannot be called in-place!
 *  @param src Source linear memory
 *  @param dst Destination linear memory
 */
template<typename PixelType, unsigned int Ndim, typename ResultType = void>
typename std::enable_if<(Ndim == 2), ResultType>::type ifftshift2(
    const iu::LinearDeviceMemory<PixelType, Ndim>& src,
    iu::LinearDeviceMemory<PixelType, Ndim>& dst)
{
  dim3 dimBlock(COMMON_BLOCK_SIZE_X, COMMON_BLOCK_SIZE_Y, 1);
  dim3 dimGrid(iu::divUp(src.size()[0], dimBlock.x),
               iu::divUp(src.size()[1], dimBlock.y));

  ifftshift2_kernel<PixelType> <<<dimGrid, dimBlock>>>(src, dst);
  IU_CUDA_CHECK;
}

/** \brief Slice-wise 2d ifftshift for LinearDeviceMemory with dimension > 2.
 *
 *  This function cannot be called in-place!
 *  @param src Source linear memory
 *  @param dst Destination linear memory
 */
template<typename PixelType, unsigned int Ndim, typename ResultType = void>
typename std::enable_if<(Ndim > 2), ResultType>::type ifftshift2(
    const iu::LinearDeviceMemory<PixelType, Ndim>& src,
    iu::LinearDeviceMemory<PixelType, Ndim>& dst)
{
  dim3 dimBlock(COMMON_BLOCK_SIZE_X, COMMON_BLOCK_SIZE_Y, COMMON_BLOCK_SIZE_Z);
  dim3 dimGrid(
      iu::divUp(src.size()[0], dimBlock.x),
      iu::divUp(src.size()[1], dimBlock.y),
      iu::divUp(src.numel() / (src.size()[0] * src.size()[1]), dimBlock.z));

  ifftshift2_kernel<PixelType, Ndim> <<<dimGrid, dimBlock>>>(src, dst);
  IU_CUDA_CHECK;
}

/** \brief batched 1d real-to-complex fft for linear memory types.
 *
 *  This function cannot be called in-place!
 *  This template will be enabled for float-to-float2 and
 *  double-to-double2 fft only!
 *  The 1d fft will be executed line-wise for linear memory with
 *      dimension Ndim > 1,
 *  @param src Source linear memory
 *  @param dst Destination linear memory. The width of the
 *      destination memory has to be half the width of the source memory.
 *  @param scale_sqrt Boolean to choose whether the result will be
 *      scaled by sqrt(1/(width*height)) or leaved untouched (=default).
 */
template<template<typename, unsigned int> class LinearMemoryType,
    typename InputType, typename OutputType, unsigned int Ndim>
typename std::enable_if<
    std::is_same<InputType, float>::value
        || std::is_same<InputType, double>::value, void>::type fft1(
    const LinearMemoryType<InputType, Ndim>& src,
    LinearMemoryType<OutputType, Ndim>& dst, bool scale_sqrt = false)
{
  iu::Size<Ndim> halfsize = src.size();
  halfsize[0] = halfsize[0] / 2 + 1;

  if (!(dst.size() == halfsize))
  {
    std::stringstream msg;
    msg << "Size mismatch! Size of destination (complex) image is "
        << dst.size();
    msg << " Size of source (real) image with half width (" << halfsize << ")"
        << ". ";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  Plan<InputType, OutputType, 1> plan(src.size());
  plan.exec(src.data(), dst.data());

  if (scale_sqrt)
  {
    OutputType scale = iu::type_trait<OutputType>::make_complex(
        sqrt(
            static_cast<double>(1)
                / static_cast<double>(src.size()[0])));
    iuprivate::math::mulC(dst, scale, dst);
  }
}

/** \brief 2d complex-to-complex fft for pitched memory types.
 *
 *  This function cannot be called in-place!
 *  The 2d fft will be executed slice-wise for VolumeGpu.
 *  @param src Source pitched memory
 *  @param dst Destination pitched memory
 *  @param scale_sqrt Boolean to choose whether the result will be
 *      scaled by sqrt(1/(width*height)) or leaved untouched (=default).
 */
template<template<typename, typename > class PitchedMemoryType, template<
    typename > class Allocator, typename ComplexType>
void fft2(const PitchedMemoryType<ComplexType, Allocator<ComplexType> >& src,
          PitchedMemoryType<ComplexType, Allocator<ComplexType> >& dst,
          bool scale_sqrt = false)
{
  IU_SIZE_CHECK(&src, &dst);
  Plan<ComplexType, ComplexType, 2> plan(src.size(), src.stride(),
                                         dst.stride());
  plan.exec(src.data(), dst.data(), CUFFT_FORWARD);

  if (scale_sqrt)
  {
    ComplexType scale = iu::type_trait<ComplexType>::make_complex(
        sqrt(
            static_cast<double>(1)
                / static_cast<double>(src.size().width * src.size().height)));
    iuprivate::math::mulC(dst, scale, dst);
  }
}

/** \brief 2d complex-to-complex fft for linear memory types.
 *
 *  This function cannot be called in-place!
 *  The 2d fft will be executed slice-wise for a dimension Ndim > 2.
 *  @param src Source linear memory
 *  @param dst Destination linear memory
 *  @param scale_sqrt Boolean to choose whether the result will be
 *      scaled by sqrt(1/(size()[0]*size()[1])) or leaved untouched (=default).
 */
template<template<typename, unsigned int> class LinearMemoryType,
    typename ComplexType, unsigned int Ndim>
void fft2(const LinearMemoryType<ComplexType, Ndim>& src,
          LinearMemoryType<ComplexType, Ndim>& dst, bool scale_sqrt = false)
{
  IU_SIZE_CHECK(&src, &dst);
  Plan<ComplexType, ComplexType, 2> plan(src.size());
  plan.exec(src.data(), dst.data(), CUFFT_FORWARD);

  if (scale_sqrt)
  {
    ComplexType scale = iu::type_trait<ComplexType>::make_complex(
        sqrt(
            static_cast<double>(1)
                / static_cast<double>(src.size()[0] * src.size()[1])));
    iuprivate::math::mulC(dst, scale, dst);
  }
}

/** \brief 2d real-to-complex fft for pitched memory types.
 *
 *  This function cannot be called in-place!
 *  This template will be enabled for float-to-float2 and
 *  double-to-double2 fft2 only!
 *  The 2d fft will be executed slice-wise for VolumeGpu.
 *  @param src Source pitched memory
 *  @param dst Destination pitched memory. The width of the
 *     destination memory has to be half the width of the source memory.
 *  @param scale_sqrt Boolean to choose whether the result will be
 *      scaled by sqrt(1/(width*height)) or leaved untouched (=default).
 */
template<template<typename, typename > class PitchedMemoryType, template<
    typename > class Allocator, typename InputType, typename OutputType>
typename std::enable_if<
    std::is_same<InputType, float>::value
        || std::is_same<InputType, double>::value, void>::type fft2(
    const PitchedMemoryType<InputType, Allocator<InputType> >& src,
    PitchedMemoryType<OutputType, Allocator<OutputType> >& dst,
    bool scale_sqrt = false)
{
  auto halfsize = src.size();
  halfsize.width = halfsize.width / 2 + 1;

  if (!(dst.size() == halfsize))
  {
    std::stringstream msg;
    msg << "Size mismatch! Size of destination (complex) image is "
        << dst.size();
    msg << " Size of source (real) image with half width (" << halfsize << ")"
        << ". ";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  Plan<InputType, OutputType, 2> plan(src.size(), src.stride(), dst.stride());
  plan.exec(src.data(), dst.data());

  if (scale_sqrt)
  {
    OutputType scale = iu::type_trait<OutputType>::make_complex(
        sqrt(
            static_cast<double>(1)
                / static_cast<double>(src.size().width * src.size().height)));
    iuprivate::math::mulC(dst, scale, dst);
  }
}

/** \brief 2d real-to-complex fft for linear memory types.
 *
 *  This function cannot be called in-place!
 *  This template will be enabled for float-to-float2 and
 *  double-to-double2 fft2 only!
 *  The 2d fft will be executed slice-wise for linear memory with
 *      dimension Ndim > 2,
 *  @param src Source linear memory
 *  @param dst Destination linear memory. The width of the
 *      destination memory has to be half the width of the source memory.
 *  @param scale_sqrt Boolean to choose whether the result will be
 *      scaled by sqrt(1/(width*height)) or leaved untouched (=default).
 */
template<template<typename, unsigned int> class LinearMemoryType,
    typename InputType, typename OutputType, unsigned int Ndim>
typename std::enable_if<
    std::is_same<InputType, float>::value
        || std::is_same<InputType, double>::value, void>::type fft2(
    const LinearMemoryType<InputType, Ndim>& src,
    LinearMemoryType<OutputType, Ndim>& dst, bool scale_sqrt = false)
{
  iu::Size<Ndim> halfsize = src.size();
  halfsize[0] = halfsize[0] / 2 + 1;

  if (!(dst.size() == halfsize))
  {
    std::stringstream msg;
    msg << "Size mismatch! Size of destination (complex) image is "
        << dst.size();
    msg << " Size of source (real) image with half width (" << halfsize << ")"
        << ". ";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  Plan<InputType, OutputType, 2> plan(src.size());
  plan.exec(src.data(), dst.data());

  if (scale_sqrt)
  {
    OutputType scale = iu::type_trait<OutputType>::make_complex(
        sqrt(
            static_cast<double>(1)
                / static_cast<double>(src.size()[0] * src.size()[1])));
    iuprivate::math::mulC(dst, scale, dst);
  }
}

///** \brief 2d complex-to-real fft for pitched memory types.
// *
// *  This function cannot be called in-place!
// *  For VolumeGpu the 2d fft will be executed slice-wise.
// *  This template will be enabled for float2-to-float and
// *  double2-to-double fft2 only!
// *  @param src Source pitched memory. The width of the
// *     source memory has to be half the width of the destination memory.
// *  @param dst Destination pitched memory
// *  @param scale_sqrt Boolean to choose whether the result will be
// *      scaled by sqrt(1/(width*height)) or leaved untouched (=default).
// */template<template<typename, typename > class PitchedMemoryType, template<
//    typename > class Allocator, typename InputType, typename OutputType>
//typename std::enable_if<
//    std::is_same<InputType, float2>::value
//        || std::is_same<InputType, double2>::value, void>::type fft2(
//    PitchedMemoryType<InputType, Allocator<InputType> >& src,
//    PitchedMemoryType<OutputType, Allocator<OutputType> >& dst,
//    bool scale_sqrt = false)
//{
//  iu::Size<3> halfsize = dst.size();
//  halfsize.width = halfsize.width / 2 + 1;
//
//  if (!(src.size() == halfsize))
//  {
//    std::stringstream msg;
//    msg << "Size mismatch! Size of source (complex) image is " << src.size()
//        << ". ";
//    msg << " Size of destination (real) image with half width (" << halfsize
//        << ")" << ". ";
//    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
//  }
//
//  Plan<InputType, OutputType, 2> plan(dst.size(), src.stride(), dst.stride());
//  plan.exec(src.data(), dst.data());
//
//  if (scale_sqrt)
//  {
//    OutputType scale = static_cast<OutputType>(sqrt(
//        static_cast<double>(1)
//            / static_cast<double>(dst.size().width * dst.size().height)));
//    iuprivate::math::mulC(dst, scale, dst);
//  }
//}

/** \brief batched 1d complex-to-real ifft for linear memory types.
 *
 *  This function cannot be called in-place!
 *  This template will be enabled for float2-to-float and
 *  double2-to-double ifft only!
 *  The 1d ifft will be executed line-wise for linear memory with
 *      dimension Ndim > 2,
 *  @param src Source linear memory. The width of the
 *      source memory has to be half the width of the destination memory.
 *  @param dst Destination linear memory
 *  @param scale_sqrt Boolean to choose whether the result will be
 *      scaled by sqrt(1/(width*height)) or by 1/(width*height) (=default).
 */
template<template<typename, unsigned int> class LinearMemoryType,
    typename InputType, typename OutputType, unsigned int Ndim>
typename std::enable_if<
    std::is_same<InputType, float2>::value
        || std::is_same<InputType, double2>::value, void>::type ifft1(
    const LinearMemoryType<InputType, Ndim>& src,
    LinearMemoryType<OutputType, Ndim>& dst, bool scale_sqrt = false)
{
  iu::Size<Ndim> halfsize = dst.size();
  halfsize[0] = halfsize[0] / 2 + 1;

  if (!(src.size() == halfsize))
  {
    std::stringstream msg;
    msg << "Size mismatch! Size of source (complex) image is " << src.size()
        << ". ";
    msg << " Size of destination (real) image with half width (" << halfsize
        << ")" << ". ";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  Plan<InputType, OutputType, 1> plan(dst.size());
  plan.exec(src.data(), dst.data());

  if (scale_sqrt)
  {
    OutputType scale = static_cast<OutputType>(sqrt(
        static_cast<double>(1)
            / static_cast<double>(dst.size()[0])));
    iuprivate::math::mulC(dst, scale, dst);
  }
  else
  {
    OutputType scale = static_cast<OutputType>(static_cast<double>(1)
        / static_cast<double>(dst.size()[0]));
    iuprivate::math::mulC(dst, scale, dst);
  }
}

 /** \brief 2d complex-to-complex ifft for pitched memory types.
  *
  *  This function cannot be called in-place!
  *  The 2d ifft will be executed slice-wise for VolumeGpu.
  *  @param src Source pitched memory
  *  @param dst Destination pitched memory
  *  @param scale_sqrt Boolean to choose whether the result will be
  *      scaled by sqrt(1/(width*height)) or by 1/(width*height) (=default).
  */
template<template<typename, typename > class PitchedMemoryType, template<
    typename > class Allocator, typename ComplexType>
void ifft2(const PitchedMemoryType<ComplexType, Allocator<ComplexType> >& src,
           PitchedMemoryType<ComplexType, Allocator<ComplexType> >& dst,
           bool scale_sqrt = false)
{
  IU_SIZE_CHECK(&src, &dst);
  Plan<ComplexType, ComplexType, 2> plan(src.size(), src.stride(),
                                         dst.stride());
  plan.exec(src.data(), dst.data(), CUFFT_INVERSE);

  if (scale_sqrt)
  {
    ComplexType scale = iu::type_trait<ComplexType>::make_complex(
        sqrt(
            static_cast<double>(1)
                / static_cast<double>(src.size().width * src.size().height)));
    iuprivate::math::mulC(dst, scale, dst);
  }
  else
  {
    ComplexType scale = iu::type_trait<ComplexType>::make_complex(
        static_cast<double>(1)
            / static_cast<double>(src.size().width * src.size().height));
    iuprivate::math::mulC(dst, scale, dst);
  }
}

/** \brief 2d complex-to-complex ifft for linear memory types.
 *
 *  This function cannot be called in-place!
 *  The 2d ifft will be executed slice-wise for a dimension Ndim > 2.
 *  @param src Source linear memory
 *  @param dst Destination linear memory
 *  @param scale_sqrt Boolean to choose whether the result will be
 *      scaled by sqrt(1/(size()[0]*size()[1])) or by 1/(width*height) (=default).
 */
template<template<typename, unsigned int> class LinearMemoryType,
    typename ComplexType, unsigned int Ndim>
void ifft2(const LinearMemoryType<ComplexType, Ndim>& src,
           LinearMemoryType<ComplexType, Ndim>& dst, bool scale_sqrt = false)
{
  IU_SIZE_CHECK(&src, &dst);
  Plan<ComplexType, ComplexType, 2> plan(src.size());
  plan.exec(src.data(), dst.data(), CUFFT_INVERSE);

  if (scale_sqrt)
  {
    ComplexType scale = iu::type_trait<ComplexType>::make_complex(
        sqrt(
            static_cast<double>(1)
                / static_cast<double>(src.size()[0] * src.size()[1])));
    iuprivate::math::mulC(dst, scale, dst);
  }
  else
  {
    ComplexType scale = iu::type_trait<ComplexType>::make_complex(
        static_cast<double>(1)
            / static_cast<double>(src.size()[0] * src.size()[1]));
    iuprivate::math::mulC(dst, scale, dst);
  }
}

///** \brief 2d real-to-complex ifft for pitched memory types.
// *
// *  This function cannot be called in-place!
// *  This template will be enabled for float-to-float2 and
// *  double-to-double2 fft2 only!
// *  The 2d ifft will be executed slice-wise for VolumeGpu.
// *  @param src Source pitched memory
// *  @param dst Destination pitched memory. The width of the
// *     destination memory has to be half the width of the source memory.
// *  @param scale_sqrt Boolean to choose whether the result will be
// *      scaled by sqrt(1/(width*height)) or by 1/(width*height) (=default).
// */
//template<template<typename, typename > class PitchedMemoryType, template<
//    typename > class Allocator, typename InputType, typename OutputType>
//typename std::enable_if<
//    std::is_same<InputType, float>::value
//        || std::is_same<InputType, double>::value, void>::type ifft2(
//    PitchedMemoryType<InputType, Allocator<InputType> >& src,
//    PitchedMemoryType<OutputType, Allocator<OutputType> >& dst,
//    bool scale_sqrt = false)
//{
//  iu::Size<3> halfsize = src.size();
//  halfsize.width = halfsize.width / 2 + 1;
//
//  if (!(dst.size() == halfsize))
//  {
//    std::stringstream msg;
//    msg << "Size mismatch! Size of destination (complex) image is "
//        << dst.size();
//    msg << " Size of source (real) image with half width (" << halfsize << ")"
//        << ". ";
//    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
//  }
//
//  Plan<InputType, OutputType, 2> plan(src.size(), src.stride(), dst.stride());
//  plan.exec(src.data(), dst.data());
//
//  if (scale_sqrt)
//  {
//    OutputType scale = iu::VectorType<InputType, 2>::makeComplex(
//        sqrt(
//            static_cast<double>(1)
//                / static_cast<double>(src.size().width * src.size().height)));
//    iuprivate::math::mulC(dst, scale, dst);
//  }
//  else
//  {
//    OutputType scale = iu::VectorType<InputType, 2>::makeComplex(
//        static_cast<double>(1)
//            / static_cast<double>(src.size().width * src.size().height));
//    iuprivate::math::mulC(dst, scale, dst);
//  }
//}

/** \brief 2d complex-to-real ifft for pitched memory types.
 *
 *  This function cannot be called in-place!
 *  This template will be enabled for float2-to-float and
 *  double2-to-double ifft2 only!
 *  The 2d ifft will be executed slice-wise for VolumeGpu.
 *  @param src Source pitched memory. The width of the
 *      source memory has to be half the width of the destination memory.
 *  @param dst Destination pitched memory
 *  @param scale_sqrt Boolean to choose whether the result will be
 *      scaled by sqrt(1/(width*height)) or by 1/(width*height) (=default).
 */template<template<typename, typename > class PitchedMemoryType, template<
    typename > class Allocator, typename InputType, typename OutputType>
typename std::enable_if<
    std::is_same<InputType, float2>::value
        || std::is_same<InputType, double2>::value, void>::type ifft2(
    const PitchedMemoryType<InputType, Allocator<InputType> >& src,
    PitchedMemoryType<OutputType, Allocator<OutputType> >& dst,
    bool scale_sqrt = false)
{
  auto halfsize = dst.size();
  halfsize.width = halfsize.width / 2 + 1;

  if (!(src.size() == halfsize))
  {
    std::stringstream msg;
    msg << "Size mismatch! Size of source (complex) image is " << src.size()
        << ". ";
    msg << " Size of destination (real) image with half width (" << halfsize
        << ")" << ". ";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  Plan<InputType, OutputType, 2> plan(dst.size(), src.stride(), dst.stride());
  plan.exec(src.data(), dst.data());

  if (scale_sqrt)
  {
    OutputType scale = static_cast<OutputType>(sqrt(
        static_cast<double>(1)
            / static_cast<double>(dst.size().width * dst.size().height)));
    iuprivate::math::mulC(dst, scale, dst);
  }
  else
  {
    OutputType scale = static_cast<OutputType>(static_cast<double>(1)
        / static_cast<double>(dst.size().width * dst.size().height));
    iuprivate::math::mulC(dst, scale, dst);
  }
}

/** \brief 2d complex-to-real ifft for linear memory types.
 *
 *  This function cannot be called in-place!
 *  This template will be enabled for float2-to-float and
 *  double2-to-double ifft2 only!
 *  The 2d ifft will be executed slice-wise for linear memory with
 *      dimension Ndim > 2,
 *  @param src Source linear memory. The width of the
 *      source memory has to be half the width of the destination memory.
 *  @param dst Destination linear memory
 *  @param scale_sqrt Boolean to choose whether the result will be
 *      scaled by sqrt(1/(width*height)) or by 1/(width*height) (=default).
 */
template<template<typename, unsigned int> class LinearMemoryType,
    typename InputType, typename OutputType, unsigned int Ndim>
typename std::enable_if<
    std::is_same<InputType, float2>::value
        || std::is_same<InputType, double2>::value, void>::type ifft2(
    const LinearMemoryType<InputType, Ndim>& src,
    LinearMemoryType<OutputType, Ndim>& dst, bool scale_sqrt = false)
{
  iu::Size<Ndim> halfsize = dst.size();
  halfsize[0] = halfsize[0] / 2 + 1;

  if (!(src.size() == halfsize))
  {
    std::stringstream msg;
    msg << "Size mismatch! Size of source (complex) image is " << src.size()
        << ". ";
    msg << " Size of destination (real) image with half width (" << halfsize
        << ")" << ". ";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  Plan<InputType, OutputType, 2> plan(dst.size());
  plan.exec(src.data(), dst.data());

  if (scale_sqrt)
  {
    OutputType scale = static_cast<OutputType>(sqrt(
        static_cast<double>(1)
            / static_cast<double>(dst.size()[0] * dst.size()[1])));
    iuprivate::math::mulC(dst, scale, dst);
  }
  else
  {
    OutputType scale = static_cast<OutputType>(static_cast<double>(1)
        / static_cast<double>(dst.size()[0] * dst.size()[1]));
    iuprivate::math::mulC(dst, scale, dst);
  }
}


/** \brief centered 2d complex-to-complex fft for pitched memory types.
 *
 *  This function cannot be called in-place!
 *  The centered 2d fft will be executed slice-wise for VolumeGpu.
 *  The centered fft performs ifftshift2 followed by fft2 and fftshift2.
 *  @param src Source pitched memory
 *  @param dst Destination pitched memory
 *  @param scale_sqrt Boolean to choose whether the result will be
 *      scaled by sqrt(1/(width*height)) or leaved untouched (=default).
 */
template<template<typename, typename > class PitchedMemoryType, template<
    typename > class Allocator, typename ComplexType>
void fft2c(const PitchedMemoryType<ComplexType, Allocator<ComplexType> >& src,
           PitchedMemoryType<ComplexType, Allocator<ComplexType> >& dst,
           bool scale_sqrt = false)
{
  PitchedMemoryType<ComplexType, Allocator<ComplexType> > tmp(src.size());
  ifftshift2(src, dst);
  fft2(dst, tmp, scale_sqrt);
  fftshift2(tmp, dst);
}

/** \brief centered 2d complex-to-complex fft for linear memory types.
 *
 *  This function cannot be called in-place!
 *  The centered 2d fft will be executed slice-wise for a dimension Ndim > 2.
 *  The centered fft performs ifftshift2 followed by fft2 and fftshift2.
 *  @param src Source linear memory
 *  @param dst Destination linear memory
 *  @param scale_sqrt Boolean to choose whether the result will be
 *      scaled by sqrt(1/(size()[0]*size()[1])) or leaved untouched (=default).
 */
template<template<typename, unsigned int> class LinearMemoryType,
    typename ComplexType, unsigned int Ndim>
void fft2c(const LinearMemoryType<ComplexType, Ndim>& src,
           LinearMemoryType<ComplexType, Ndim>& dst, bool scale_sqrt = false)
{
  LinearMemoryType<ComplexType, Ndim> tmp(src.size());
  ifftshift2(src, dst);
  fft2(dst, tmp, scale_sqrt);
  fftshift2(tmp, dst);
}

/** \brief centered 2d complex-to-complex ifft for pitched memory types.
 *
 *  This function cannot be called in-place!
 *  The centered 2d ifft will be executed slice-wise for VolumeGpu.
 *  The centered ifft performs ifftshift2 followed by ifft2 and fftshift2.
 *  @param src Source pitched memory
 *  @param dst Destination pitched memory
 *  @param scale_sqrt Boolean to choose whether the result will be
 *      scaled by sqrt(1/(width*height)) or by 1/(width*height) (=default).
 */
template<template<typename, typename > class PitchedMemoryType, template<
    typename > class Allocator, typename ComplexType>
void ifft2c(const PitchedMemoryType<ComplexType, Allocator<ComplexType> >& src,
            PitchedMemoryType<ComplexType, Allocator<ComplexType> >& dst,
            bool scale_sqrt = false)
{
  PitchedMemoryType<ComplexType, Allocator<ComplexType> > tmp(src.size());
  ifftshift2(src, dst);
  ifft2(dst, tmp, scale_sqrt);
  fftshift2(tmp, dst);
}

/** \brief centered 2d complex-to-complex ifft for linear memory types.
 *
 *  This function cannot be called in-place!
 *  The centered 2d ifft will be executed slice-wise for a dimension Ndim > 2.
 *  The centered ifft performs ifftshift2 followed by ifft2 and fftshift2.
 *  @param src Source linear memory
 *  @param dst Destination linear memory
 *  @param scale_sqrt Boolean to choose whether the result will be
 *      scaled by sqrt(1/(size()[0]*size()[1])) or by 1/(width*height) (=default).
 */
template<template<typename, unsigned int> class LinearMemoryType,
    typename ComplexType, unsigned int Ndim>
void ifft2c(const LinearMemoryType<ComplexType, Ndim>& src,
            LinearMemoryType<ComplexType, Ndim>& dst, bool scale_sqrt = false)
{
  LinearMemoryType<ComplexType, Ndim> tmp(src.size());
  ifftshift2(src, dst);
  ifft2(dst, tmp, scale_sqrt);
  fftshift2(tmp, dst);
}

} /* namespace fft */
} /* namespace math */
} /* namespace iuprivate */
