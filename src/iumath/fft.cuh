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

/** \brief (batched) nd real->complex fft for linear memory types.
 *
 *  This function cannot be called in-place!
 *  This template will be enabled for float-to-float2 and
 *  double-to-double2 fft only and equally-sized input and output
 *  @param src Source linear memory
 *  @param dst Destination linear memory. The width of the
 *      destination memory has to be half the width of the source memory.
 *  @param scale_sqrt Boolean to choose whether the result will be
 *      scaled by sqrt(1/(\Pi_{i=0}^FFTDim in.size[i])) or
 *      leaved untouched (=default).
 */
template<class InputType, class OutputType, unsigned int FFTDim>
typename std::enable_if<(std::is_same<typename InputType::pixel_type, float>::value ||
                         std::is_same<typename InputType::pixel_type, double>::value) &&
                         InputType::ndim == OutputType::ndim, void>::type
                         fft(const InputType& src, OutputType& dst, bool scale_sqrt=false)
{
  iu::Size<InputType::ndim> halfsize = src.size();
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

  Plan<InputType, OutputType, FFTDim> plan(src.size());
  plan.exec(src.data(), dst.data());

  if (scale_sqrt)
  {
    double normalization = 1;
    for (int d = 0; d < FFTDim; ++d)
      normalization *= src.size()[d];
    typename OutputType::pixel_type scale =
        iu::type_trait<typename OutputType::pixel_type>::make_complex(
        sqrt(1.0/normalization));
    iuprivate::math::mulC(dst, scale, dst);
  }
}

/** \brief (batched) nd complex->complex fft for linear memory types.
 *
 *  This function cannot be called in-place!
 *  This template will be enabled for float2-to-float2 and
 *  double2-to-double2 fft only and equally-sized input and output
 *  @param src Source linear memory
 *  @param dst Destination linear memory.
 *  @param scale_sqrt Boolean to choose whether the result will be
 *      scaled by sqrt(1/(\Pi_{i=0}^FFTDim in.size[i])) or
 *      leaved untouched (=default).
 */
template<class InputType, unsigned int FFTDim>
void fft(const InputType& src, InputType& dst, bool scale_sqrt=false)
{
  IU_SIZE_CHECK(&src, &dst);
  Plan<InputType, InputType, FFTDim> plan(src.size());
  plan.exec(src.data(), dst.data(), CUFFT_FORWARD);

  if (scale_sqrt)
  {
    double normalization = 1;
    for (int d = 0; d < FFTDim; ++d)
      normalization *= src.size()[d];
    typename InputType::pixel_type scale =
        iu::type_trait<typename InputType::pixel_type>::make_complex(
        sqrt(1.0/normalization));
    iuprivate::math::mulC(dst, scale, dst);
  }
}

/** \brief (batched) nd complex->real ifft for linear memory types.
 *
 *  This function cannot be called in-place!
 *  This template will be enabled for float-to-float2 and
 *  double-to-double2 ifft only and equally-sized input and output
 *  @param src Source linear memory
 *  @param dst Destination linear memory. The width of the
 *      destination memory has to be half the width of the source memory.
 *  @param scale_sqrt Boolean to choose whether the result will be
 *      scaled by sqrt(1/(\Pi_{i=0}^FFTDim in.size[i])) or
 *      1/(\Pi_{i=0}^FFTDim in.size[i] (=default).
 */
template<class InputType, class OutputType, unsigned int FFTDim>
typename std::enable_if<(std::is_same<typename OutputType::pixel_type, float>::value ||
                         std::is_same<typename OutputType::pixel_type, double>::value) &&
                         InputType::ndim == OutputType::ndim, void>::type
                         ifft(const InputType& src, OutputType& dst, bool scale_sqrt=false)
{
  iu::Size<OutputType::ndim> halfsize = dst.size();
  halfsize[0] = halfsize[0] / 2 + 1;

  if (!(src.size() == halfsize))
  {
    std::stringstream msg;
    msg << "Size mismatch! Size of source (complex) image is "
        << src.size();
    msg << " Size of destination (real) image with half width (" << halfsize << ")"
        << ". ";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  Plan<InputType, OutputType, FFTDim> plan(dst.size());
  plan.exec(src.data(), dst.data());

  double normalization = 1;
  for (int d = 0; d < FFTDim; ++d)
    normalization *= dst.size()[d];
  typename OutputType::pixel_type scale = 1.0/normalization;
  if (scale_sqrt)
    iuprivate::math::mulC(dst, sqrt(scale), dst);
  else
    iuprivate::math::mulC(dst, scale, dst);
}

/** \brief (batched) nd complex->complex ifft for linear memory types.
 *
 *  This function cannot be called in-place!
 *  This template will be enabled for float2-to-float2 and
 *  double2-to-double2 ifft only and equally-sized input and output
 *  @param src Source linear memory
 *  @param dst Destination linear memory.
 *  @param scale_sqrt Boolean to choose whether the result will be
 *      scaled by sqrt(1/(\Pi_{i=0}^FFTDim in.size[i])) or
 *      1/(\Pi_{i=0}^FFTDim in.size[i]) (=default).
 */
template<class InputType, unsigned int FFTDim>
void ifft(const InputType& src, InputType& dst, bool scale_sqrt=false)
{
  IU_SIZE_CHECK(&src, &dst);
  Plan<InputType, InputType, FFTDim> plan(src.size());
  plan.exec(src.data(), dst.data(), CUFFT_INVERSE);

  double normalization = 1;
  for (int d = 0; d < FFTDim; ++d)
    normalization *= src.size()[d];
  double scale = 1.0/normalization;
  if (scale_sqrt)
    iuprivate::math::mulC(dst, iu::type_trait<typename InputType::pixel_type>::make_complex(sqrt(scale)), dst);
  else
    iuprivate::math::mulC(dst, iu::type_trait<typename InputType::pixel_type>::make_complex(scale), dst);
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
template<class InputType>
void fft2c(const InputType& src, InputType& dst, bool scale_sqrt = false)
{
  InputType tmp(src.size());
  ifftshift2(src, dst);
  fft<InputType, 2>(dst, tmp, scale_sqrt);
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
template<class InputType>
void ifft2c(const InputType& src, InputType& dst, bool scale_sqrt = false)
{
  InputType tmp(src.size());
  ifftshift2(src, dst);
  ifft<InputType, 2>(dst, tmp, scale_sqrt);
  fftshift2(tmp, dst);
}

} /* namespace fft */
} /* namespace math */
} /* namespace iuprivate */
