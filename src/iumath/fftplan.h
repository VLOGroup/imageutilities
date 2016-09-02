/*
 * fftplan.h
 *
 *  Created on: Aug 2, 2016
 *      Author: kerstin
 */

#ifndef FFTPLAN_H_
#define FFTPLAN_H_

#include <cufft.h>
#include "../iudefs.h"
#include "../iucutil.h"

namespace iuprivate {
namespace math {
namespace fft {

typedef float2 fcomplex;
typedef float freal;
typedef double2 dcomplex;
typedef double dreal;

/** \brief Base class for FFT Plan.
 *
 * This class setups the memory layout for fft and executes forward and
 * inverse fft.
 */
template<typename InputPixelType, typename OutputPixelType, unsigned Dim>
class Plan
{
public:
  Plan();

  /** Special Constructor for linear memory layout.
   *  @param size Size/Layout of the linear memory.
   */
  inline Plan(const iu::Size<2> &size);
  inline Plan(const iu::Size<3> &size);
  inline Plan(const iu::Size<4> &size);

  /** Special Constructor for pitched memory layout.
   *  @param size Size of the pitched memory.
   *  @param src_pitch Source pitch
   *  @param dst_pitch Destination pitch
   */
  inline Plan(const iu::Size<2> &size, const int src_pitch,
              const int dst_pitch);
  inline Plan(const iu::Size<3> &size, const int src_pitch,
              const int dst_pitch);

  /** Destructor. */
  virtual ~Plan()
  {
    IU_CUFFT_SAFE_CALL(cufftDestroy(plan_));
  }

  /** FFT plan execution (float): complex -> real
   *  @param[in] input Complex input float buffer.
   *  @param[out] output Real output float buffer.
   */
  inline void exec(fcomplex * input, freal * output)
  {
    IU_CUFFT_SAFE_CALL(cufftExecC2R(plan_, input, output));
  }

  /** FFT plan execution (float): complex -> complex
   *  @param[in] input Complex input float buffer.
   *  @param[out] output Complex output float buffer.
   *  @param[in] direction CUFFT_FORWARD or CUFFT_INVERSE
   */
  inline void exec(fcomplex * input, fcomplex * output, int direction)
  {
    IU_CUFFT_SAFE_CALL(cufftExecC2C(plan_, input, output, direction));
  }

  /** FFT plan execution (float): real -> complex
   *  @param[in] input Real input float buffer.
   *  @param[out] output Complex output float buffer.
   */
  inline void exec(freal * input, fcomplex * output)
  {
    IU_CUFFT_SAFE_CALL(cufftExecR2C(plan_, input, output));
  }

  /** FFT plan execution (double): complex -> real
   *  @param[in] input Complex input double buffer.
   *  @param[out] output Real output double buffer.
   */
  inline void exec(dcomplex * input, dreal * output)
  {
    IU_CUFFT_SAFE_CALL(cufftExecZ2D(plan_, input, output));
  }

  /** FFT plan execution (double): complex -> complex
   *  @param[in] input Complex input double buffer.
   *  @param[out] output Complex output double buffer.
   *  @param[in] direction CUFFT_FORWARD or CUFFT_INVERSE
   */
  inline void exec(dcomplex * input, dcomplex * output, int direction)
  {
    IU_CUFFT_SAFE_CALL(cufftExecZ2Z(plan_, input, output, direction));
  }

  /** FFT plan execution (double): real -> complex
   *  @param[in] input Real input double buffer.
   *  @param[out] output Complex output double buffer.
   */
  inline void exec(dreal * input, dcomplex * output)
  {
    IU_CUFFT_SAFE_CALL(cufftExecD2Z(plan_, input, output));
  }

private:
  cufftHandle plan_;
};

/** Explicit FFT2 plan constructors for pitched memory (Image, float): real -> complex
 *  @param size Size of the pitched memory.
 *  @param src_pitch Source pitch
 *  @param dst_pitch Destination pitch
 */
template<>
inline Plan<freal, fcomplex, 2>::Plan(const iu::Size<2> &size,
                                      const int src_pitch, const int dst_pitch)
{
  int rank = 2;
  int width = static_cast<int>(size.width);
  int height = static_cast<int>(size.height);
  int n[2] = { height, width };
  int istride = 1, ostride = 1;  // Stride lengths
  int idist = src_pitch * height, odist = dst_pitch * height;  // Distance between batches
  int inembed[] = { height, src_pitch };  // src size with pitch
  int onembed[] = { height, dst_pitch };  // dst size with pitch
  int batch = 1;
  if (width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, inembed, istride, idist,  // *inembed, istride, idist
                                   onembed, ostride, odist,  // *onembed, ostride, odist
                                   CUFFT_R2C, batch));
}

/** Explicit FFT2 plan constructors for pitched memory (Image, float): complex -> real
 *  @param size Size of the pitched memory.
 *  @param src_pitch Source pitch
 *  @param dst_pitch Destination pitch
 */
template<>
inline Plan<fcomplex, freal, 2>::Plan(const iu::Size<2> &size, const int src_pitch,
                                      const int dst_pitch)
{
  int rank = 2;
  int width = static_cast<int>(size.width);
  int height = static_cast<int>(size.height);
  int n[2] = { height, width };
  int istride = 1, ostride = 1;  // Stride lengths
  int idist = src_pitch * height, odist = dst_pitch * height;  // Distance between batches
  int inembed[] = { height, src_pitch };  // src size with pitch
  int onembed[] = { height, dst_pitch };  // dst size with pitch
  int batch = 1;
  if (width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, inembed, istride, idist,  // *inembed, istride, idist
                                   onembed, ostride, odist,  // *onembed, ostride, odist
                                   CUFFT_C2R, batch));
}

/** Explicit FFT2 plan constructors for pitched memory (Image, float): complex -> complex
 *  @param size Size of the pitched memory.
 *  @param src_pitch Source pitch
 *  @param dst_pitch Destination pitch
 */
template<>
inline Plan<fcomplex, fcomplex, 2>::Plan(const iu::Size<2> &size,
                                         const int src_pitch,
                                         const int dst_pitch)
{
  int rank = 2;
  int width = static_cast<int>(size.width);
  int height = static_cast<int>(size.height);
  int n[2] = { height, width };
  int istride = 1, ostride = 1;  // Stride lengths
  int idist = src_pitch * height, odist = dst_pitch * height;  // Distance between batches
  int inembed[] = { height, src_pitch };  // src size with pitch
  int onembed[] = { height, dst_pitch };  // dst size with pitch
  int batch = 1;
  if (width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, inembed, istride, idist,  // *inembed, istride, idist
                                   onembed, ostride, odist,  // *onembed, ostride, odist
                                   CUFFT_C2C, batch));
}

/** Explicit FFT2 plan constructors for pitched memory (Volume, float): real -> complex
 *  @param size Size of the pitched memory.
 *  @param src_pitch Source pitch
 *  @param dst_pitch Destination pitch
 */
template<>
inline Plan<freal, fcomplex, 2>::Plan(const iu::Size<3> &size,
                                      const int src_pitch, const int dst_pitch)
{
  int rank = 2;
  int width = static_cast<int>(size.width);
  int height = static_cast<int>(size.height);
  int n[2] = { height, width };
  int istride = 1, ostride = 1;  // Stride lengths
  int idist = src_pitch * height, odist = dst_pitch * height;  // Distance between batches
  int inembed[] = { height, src_pitch };  // src size with pitch
  int onembed[] = { height, dst_pitch };  // dst size with pitch
  int batch = static_cast<int>(size.depth);
  if (batch == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, inembed, istride, idist,  // *inembed, istride, idist
                                   onembed, ostride, odist,  // *onembed, ostride, odist
                                   CUFFT_R2C, batch));
}

/** Explicit FFT2 plan constructors for pitched memory (Volume, float): complex -> real
 *  @param size Size of the pitched memory.
 *  @param src_pitch Source pitch
 *  @param dst_pitch Destination pitch
 */
template<>
inline Plan<fcomplex, freal, 2>::Plan(const iu::Size<3> &size, const int src_pitch,
                                      const int dst_pitch)
{
  int rank = 2;
  int width = static_cast<int>(size.width);
  int height = static_cast<int>(size.height);
  int n[2] = { height, width };
  int istride = 1, ostride = 1;  // Stride lengths
  int idist = src_pitch * height, odist = dst_pitch * height;  // Distance between batches
  int inembed[] = { height, src_pitch };  // src size with pitch
  int onembed[] = { height, dst_pitch };  // dst size with pitch
  int batch = static_cast<int>(size.depth);
  if (batch == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, inembed, istride, idist,  // *inembed, istride, idist
                                   onembed, ostride, odist,  // *onembed, ostride, odist
                                   CUFFT_C2R, batch));
}

/** Explicit FFT2 plan constructors for pitched memory (Volume, float): complex -> complex
 *  @param size Size of the pitched memory.
 *  @param src_pitch Source pitch
 *  @param dst_pitch Destination pitch
 */
template<>
inline Plan<fcomplex, fcomplex, 2>::Plan(const iu::Size<3> &size,
                                         const int src_pitch,
                                         const int dst_pitch)
{
  int rank = 2;
  int width = static_cast<int>(size.width);
  int height = static_cast<int>(size.height);
  int n[2] = { height, width };
  int istride = 1, ostride = 1;  // Stride lengths
  int idist = src_pitch * height, odist = dst_pitch * height;  // Distance between batches
  int inembed[] = { height, src_pitch };  // src size with pitch
  int onembed[] = { height, dst_pitch };  // dst size with pitch
  int batch = static_cast<int>(size.depth);
  if (batch == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, inembed, istride, idist,  // *inembed, istride, idist
                                   onembed, ostride, odist,  // *onembed, ostride, odist
                                   CUFFT_C2C, batch));
}

/** Explicit FFT3 plan constructors for pitched memory (Volume, float): real -> complex
 *  @param size Size of the pitched memory.
 *  @param src_pitch Source pitch
 *  @param dst_pitch Destination pitch
 */
template<>
inline Plan<freal, fcomplex, 3>::Plan(const iu::Size<3> &size,
                                      const int src_pitch, const int dst_pitch)
{
  int rank = 3;
  int width = static_cast<int>(size.width);
  int height = static_cast<int>(size.height);
  int depth = static_cast<int>(size.depth);
  if (depth == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }
  int n[3] = { depth, height, width };
  int istride = 1, ostride = 1;  // Stride lengths
  int idist = src_pitch * height * depth, odist = dst_pitch * height * depth;  // Distance between batches
  int inembed[] = { depth, height, src_pitch };  // src size with pitch
  int onembed[] = { depth, height, dst_pitch };  // dst size with pitch
  int batch = 1;

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, inembed, istride, idist,  // *inembed, istride, idist
                                   onembed, ostride, odist,  // *onembed, ostride, odist
                                   CUFFT_R2C, batch));
}

/** Explicit FFT3 plan constructors for pitched memory (Volume, float): complex -> real
 *  @param size Size of the pitched memory.
 *  @param src_pitch Source pitch
 *  @param dst_pitch Destination pitch
 */
template<>
inline Plan<fcomplex, freal, 3>::Plan(const iu::Size<3> &size, const int src_pitch,
                                      const int dst_pitch)
{
  int rank = 3;
  int width = static_cast<int>(size.width);
  int height = static_cast<int>(size.height);
  int depth = static_cast<int>(size.depth);
  if (depth == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }
  int n[3] = { depth, height, width };
  int istride = 1, ostride = 1;  // Stride lengths
  int idist = src_pitch * height * depth, odist = dst_pitch * height * depth;  // Distance between batches
  int inembed[] = { depth, height, src_pitch };  // src size with pitch
  int onembed[] = { depth, height, dst_pitch };  // dst size with pitch
  int batch = 1;

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, inembed, istride, idist,  // *inembed, istride, idist
                                   onembed, ostride, odist,  // *onembed, ostride, odist
                                   CUFFT_C2R, batch));
}

/** Explicit FFT3 plan constructors for pitched memory (Volume, float): complex -> complex
 *  @param size Size of the pitched memory.
 *  @param src_pitch Source pitch
 *  @param dst_pitch Destination pitch
 */
template<>
inline Plan<fcomplex, fcomplex, 3>::Plan(const iu::Size<3> &size,
                                         const int src_pitch,
                                         const int dst_pitch)
{
  int rank = 3;
  int width = static_cast<int>(size.width);
  int height = static_cast<int>(size.height);
  int depth = static_cast<int>(size.depth);
  if (depth == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }
  int n[3] = { depth, height, width };
  int istride = 1, ostride = 1;  // Stride lengths
  int idist = src_pitch * height * depth, odist = dst_pitch * height * depth;  // Distance between batches
  int inembed[] = { depth, height, src_pitch };  // src size with pitch
  int onembed[] = { depth, height, dst_pitch };  // dst size with pitch
  int batch = 1;

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, inembed, istride, idist,  // *inembed, istride, idist
                                   onembed, ostride, odist,  // *onembed, ostride, odist
                                   CUFFT_C2C, batch));
}

/** Explicit FFT2 plan constructors for linear 2d memory (float): real -> complex
 *  @param size Size of the linear memory.
 */
template<>
inline Plan<freal, fcomplex, 2>::Plan(const iu::Size<2> &size)
{
  int rank = 2;
  int width = static_cast<int>(size[0]);
  int height = static_cast<int>(size[1]);
  int n[2] = { height, width };
  int batch = 1;
  if (width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, NULL, 1, width * height,  // *inembed, istride, idist
      NULL, 1, width * height / 2,// *onembed, ostride, odist
      CUFFT_R2C, batch));
}

/** Explicit FFT2 plan constructors for linear 2d memory (float): complex -> real
 *  @param size Size of the linear memory.
 */
template<>
inline Plan<fcomplex, freal, 2>::Plan(const iu::Size<2> &size)
{
  int rank = 2;
  int width = static_cast<int>(size[0]);
  int height = static_cast<int>(size[1]);
  int n[2] = { height, width };
  int batch = 1;
  if (width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, NULL, 1, width * height / 2,  // *inembed, istride, idist
      NULL, 1, width * height,// *onembed, ostride, odist
      CUFFT_C2R, batch));
}

/** Explicit FFT2 plan constructors for linear 2d memory (float): complex -> complex
 *  @param size Size of the linear memory.
 */
template<>
inline Plan<fcomplex, fcomplex, 2>::Plan(const iu::Size<2> &size)
{
  int rank = 2;
  int width = static_cast<int>(size[0]);
  int height = static_cast<int>(size[1]);
  int n[2] = { height, width };
  int batch = 1;
  if (width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, NULL, 1, width * height,  // *inembed, istride, idist
      NULL, 1, width * height,// *onembed, ostride, odist
      CUFFT_C2C, batch));
}

/** Explicit FFT2 plan constructors for linear 3d memory (float): real -> complex
 *
 *  Compute FFT2 slice-wise. Batch is size[2].
 *  @param size Size of the linear memory.
 */
template<>
inline Plan<freal, fcomplex, 2>::Plan(const iu::Size<3> &size)
{
  int rank = 2;
  int width = static_cast<int>(size[0]);
  int height = static_cast<int>(size[1]);
  int n[2] = { height, width };
  int batch = static_cast<int>(size[2]);
  if (batch == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, NULL, 1, size[0] * size[1],  // *inembed, istride, idist
      NULL, 1, width * height / 2,// *onembed, ostride, odist
      CUFFT_R2C, batch));
}

/** Explicit FFT2 plan constructors for linear 3d memory (float): complex -> real
 *
 *  Compute FFT2 slice-wise. Batch is size[2].
 *  @param size Size of the linear memory.
 */
template<>
inline Plan<fcomplex, freal, 2>::Plan(const iu::Size<3> &size)
{
  int rank = 2;
  int width = static_cast<int>(size[0]);
  int height = static_cast<int>(size[1]);
  int n[2] = { height, width };
  int batch = static_cast<int>(size[2]);
  if (batch == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, NULL, 1, width * height / 2,  // *inembed, istride, idist
      NULL, 1, width * height,// *onembed, ostride, odist
      CUFFT_C2R, batch));
}

/** Explicit FFT2 plan constructors for linear 3d memory (float): complex -> complex
 *
 *  Compute FFT2 slice-wise. Batch is size[2].
 *  @param size Size of the linear memory.
 */
template<>
inline Plan<fcomplex, fcomplex, 2>::Plan(const iu::Size<3> &size)
{
  int rank = 2;
  int width = static_cast<int>(size[0]);
  int height = static_cast<int>(size[1]);
  int n[2] = { height, width };
  int batch = static_cast<int>(size[2]);
  if (batch == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, NULL, 1, width * height,  // *inembed, istride, idist
      NULL, 1, width * height,// *onembed, ostride, odist
      CUFFT_C2C, batch));
}

/** Explicit FFT2 plan constructors for linear 4d memory (float): real -> complex
 *
 *  Compute FFT2 slice-wise. Batch is size[2]*size[3].
 *  @param size Size of the linear memory.
 */
template<>
inline Plan<freal, fcomplex, 2>::Plan(const iu::Size<4> &size)
{
  int rank = 2;
  int width = static_cast<int>(size[0]);
  int height = static_cast<int>(size[1]);
  int n[2] = { height, width };
  int batch = static_cast<int>(size[2] * size[3]);
  if (batch == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, NULL, 1, width * height,  // *inembed, istride, idist
      NULL, 1, width * height/2,// *onembed, ostride, odist
      CUFFT_R2C, batch));
}

/** Explicit FFT2 plan constructors for linear 4d memory (float): complex -> real
 *
 *  Compute FFT2 slice-wise. Batch is size[2]*size[3].
 *  @param size Size of the linear memory.
 */
template<>
inline Plan<fcomplex, freal, 2>::Plan(const iu::Size<4> &size)
{
  int rank = 2;
  int width = static_cast<int>(size[0]);
  int height = static_cast<int>(size[1]);
  int n[2] = { height, width };
  int batch = static_cast<int>(size[2]);
  if (batch == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, NULL, 1, width * height/2,  // *inembed, istride, idist
      NULL, 1, width * height,// *onembed, ostride, odist
      CUFFT_C2R, batch));
}

/** Explicit FFT2 plan constructors for linear 4d memory (float): complex -> complex
 *
 *  Compute FFT2 slice-wise. Batch is size[2]*size[3].
 *  @param size Size of the linear memory.
 */
template<>
inline Plan<fcomplex, fcomplex, 2>::Plan(const iu::Size<4> &size)
{
  int rank = 2;
  int width = static_cast<int>(size[0]);
  int height = static_cast<int>(size[1]);
  int n[2] = { height, width };
  int batch = static_cast<int>(size[2] * size[3]);
  if (batch == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, NULL, 1, width * height,  // *inembed, istride, idist
      NULL, 1, width * height,// *onembed, ostride, odist
      CUFFT_C2C, batch));
}

/** Explicit FFT3 plan constructors for linear 3d memory (float): real -> complex
 *  @param size Size of the linear memory.
 */
template<>
inline Plan<freal, fcomplex, 3>::Plan(const iu::Size<3> &size)
{
  int rank = 3;
  int width = static_cast<int>(size[0]);
  int height = static_cast<int>(size[1]);
  int depth = static_cast<int>(size[2]);
  int n[3] = { depth, height, width };
  int batch = 1;
  if (depth == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(
      cufftPlanMany(&plan_, rank, n, NULL, 1, width * height * depth,  // *inembed, istride, idist
      NULL, 1, width * height * depth/2,// *onembed, ostride, odist
      CUFFT_R2C, batch));
}

/** Explicit FFT3 plan constructors for linear 3d memory (float): complex -> real
 *  @param size Size of the linear memory.
 */
template<>
inline Plan<fcomplex, freal, 3>::Plan(const iu::Size<3> &size)
{
  int rank = 3;
  int width = static_cast<int>(size[0]);
  int height = static_cast<int>(size[1]);
  int depth = static_cast<int>(size[2]);
  int n[3] = { depth, height, width };
  int batch = 1;
  if (depth == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(
      cufftPlanMany(&plan_, rank, n, NULL, 1, width * height * depth/2,  // *inembed, istride, idist
      NULL, 1, width * height * depth,// *onembed, ostride, odist
      CUFFT_C2R, batch));
}

/** Explicit FFT3 plan constructors for linear 3d memory (float): complex -> complex
 *  @param size Size of the linear memory.
 */
template<>
inline Plan<fcomplex, fcomplex, 3>::Plan(const iu::Size<3> &size)
{
  int rank = 3;
  int width = static_cast<int>(size[0]);
  int height = static_cast<int>(size[1]);
  int depth = static_cast<int>(size[2]);
  int n[3] = { depth, height, width };
  int batch = 1;
  if (depth == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(
      cufftPlanMany(&plan_, rank, n, NULL, 1, width * height * depth,  // *inembed, istride, idist
      NULL, 1, width * height * depth,// *onembed, ostride, odist
      CUFFT_C2C, batch));
}

/** Explicit FFT3 plan constructors for linear 4d memory (float): real -> complex
 *
 *  Compute FFT3 volume-wise. Batch is size[3].
 *  @param size Size of the linear memory.
 */
template<>
inline Plan<freal, fcomplex, 3>::Plan(const iu::Size<4> &size)
{
  int rank = 3;
  int width = static_cast<int>(size[0]);
  int height = static_cast<int>(size[1]);
  int depth = static_cast<int>(size[2]);
  int n[3] = { depth, height, width };
  int batch = static_cast<int>(size[3]);
  if (batch == 0 || depth == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(
      cufftPlanMany(&plan_, rank, n, NULL, 1, width * height * depth,  // *inembed, istride, idist
      NULL, 1, width * height * depth/2,// *onembed, ostride, odist
      CUFFT_R2C, batch));
}

/** Explicit FFT3 plan constructors for linear 4d memory (float): complex -> real
 *
 *  Compute FFT3 volume-wise. Batch is size[3].
 *  @param size Size of the linear memory.
 */
template<>
inline Plan<fcomplex, freal, 3>::Plan(const iu::Size<4> &size)
{
  int rank = 3;
  int width = static_cast<int>(size[0]);
  int height = static_cast<int>(size[1]);
  int depth = static_cast<int>(size[2]);
  int n[3] = { depth, height, width };
  int batch = static_cast<int>(size[3]);
  if (batch == 0 || depth == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(
      cufftPlanMany(&plan_, rank, n, NULL, 1, width * height * depth/2,  // *inembed, istride, idist
      NULL, 1, width * height * depth,// *onembed, ostride, odist
      CUFFT_C2R, batch));
}

/** Explicit FFT3 plan constructors for linear 4d memory (float): complex -> complex
 *
 *  Compute FFT3 volume-wise. Batch is size[3].
 *  @param size Size of the linear memory.
 */
template<>
inline Plan<fcomplex, fcomplex, 3>::Plan(const iu::Size<4> &size)
{
  int rank = 3;
  int width = static_cast<int>(size[0]);
  int height = static_cast<int>(size[1]);
  int depth = static_cast<int>(size[2]);
  int n[3] = { depth, height, width };
  int batch = static_cast<int>(size[3]);
  if (batch == 0 || depth == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(
      cufftPlanMany(&plan_, rank, n, NULL, 1, width * height * depth,  // *inembed, istride, idist
      NULL, 1, width * height * depth,// *onembed, ostride, odist
      CUFFT_C2C, batch));
}

/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////
/** Explicit FFT2 plan constructors for pitched memory (Image, double): real -> complex
 *  @param size Size of the pitched memory.
 *  @param src_pitch Source pitch
 *  @param dst_pitch Destination pitch
 */
template<>
inline Plan<dreal, dcomplex, 2>::Plan(const iu::Size<2> &size,
                                      const int src_pitch, const int dst_pitch)
{
  int rank = 2;
  int width = static_cast<int>(size.width);
  int height = static_cast<int>(size.height);
  int n[2] = { height, width };
  int istride = 1, ostride = 1;  // Stride lengths
  int idist = src_pitch * height, odist = dst_pitch * height;  // Distance between batches
  int inembed[] = { height, src_pitch };  // src size with pitch
  int onembed[] = { height, dst_pitch };  // dst size with pitch
  int batch = 1;
  if (width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, inembed, istride, idist,  // *inembed, istride, idist
                                   onembed, ostride, odist,  // *onembed, ostride, odist
                                   CUFFT_D2Z, batch));
}

/** Explicit FFT2 plan constructors for pitched memory (Image, double): complex -> real
 *  @param size Size of the pitched memory.
 *  @param src_pitch Source pitch
 *  @param dst_pitch Destination pitch
 */
template<>
inline Plan<dcomplex, dreal, 2>::Plan(const iu::Size<2> &size, const int src_pitch,
                                      const int dst_pitch)
{
  int rank = 2;
  int width = static_cast<int>(size.width);
  int height = static_cast<int>(size.height);
  int n[2] = { height, width };
  int istride = 1, ostride = 1;  // Stride lengths
  int idist = src_pitch * height, odist = dst_pitch * height;  // Distance between batches
  int inembed[] = { height, src_pitch };  // src size with pitch
  int onembed[] = { height, dst_pitch };  // dst size with pitch
  int batch = 1;
  if (width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, inembed, istride, idist,  // *inembed, istride, idist
                                   onembed, ostride, odist,  // *onembed, ostride, odist
                                   CUFFT_Z2D, batch));
}

/** Explicit FFT2 plan constructors for pitched memory (Image, double): complex -> complex
 *  @param size Size of the pitched memory.
 *  @param src_pitch Source pitch
 *  @param dst_pitch Destination pitch
 */
template<>
inline Plan<dcomplex, dcomplex, 2>::Plan(const iu::Size<2> &size,
                                         const int src_pitch,
                                         const int dst_pitch)
{
  int rank = 2;
  int width = static_cast<int>(size.width);
  int height = static_cast<int>(size.height);
  int n[2] = { height, width };
  int istride = 1, ostride = 1;  // Stride lengths
  int idist = src_pitch * height, odist = dst_pitch * height;  // Distance between batches
  int inembed[] = { height, src_pitch };  // src size with pitch
  int onembed[] = { height, dst_pitch };  // dst size with pitch
  int batch = 1;
  if (width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, inembed, istride, idist,  // *inembed, istride, idist
                                   onembed, ostride, odist,  // *onembed, ostride, odist
                                   CUFFT_Z2Z, batch));
}


/** Explicit FFT2 plan constructors for pitched memory (Volume, double): real -> complex
 *  @param size Size of the pitched memory.
 *  @param src_pitch Source pitch
 *  @param dst_pitch Destination pitch
 */
template<>
inline Plan<dreal, dcomplex, 2>::Plan(const iu::Size<3> &size,
                                      const int src_pitch, const int dst_pitch)
{
  int rank = 2;
  int width = static_cast<int>(size.width);
  int height = static_cast<int>(size.height);
  int n[2] = { height, width };
  int istride = 1, ostride = 1;  // Stride lengths
  int idist = src_pitch * height, odist = dst_pitch * height;  // Distance between batches
  int inembed[] = { height, src_pitch };  // src size with pitch
  int onembed[] = { height, dst_pitch };  // dst size with pitch
  int batch = static_cast<int>(size.depth);
  if (batch == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, inembed, istride, idist,  // *inembed, istride, idist
                                   onembed, ostride, odist,  // *onembed, ostride, odist
                                   CUFFT_D2Z, batch));
}

/** Explicit FFT2 plan constructors for pitched memory (Volume, double): complex -> real
 *  @param size Size of the pitched memory.
 *  @param src_pitch Source pitch
 *  @param dst_pitch Destination pitch
 */
template<>
inline Plan<dcomplex, dreal, 2>::Plan(const iu::Size<3> &size, const int src_pitch,
                                      const int dst_pitch)
{
  int rank = 2;
  int width = static_cast<int>(size.width);
  int height = static_cast<int>(size.height);
  int n[2] = { height, width };
  int istride = 1, ostride = 1;  // Stride lengths
  int idist = src_pitch * height, odist = dst_pitch * height;  // Distance between batches
  int inembed[] = { height, src_pitch };  // src size with pitch
  int onembed[] = { height, dst_pitch };  // dst size with pitch
  int batch = static_cast<int>(size.depth);
  if (batch == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, inembed, istride, idist,  // *inembed, istride, idist
                                   onembed, ostride, odist,  // *onembed, ostride, odist
                                   CUFFT_Z2D, batch));
}

/** Explicit FFT2 plan constructors for pitched memory (Volume, double): complex -> complex
 *  @param size Size of the pitched memory.
 *  @param src_pitch Source pitch
 *  @param dst_pitch Destination pitch
 */
template<>
inline Plan<dcomplex, dcomplex, 2>::Plan(const iu::Size<3> &size,
                                         const int src_pitch,
                                         const int dst_pitch)
{
  int rank = 2;
  int width = static_cast<int>(size.width);
  int height = static_cast<int>(size.height);
  int n[2] = { height, width };
  int istride = 1, ostride = 1;  // Stride lengths
  int idist = src_pitch * height, odist = dst_pitch * height;  // Distance between batches
  int inembed[] = { height, src_pitch };  // src size with pitch
  int onembed[] = { height, dst_pitch };  // dst size with pitch
  int batch = static_cast<int>(size.depth);
  if (batch == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, inembed, istride, idist,  // *inembed, istride, idist
                                   onembed, ostride, odist,  // *onembed, ostride, odist
                                   CUFFT_Z2Z, batch));
}

/** Explicit FFT3 plan constructors for pitched memory (Volume, double): real -> complex
 *  @param size Size of the pitched memory.
 *  @param src_pitch Source pitch
 *  @param dst_pitch Destination pitch
 */
template<>
inline Plan<dreal, dcomplex, 3>::Plan(const iu::Size<3> &size,
                                      const int src_pitch, const int dst_pitch)
{
  int rank = 3;
  int width = static_cast<int>(size.width);
  int height = static_cast<int>(size.height);
  int depth = static_cast<int>(size.depth);
  if (depth == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }
  int n[3] = { depth, height, width };
  int istride = 1, ostride = 1;  // Stride lengths
  int idist = src_pitch * height * depth, odist = dst_pitch * height * depth;  // Distance between batches
  int inembed[] = { depth, height, src_pitch };  // src size with pitch
  int onembed[] = { depth, height, dst_pitch };  // dst size with pitch
  int batch = 1;

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, inembed, istride, idist,  // *inembed, istride, idist
                                   onembed, ostride, odist,  // *onembed, ostride, odist
                                   CUFFT_D2Z, batch));
}

/** Explicit FFT3 plan constructors for pitched memory (Volume, double): complex -> real
 *  @param size Size of the pitched memory.
 *  @param src_pitch Source pitch
 *  @param dst_pitch Destination pitch
 */
template<>
inline Plan<dcomplex, dreal, 3>::Plan(const iu::Size<3> &size, const int src_pitch,
                                      const int dst_pitch)
{
  int rank = 3;
  int width = static_cast<int>(size.width);
  int height = static_cast<int>(size.height);
  int depth = static_cast<int>(size.depth);
  if (depth == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }
  int n[3] = { depth, height, width };
  int istride = 1, ostride = 1;  // Stride lengths
  int idist = src_pitch * height * depth, odist = dst_pitch * height * depth;  // Distance between batches
  int inembed[] = { depth, height, src_pitch };  // src size with pitch
  int onembed[] = { depth, height, dst_pitch };  // dst size with pitch
  int batch = 1;

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, inembed, istride, idist,  // *inembed, istride, idist
                                   onembed, ostride, odist,  // *onembed, ostride, odist
                                   CUFFT_Z2D, batch));
}

/** Explicit FFT3 plan constructors for pitched memory (Volume, double): complex -> complex
 *  @param size Size of the pitched memory.
 *  @param src_pitch Source pitch
 *  @param dst_pitch Destination pitch
 */
template<>
inline Plan<dcomplex, dcomplex, 3>::Plan(const iu::Size<3> &size,
                                         const int src_pitch,
                                         const int dst_pitch)
{
  int rank = 3;
  int width = static_cast<int>(size.width);
  int height = static_cast<int>(size.height);
  int depth = static_cast<int>(size.depth);
  if (depth == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }
  int n[3] = { depth, height, width };
  int istride = 1, ostride = 1;  // Stride lengths
  int idist = src_pitch * height * depth, odist = dst_pitch * height * depth;  // Distance between batches
  int inembed[] = { depth, height, src_pitch };  // src size with pitch
  int onembed[] = { depth, height, dst_pitch };  // dst size with pitch
  int batch = 1;

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, inembed, istride, idist,  // *inembed, istride, idist
                                   onembed, ostride, odist,  // *onembed, ostride, odist
                                   CUFFT_Z2Z, batch));
}

/** Explicit FFT2 plan constructors for linear 2d memory (double): real -> complex
 *  @param size Size of the linear memory.
 */
template<>
inline Plan<dreal, dcomplex, 2>::Plan(const iu::Size<2> &size)
{
  int rank = 2;
  int width = static_cast<int>(size[0]);
  int height = static_cast<int>(size[1]);
  int n[2] = { height, width };
  int batch = 1;
  if (batch == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, NULL, 1, width * height,  // *inembed, istride, idist
      NULL, 1, width * height/2,// *onembed, ostride, odist
      CUFFT_D2Z, batch));
}

/** Explicit FFT2 plan constructors for linear 2d memory (double): complex -> real
 *  @param size Size of the linear memory.
 */
template<>
inline Plan<dcomplex, dreal, 2>::Plan(const iu::Size<2> &size)
{
  int rank = 2;
  int width = static_cast<int>(size[0]);
  int height = static_cast<int>(size[1]);
  int n[2] = { height, width };
  int batch = 1;
  if (batch == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, NULL, 1, width * height/2,  // *inembed, istride, idist
      NULL, 1, width * height,// *onembed, ostride, odist
      CUFFT_Z2D, batch));
}

/** Explicit FFT2 plan constructors for linear 2d memory (double): complex -> complex
 *  @param size Size of the linear memory.
 */
template<>
inline Plan<dcomplex, dcomplex, 2>::Plan(const iu::Size<2> &size)
{
  int rank = 2;
  int width = static_cast<int>(size[0]);
  int height = static_cast<int>(size[1]);
  int n[2] = { height, width };
  int batch = 1;
  if (batch == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, NULL, 1, width * height,  // *inembed, istride, idist
      NULL, 1, width * height,// *onembed, ostride, odist
      CUFFT_Z2Z, batch));
}

/** Explicit FFT2 plan constructors for linear 3d memory (double): real -> complex
 *
 *  Compute FFT2 slice-wise. Batch is size[2].
 *  @param size Size of the linear memory.
 */
template<>
inline Plan<dreal, dcomplex, 2>::Plan(const iu::Size<3> &size)
{
  int rank = 2;
  int width = static_cast<int>(size[0]);
  int height = static_cast<int>(size[1]);
  int n[2] = { height, width };
  int batch = static_cast<int>(size[2]);
  if (batch == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, NULL, 1, width * height,  // *inembed, istride, idist
      NULL, 1, width * height/2,// *onembed, ostride, odist
      CUFFT_D2Z, batch));
}

/** Explicit FFT2 plan constructors for linear 3d memory (double): complex -> real
 *
 *  Compute FFT2 slice-wise. Batch is size[2].
 *  @param size Size of the linear memory.
 */
template<>
inline Plan<dcomplex, dreal, 2>::Plan(const iu::Size<3> &size)
{
  int rank = 2;
  int width = static_cast<int>(size[0]);
  int height = static_cast<int>(size[1]);
  int n[2] = { height, width };
  int batch = static_cast<int>(size[2]);
  if (batch == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, NULL, 1, width * height/2,  // *inembed, istride, idist
      NULL, 1, width * height,// *onembed, ostride, odist
      CUFFT_Z2D, batch));
}

/** Explicit FFT2 plan constructors for linear 3d memory (double): complex -> complex
 *
 *  Compute FFT2 slice-wise. Batch is size[2].
 *  @param size Size of the linear memory.
 */
template<>
inline Plan<dcomplex, dcomplex, 2>::Plan(const iu::Size<3> &size)
{
  int rank = 2;
  int width = static_cast<int>(size[0]);
  int height = static_cast<int>(size[1]);
  int n[2] = { height, width };
  int batch = static_cast<int>(size[2]);
  if (batch == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, NULL, 1, width * height,  // *inembed, istride, idist
      NULL, 1, width * height,// *onembed, ostride, odist
      CUFFT_Z2Z, batch));
}

/** Explicit FFT2 plan constructors for linear 4d memory (double): real -> complex
 *
 *  Compute FFT2 slice-wise. Batch is size[2]*size[3].
 *  @param size Size of the linear memory.
 */
template<>
inline Plan<dreal, dcomplex, 2>::Plan(const iu::Size<4> &size)
{
  int rank = 2;
  int width = static_cast<int>(size[0]);
  int height = static_cast<int>(size[1]);
  int n[2] = { height, width };
  int batch = static_cast<int>(size[2] * size[3]);
  if (batch == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, NULL, 1, width * height,  // *inembed, istride, idist
      NULL, 1, width * height/2,// *onembed, ostride, odist
      CUFFT_D2Z, batch));
}

/** Explicit FFT2 plan constructors for linear 4d memory (double): complex -> real
 *
 *  Compute FFT2 slice-wise. Batch is size[2]*size[3].
 *  @param size Size of the linear memory.
 */
template<>
inline Plan<dcomplex, dreal, 2>::Plan(const iu::Size<4> &size)
{
  int rank = 2;
  int width = static_cast<int>(size[0]);
  int height = static_cast<int>(size[1]);
  int n[2] = { height, width };
  int batch = static_cast<int>(size[2]);
  if (batch == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, NULL, 1, width * height/2,  // *inembed, istride, idist
      NULL, 1, width * height,// *onembed, ostride, odist
      CUFFT_Z2D, batch));
}

/** Explicit FFT2 plan constructors for linear 4d memory (double): complex -> complex
 *
 *  Compute FFT2 slice-wise. Batch is size[2]*size[3].
 *  @param size Size of the linear memory.
 */
template<>
inline Plan<dcomplex, dcomplex, 2>::Plan(const iu::Size<4> &size)
{
  int rank = 2;
  int width = static_cast<int>(size[0]);
  int height = static_cast<int>(size[1]);
  int n[2] = { height, width };
  int batch = static_cast<int>(size[2] * size[3]);
  if (batch == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, NULL, 1, width * height,  // *inembed, istride, idist
      NULL, 1, width * height,// *onembed, ostride, odist
      CUFFT_Z2Z, batch));
}

/** Explicit FFT3 plan constructors for linear 3d memory (double): real -> complex
 *  @param size Size of the linear memory.
 */
template<>
inline Plan<dreal, dcomplex, 3>::Plan(const iu::Size<3> &size)
{
  int rank = 3;
  int width = static_cast<int>(size[0]);
  int height = static_cast<int>(size[1]);
  int depth = static_cast<int>(size[2]);
  int n[3] = { depth, height, width };
  int batch = 1;
  if (depth == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(
      cufftPlanMany(&plan_, rank, n, NULL, 1, width * height * depth,  // *inembed, istride, idist
      NULL, 1, width * height * depth/2,// *onembed, ostride, odist
      CUFFT_D2Z, batch));
}

/** Explicit FFT3 plan constructors for linear 3d memory (double): complex -> real
 *  @param size Size of the linear memory.
 */
template<>
inline Plan<dcomplex, dreal, 3>::Plan(const iu::Size<3> &size)
{
  int rank = 3;
  int width = static_cast<int>(size[0]);
  int height = static_cast<int>(size[1]);
  int depth = static_cast<int>(size[2]);
  int n[3] = { depth, height, width };
  int batch = 1;
  if (depth == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(
      cufftPlanMany(&plan_, rank, n, NULL, 1, width * height * depth/2,  // *inembed, istride, idist
      NULL, 1, width * height * depth,// *onembed, ostride, odist
      CUFFT_Z2D, batch));
}

/** Explicit FFT3 plan constructors for linear 3d memory (double): complex -> complex
 *  @param size Size of the linear memory.
 */
template<>
inline Plan<dcomplex, dcomplex, 3>::Plan(const iu::Size<3> &size)
{
  int rank = 3;
  int width = static_cast<int>(size[0]);
  int height = static_cast<int>(size[1]);
  int depth = static_cast<int>(size[2]);
  int n[3] = { depth, height, width };
  int batch = 1;
  if (depth == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(
      cufftPlanMany(&plan_, rank, n, NULL, 1, width * height * depth,  // *inembed, istride, idist
      NULL, 1, width * height * depth,// *onembed, ostride, odist
      CUFFT_Z2Z, batch));
}

/** Explicit FFT3 plan constructors for linear 4d memory (double): real -> complex
 *
 *  Compute FFT3 volume-wise. Batch is size[3].
 *  @param size Size of the linear memory.
 */
template<>
inline Plan<dreal, dcomplex, 3>::Plan(const iu::Size<4> &size)
{
  int rank = 3;
  int width = static_cast<int>(size[0]);
  int height = static_cast<int>(size[1]);
  int depth = static_cast<int>(size[2]);
  int n[3] = { depth, height, width };
  int batch = static_cast<int>(size[3]);
  if (batch == 0 || depth == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(
      cufftPlanMany(&plan_, rank, n, NULL, 1, width * height * depth,  // *inembed, istride, idist
      NULL, 1, width * height * depth/2,// *onembed, ostride, odist
      CUFFT_D2Z, batch));
}

/** Explicit FFT3 plan constructors for linear 4d memory (double): complex -> real
 *
 *  Compute FFT3 volume-wise. Batch is size[3].
 *  @param size Size of the linear memory.
 */
template<>
inline Plan<dcomplex, dreal, 3>::Plan(const iu::Size<4> &size)
{
  int rank = 3;
  int width = static_cast<int>(size[0]);
  int height = static_cast<int>(size[1]);
  int depth = static_cast<int>(size[2]);
  int n[3] = { depth, height, width };
  int batch = static_cast<int>(size[3]);
  if (batch == 0 || depth == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(
      cufftPlanMany(&plan_, rank, n, NULL, 1, width * height * depth/2,  // *inembed, istride, idist
      NULL, 1, width * height * depth,// *onembed, ostride, odist
      CUFFT_Z2D, batch));
}

/** Explicit FFT3 plan constructors for linear 4d memory (double): complex -> complex
 *
 *  Compute FFT3 volume-wise. Batch is size[3].
 *  @param size Size of the linear memory.
 */
template<>
inline Plan<dcomplex, dcomplex, 3>::Plan(const iu::Size<4> &size)
{
  int rank = 3;
  int width = static_cast<int>(size[0]);
  int height = static_cast<int>(size[1]);
  int depth = static_cast<int>(size[2]);
  int n[3] = { depth, height, width };
  int batch = static_cast<int>(size[3]);
  if (batch == 0 || depth == 0 || width == 0 || height == 0)
  {
    std::stringstream msg;
    msg << "Size elements cannot be zero! (Size: " << size << ")";
    throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
  }

  IU_CUFFT_SAFE_CALL(
      cufftPlanMany(&plan_, rank, n, NULL, 1, width * height * depth,  // *inembed, istride, idist
      NULL, 1, width * height * depth,// *onembed, ostride, odist
      CUFFT_Z2Z, batch));
}

}
}
}

#endif /* FFTPLAN_H_ */
