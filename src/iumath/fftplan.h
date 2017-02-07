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

template<typename InputType, typename OutputType>
struct CUFFTWrapper
{
};

template<>
struct CUFFTWrapper<freal, fcomplex>
{
  /** Type deduction of FFT
   */
  inline cufftType getType() const
  {
    return CUFFT_R2C;
  }

  /** FFT plan execution (float): real -> complex
   *  @param[in] input real input float buffer.
   *  @param[out] output complex output float buffer.
   */
  inline void exec(cufftHandle &plan, const freal *input, fcomplex *output)
  {
    freal * nonconst_input = const_cast<freal *>(input);
    IU_CUFFT_SAFE_CALL(cufftExecR2C(plan, nonconst_input, output));
  }
};

template<>
struct CUFFTWrapper<dreal, dcomplex>
{
  /** Type deduction of FFT
   */
  inline cufftType getType() const
  {
    return CUFFT_D2Z;
  }

  /** FFT plan execution (double): real -> complex
   *  @param[in] input real input float buffer.
   *  @param[out] output complex output float buffer.
   */
  inline void exec(cufftHandle &plan, const dreal *input, dcomplex *output)
  {
    dreal * nonconst_input = const_cast<dreal *>(input);
    IU_CUFFT_SAFE_CALL(cufftExecD2Z(plan, nonconst_input, output));
  }
};

template<>
struct CUFFTWrapper<fcomplex, fcomplex>
{
  /** Type deduction of FFT
   */
  inline cufftType getType() const
  {
    return CUFFT_C2C;
  }

  /** FFT plan execution (float): complex -> complex
   *  @param[in] input complex input float buffer.
   *  @param[out] output complex output float buffer.
   */
  inline void exec(cufftHandle &plan, const fcomplex *input, fcomplex *output, int direction)
  {
    fcomplex * nonconst_input = const_cast<fcomplex *>(input);
    IU_CUFFT_SAFE_CALL(cufftExecC2C(plan, nonconst_input, output, direction));
  }
};

template<>
struct CUFFTWrapper<dcomplex, dcomplex>
{
  /** Type deduction of FFT
   */
  inline cufftType getType() const
  {
    return CUFFT_Z2Z;
  }

  /** FFT plan execution (double): complex -> complex
   *  @param[in] input complex input float buffer.
   *  @param[out] output complex output float buffer.
   */
  inline void exec(cufftHandle &plan, const dcomplex *input, dcomplex *output, int direction)
  {
    dcomplex * nonconst_input = const_cast<dcomplex *>(input);
    IU_CUFFT_SAFE_CALL(cufftExecZ2Z(plan, nonconst_input, output, direction));
  }
};

template<>
struct CUFFTWrapper<fcomplex, freal>
{
  /** Type deduction of FFT
   */
  inline cufftType getType() const
  {
    return CUFFT_C2R;
  }

  /** FFT plan execution (float): complex -> real
   *  @param[in] input Complex input float buffer.
   *  @param[out] output Real output float buffer.
   */
  inline void exec(cufftHandle &plan, const fcomplex *input, freal *output)
  {
    fcomplex * nonconst_input = const_cast<fcomplex *>(input);
    IU_CUFFT_SAFE_CALL(cufftExecC2R(plan, nonconst_input, output));
  }
};

template<>
struct CUFFTWrapper<dcomplex, dreal>
{
  /** Type deduction of FFT
   */
  inline cufftType getType() const
  {
    return CUFFT_Z2D;
  }

  /** FFT plan execution (double): complex -> real
   *  @param[in] input Complex input float buffer.
   *  @param[out] output Real output float buffer.
   */
  inline void exec(cufftHandle &plan, const dcomplex *input, dreal *output)
  {
    dcomplex * nonconst_input = const_cast<dcomplex *>(input);
    IU_CUFFT_SAFE_CALL(cufftExecZ2D(plan, nonconst_input, output));
  }
};

/** \brief Base class for FFT Plan.
 *
 * This class setups the memory layout for fft and executes forward and
 * inverse fft.
 */
template<class InputType, class OutputType, int FFTDim>
class Plan
{
public:
  Plan();

  /** Constructor for linear memory layout.
   *  @param size Size/Layout of the linear memory.
   */
  inline Plan(const iu::Size<InputType::ndim> &size)
  {
    if (FFTDim > InputType::ndim)
    {
      std::stringstream msg;
      msg << FFTDim << "-FFT dimension larger than input dimension! (" << InputType::ndim << ")";
      throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
    }
    // dimensionality of the Fourier transform
    int rank = FFTDim;
    // setup the size array
    int n[FFTDim];
    int d = 0;
    int n_elem = 1;
    for (; d < FFTDim; ++d)
    {
      n[d] = size[d];
      n_elem *= n[d];
    }
    // compute the number of batches
    int batch = 1;
    for (; d < InputType::ndim; ++d)
      batch *= size[d];
    // size check
    if (batch == 0 || n_elem == 0)
    {
      std::stringstream msg;
      msg << "Size elements cannot be zero! (Size: " << size << ")";
      throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
    }

    IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, NULL, 1, size[0],
        NULL, 1, size[0] / 2,
        CUFFTWrapper<typename InputType::pixel_type, typename OutputType::pixel_type>().getType(),
        batch));
  }

  /** Destructor. */
  virtual ~Plan()
  {
    IU_CUFFT_SAFE_CALL(cufftDestroy(plan_));
  }

  /** FFT plan execution
   *  @param[in] input input float buffer.
   *  @param[out] output output float buffer.
   */
  inline void exec(const typename InputType::pixel_type *input,
                   typename OutputType::pixel_type *output)
  {
    CUFFTWrapper<typename InputType::pixel_type, typename OutputType::pixel_type>().exec(plan_, input, output);
  }

private:
  cufftHandle plan_;
};

/** \brief Base class for FFT Plan complex <-> complex.
 *
 * This class setups the memory layout for fft and executes forward and
 * inverse fft.
 */
template<typename InputType, int FFTDim>
class Plan<InputType, InputType, FFTDim>
{
public:
  Plan();

  /** Constructor for linear memory layout.
   *  @param size Size/Layout of the linear memory.
   */
  inline Plan(const iu::Size<InputType::ndim> &size)
  {
    if (FFTDim > InputType::ndim)
    {
      std::stringstream msg;
      msg << FFTDim << "-FFT dimension larger than input dimension! (" << InputType::ndim << ")";
      throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
    }
    // dimensionality of the Fourier transform
    int rank = FFTDim;
    // setup the size array
    int n[FFTDim];
    int d = 0;
    int n_elem = 1;
    for (; d < FFTDim; ++d)
    {
      n[d] = size[d];
      n_elem *= n[d];
    }
    // compute the number of batches
    int batch = 1;
    for (; d < InputType::ndim; ++d)
      batch *= size[d];
    // size check
    if (batch == 0 || n_elem == 0)
    {
      std::stringstream msg;
      msg << "Size elements cannot be zero! (Size: " << size << ")";
      throw IuException(msg.str(), __FILE__, __FUNCTION__, __LINE__);
    }

    IU_CUFFT_SAFE_CALL(cufftPlanMany(&plan_, rank, n, NULL, 1, size[0],
        NULL, 1, size[0] / 2,
        CUFFTWrapper<typename InputType::pixel_type, typename InputType::pixel_type>().getType(), batch));
  }

  /** Destructor. */
  virtual ~Plan()
  {
    IU_CUFFT_SAFE_CALL(cufftDestroy(plan_));
  }

  /** FFT plan execution direction
   *  @param[in] input Complex input float buffer.
   *  @param[out] output Complex output float buffer.
   *  @param[in] direction CUFFT_FORWARD or CUFFT_INVERSE
   */
  inline void exec(const typename InputType::pixel_type *input,
                   typename InputType::pixel_type *output, int direction)
  {
    CUFFTWrapper<typename InputType::pixel_type, typename InputType::pixel_type>().exec(plan_, input, output, direction);
  }

private:
  cufftHandle plan_;
};

}
}
}

#endif /* FFTPLAN_H_ */
