#pragma once

#include <mex.h>

#include "iucore.h"
#include "iucore/linearhostmemory.h"
#include "iumath/typetraits.h"

#include <memory>
#include <sstream>

namespace iu {

namespace matlab {

template<typename PixelType>
struct dtype
{
};

template<>
struct dtype<float>
{
  static const mxComplexity matlab_type = mxREAL;
};

template<>
struct dtype<double>
{
  static const mxComplexity matlab_type = mxREAL;
};

template<>
struct dtype<float2>
{
  static const mxComplexity matlab_type = mxCOMPLEX;
};

template<>
struct dtype<double2>
{
  static const mxComplexity matlab_type = mxCOMPLEX;
};

template<unsigned int Ndim>
void getMatlabDims(const iu::Size<Ndim> size, mwSize *dims)
{
  dims[0] = size[1];
  dims[1] = size[0];

  for (unsigned int i = 2; i < 2; i++)
  {
    dims[i] = size[i];
  }
}

template<typename PixelType, unsigned int Ndim, typename ResultType = void>
typename std::enable_if<
    std::is_same<PixelType, typename iu::type_trait<PixelType>::complex_type>::value,
    ResultType>::type convertMatlabToC(
    const mxArray_tag& mex_array,
    iu::LinearHostMemory<PixelType, Ndim>& hostmem)
{
  if (!mxIsComplex(&mex_array))
  {
    mexErrMsgIdAndTxt("MATLAB:iumatlab:invalidType",
                      "Requested complex data. Matlab array is not complex!");
  }

  typedef typename iu::type_trait<PixelType>::real_type real_type;

  iu::Size<Ndim> size = hostmem.size();

  unsigned int offset = 1;

  for (unsigned int i = 2; i < Ndim; i++)
  {
    offset *= size[i];
  }

  for (int x = 0; x < size[0]; x++)
  {
    for (int y = 0; y < size[1]; y++)
    {
      for (int idx_offset = 0; idx_offset < offset; idx_offset++)
      {
        real_type real_value = static_cast<real_type>(mxGetPr(&mex_array)[y
            + size[1] * x + size[1] * size[0] * idx_offset]);
        real_type imag_value = static_cast<real_type>(mxGetPi(&mex_array)[y
            + size[1] * x + size[1] * size[0] * idx_offset]);
        hostmem.data()[x + size[0] * y + size[1] * size[0] * idx_offset] =
            iu::type_trait<PixelType>::make_complex(real_value, imag_value);
      }
    }
  }
}

template<typename PixelType, unsigned int Ndim, typename ResultType = void>
typename std::enable_if<
    std::is_same<PixelType, typename iu::type_trait<PixelType>::real_type>::value,
    ResultType>::type convertMatlabToC(
    const mxArray_tag& mex_array,
    iu::LinearHostMemory<PixelType, Ndim>& hostmem)
{
  if (mxIsComplex(&mex_array))
  {
    mexErrMsgIdAndTxt("MATLAB:iumatlab:invalidType",
                      "Requested real data. Matlab array is complex!");
  }

  typedef typename iu::type_trait<PixelType>::real_type real_type;

  iu::Size<Ndim> size = hostmem.size();
  unsigned int offset = 1;

  for (unsigned int i = 2; i < Ndim; i++)
  {
    offset *= size[i];
  }

  for (int x = 0; x < size[0]; x++)
  {
    for (int y = 0; y < size[1]; y++)
    {
      for (int idx_offset = 0; idx_offset < offset; idx_offset++)
      {
        real_type real_value = static_cast<real_type>(mxGetPr(&mex_array)[y
            + size[1] * x + size[1] * size[0] * idx_offset]);
        hostmem.data()[x + size[0] * y + size[1] * size[0] * idx_offset] =
            real_value;
      }
    }
  }
}

template<typename PixelType, unsigned int Ndim, typename ResultType = void>
typename std::enable_if<
    std::is_same<PixelType, typename iu::type_trait<PixelType>::complex_type>::value,
    ResultType>::type convertCToMatlab(
    const iu::LinearHostMemory<PixelType, Ndim>& hostmem,
    mxArray_tag *mex_array)
{
  // Convert to Matlab
  typedef typename iu::type_trait<PixelType>::complex_type complex_type;
  int nelem = mxGetNumberOfElements(mex_array);

  double* output_real = (double *) mxCalloc(nelem, sizeof(double));
  double* output_imag = (double *) mxCalloc(nelem, sizeof(double));

  iu::Size<Ndim> size = hostmem.size();

  unsigned int offset = 1;

  for (unsigned int i = 2; i < Ndim; i++)
  {
    offset *= size[i];
  }

  for (int x = 0; x < size[0]; x++)
  {
    for (int y = 0; y < size[1]; y++)
    {
      for (int idx_offset = 0; idx_offset < offset; idx_offset++)
      {
        complex_type value = hostmem.data()[x + size[0] * y
            + size[1] * size[0] * idx_offset];
        output_imag[y + size[1] * x + size[1] * size[0] * idx_offset] =
            static_cast<double>(value.y);
        output_real[y + size[1] * x + size[1] * size[0] * idx_offset] =
            static_cast<double>(value.x);
      }
    }
  }

  mxSetPr(mex_array, output_real);
  mxSetPi(mex_array, output_imag);
}

template<typename PixelType, unsigned int Ndim, typename ResultType = void>
typename std::enable_if<
    std::is_same<PixelType, typename iu::type_trait<PixelType>::real_type>::value,
    ResultType>::type convertCToMatlab(
    const iu::LinearHostMemory<PixelType, Ndim>& hostmem,
    mxArray_tag* mex_array)
{
  // Convert to Matlab
  typedef typename iu::type_trait<PixelType>::real_type real_type;
  int nelem = mxGetNumberOfElements(mex_array);

  double* output_real = (double *) mxCalloc(nelem, sizeof(double));

  iu::Size<Ndim> size = hostmem.size();

  unsigned int offset = 1;

  for (unsigned int i = 2; i < Ndim; i++)
  {
    offset *= size[i];
  }

  for (int x = 0; x < size[0]; x++)
  {
    for (int y = 0; y < size[1]; y++)
    {
      for (int idx_offset = 0; idx_offset < offset; idx_offset++)
      {
        real_type value = hostmem.data()[x + size[0] * y
            + size[1] * size[0] * idx_offset];
        output_real[y + size[1] * x + size[1] * size[0] * idx_offset] =
            static_cast<double>(value);
      }
    }
  }

  mxSetPr(mex_array, output_real);
}
}

template<typename PixelType, unsigned int Ndim>
LinearHostMemory<PixelType, Ndim>::LinearHostMemory(
    const mxArray_tag& mex_array) :
    data_(0), ext_data_pointer_(false)
{
  // Extract dims (minimum dim is always 2!)
  unsigned int matlab_ndim = mxGetNumberOfDimensions(&mex_array);

  if (matlab_ndim != Ndim)
  {
    std::stringstream msg;
    msg << "Invalid Dimension: Expected (" << Ndim << "), got (";
    msg << matlab_ndim << ")";
    mexErrMsgIdAndTxt("MATLAB:iumatlab:invalidDimension", msg.str().c_str());
  }

  // Get size
  const mwSize* matlab_size = mxGetDimensions(&mex_array);

  this->size_[0] = matlab_size[1];
  this->size_[1] = matlab_size[0];

  for (unsigned int i = 2; i < matlab_ndim; i++)
  {
    this->size_[i] = matlab_size[i];
  }

  // Allocate data
  data_ = (PixelType*) malloc(this->numel() * sizeof(PixelType));
  if (data_ == 0)
    throw std::bad_alloc();

  // convert Matlab to C
  matlab::convertMatlabToC(mex_array, *this);
}
}

