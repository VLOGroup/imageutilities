#ifndef IUPRIVATE_MATLABSPARSECONNECTOR_H
#define IUPRIVATE_MATLABSPARSECONNECTOR_H


//
//  W A R N I N G
//  -------------
//
// This file is not part of the IU API.  It exists purely as an
// implementation detail.  This header file may change from version to
// version without notice, or even be removed.
//

#include <iostream>
#include <iudefs.h>
#include <iucutil.h>
#include <iusparse/sparsematrixdefs.h>
#include <mex.h>

namespace iuprivate {

  //-----------------------------------------------------------------------------
  // convert Matlab sparse matrix to sparse matrix on host
  //template<typename PixelType>
  inline IuStatus convertSparseMatrixToCpu_32f(const mxArray* src, iu::SparseMatrixCpu_32f** dst)
  {
    int n_row = mxGetM(src);
    int n_col = mxGetN(src);
    int n_elements = mxGetNzmax(src);

    double* val = mxGetPr(src);
    mwIndex* row = mxGetIr(src);
    mwIndex* col = mxGetJc(src);

    iu::LinearHostMemory<float> h_val(n_elements);
    iu::LinearHostMemory<int> h_row(n_elements);
    iu::LinearHostMemory<int> h_col(n_col+1);

    for(int i=0; i<n_elements; i++)
    {
      *h_val.data(i) = (float)val[i];
      *h_row.data(i) = (int)row[i];
      if (i<=n_col)
        *h_col.data(i) = (int)col[i];
    }

    *dst = new iu::SparseMatrixCpu_32f(&h_val, &h_row, &h_col, n_row, n_col);

    return IU_NO_ERROR;
  }

  //-----------------------------------------------------------------------------
  // convert Matlab sparse matrix to sparse matrix on device
  //template<typename PixelType>
  inline IuStatus convertSparseMatrixToGpu_32f(cusparseHandle_t* handle,
                                    const mxArray* src, iu::SparseMatrixGpu_32f** dst)
  {
    iu::SparseMatrixCpu_32f* dst_h;
    convertSparseMatrixToCpu(src, &dst_h);
    *dst = new iu::SparseMatrixGpu_32f(handle, dst_h);

    return IU_NO_ERROR;
  }

}

#endif // IUPRIVATE_MATLABSPARSECONNECTOR_H
