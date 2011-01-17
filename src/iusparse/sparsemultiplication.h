#ifndef SPARSEMULTIPLICATION_H
#define SPARSEMULTIPLICATION_H

#include <cusparse.h>

#include <cuda_runtime_api.h>
#include <iucore/image_gpu.h>
#include "sparsematrix_gpu.h"

namespace iu {

#define CUSPARSE_SAFE_CALL_IUSTATUS(x)   if ((x) != CUSPARSE_STATUS_SUCCESS) {fprintf( stderr, "CUSPARSE ERROR\n" ); return IU_ERROR; }

  inline IuStatus sparseMultiplication(cusparseHandle_t* handle,
                                       iu::SparseMatrixGpu<float>* A,
                                       iu::ImageGpu_32f_C1* src,
                                       iu::ImageGpu_32f_C1* dst)
  {
    CUSPARSE_SAFE_CALL_IUSTATUS(cusparseScsrmv(*handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      A->n_row(), A->n_col(), 1.0f,
                                      A->mat_descriptor(), A->value()->data(),
                                      A->row()->data(), A->col()->data(),
                                      src->data(), 0.0f, dst->data()));

    return IU_NO_ERROR;
  }
} // namespace iuprivate

#endif // SPARSEMULTIPLICATION_H
