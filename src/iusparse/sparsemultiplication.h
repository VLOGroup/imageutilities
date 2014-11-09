#ifndef SPARSEMULTIPLICATION_H
#define SPARSEMULTIPLICATION_H

#include <cusparse.h>

#include <cuda_runtime_api.h>
#include <iucore/image_gpu.h>
#include "sparsematrix_gpu.h"

namespace iu {

#define CUSPARSE_SAFE_CALL_IUSTATUS(x)   if ((x) != CUSPARSE_STATUS_SUCCESS) {fprintf( stderr, "CUSPARSE ERROR\n" ); return IU_ERROR; }

// Core implementation
  inline IuStatus sparseMultiplicationCore(cusparseHandle_t* handle,
                                           iu::SparseMatrixGpu<float>* A,
                                           float* src, float* dst,
                                           cusparseOperation_t transpose=CUSPARSE_OPERATION_NON_TRANSPOSE)
  {
    if (A->sparseFormat() == CSR)
    {
      float alpha = 1.0f;
      float beta = 0.0f;
      CUSPARSE_SAFE_CALL_IUSTATUS(cusparseScsrmv(*handle, transpose,
                                                 A->n_row(), A->n_col(), A->value()->length(), &alpha,
                                                 A->mat_descriptor(), A->value()->data(),
                                                 A->row()->data(), A->col()->data(),
                                                 src, &beta, dst));
    }
    else if (A->sparseFormat() == CSC)
    {
      float alpha = 1.0f;
      float beta = 0.0f;
      
      if (transpose == CUSPARSE_OPERATION_NON_TRANSPOSE)
        transpose = CUSPARSE_OPERATION_TRANSPOSE;
      else
        transpose = CUSPARSE_OPERATION_NON_TRANSPOSE;
      CUSPARSE_SAFE_CALL_IUSTATUS(cusparseScsrmv(*handle, transpose,
                                                 A->n_col(), A->n_row(), A->value()->length(), &alpha,
                                                 A->mat_descriptor(), A->value()->data(),
                                                 A->col()->data(), A->row()->data(),
                                                 src, &beta, dst));
    }
    else
    {
      printf("ERROR: Sparse matrix format not supported!\n");
    }
    return IU_NO_ERROR;
  }

// Different wrapers
  inline IuStatus sparseMultiplication(cusparseHandle_t* handle,
                                       iu::SparseMatrixGpu<float>* A,
                                       iu::LinearDeviceMemory<float>* src,
                                       iu::LinearDeviceMemory<float>* dst,
                                       cusparseOperation_t transpose=CUSPARSE_OPERATION_NON_TRANSPOSE)
  {
    return sparseMultiplicationCore(handle, A, (float*)src->data(), (float*)dst->data(), transpose);
  }

  inline IuStatus sparseMultiplication(cusparseHandle_t* handle,
                                       iu::SparseMatrixGpu<float>* A,
                                       iu::ImageGpu_32f_C1* src,
                                       iu::ImageGpu_32f_C1* dst,
                                       cusparseOperation_t transpose=CUSPARSE_OPERATION_NON_TRANSPOSE)
  {
    return sparseMultiplicationCore(handle, A, (float*)src->data(), (float*)dst->data(), transpose);
  }

  inline IuStatus sparseMultiplication(cusparseHandle_t* handle,
                                       iu::SparseMatrixGpu<float>* A,
                                       iu::ImageGpu_32f_C1* src,
                                       iu::ImageGpu_32f_C2* dst,
                                       cusparseOperation_t transpose=CUSPARSE_OPERATION_NON_TRANSPOSE)
  {
    return sparseMultiplicationCore(handle, A, (float*)src->data(), (float*)dst->data(), transpose);
  }

  inline IuStatus sparseMultiplication(cusparseHandle_t* handle,
                                       iu::SparseMatrixGpu<float>* A,
                                       iu::ImageGpu_32f_C1* src,
                                       iu::ImageGpu_32f_C4* dst,
                                       cusparseOperation_t transpose=CUSPARSE_OPERATION_NON_TRANSPOSE)
  {
    return sparseMultiplicationCore(handle, A, (float*)src->data(), (float*)dst->data(), transpose);
  }

  inline IuStatus sparseMultiplication(cusparseHandle_t* handle,
                                       iu::SparseMatrixGpu<float>* A,
                                       iu::ImageGpu_32f_C2* src,
                                       iu::ImageGpu_32f_C1* dst,
                                       cusparseOperation_t transpose=CUSPARSE_OPERATION_NON_TRANSPOSE)
  {
    return sparseMultiplicationCore(handle, A, (float*)src->data(), (float*)dst->data(), transpose);
  }

  inline IuStatus sparseMultiplication(cusparseHandle_t* handle,
                                       iu::SparseMatrixGpu<float>* A,
                                       iu::ImageGpu_32f_C4* src,
                                       iu::ImageGpu_32f_C1* dst,
                                       cusparseOperation_t transpose=CUSPARSE_OPERATION_NON_TRANSPOSE)
  {
    return sparseMultiplicationCore(handle, A, (float*)src->data(), (float*)dst->data(), transpose);
  }

  inline IuStatus sparseMultiplication(cusparseHandle_t* handle,
                                       iu::SparseMatrixGpu<float>* A,
                                       iu::VolumeGpu_32f_C1* src,
                                       iu::VolumeGpu_32f_C1* dst,
                                       cusparseOperation_t transpose=CUSPARSE_OPERATION_NON_TRANSPOSE)
  {
    return sparseMultiplicationCore(handle, A, (float*)src->data(), (float*)dst->data(), transpose);
  }

  inline IuStatus sparseMultiplication(cusparseHandle_t* handle,
                                       iu::SparseMatrixGpu<float>* A,
                                       iu::ImageGpu_32f_C1* src,
                                       iu::VolumeGpu_32f_C1* dst,
                                       cusparseOperation_t transpose=CUSPARSE_OPERATION_NON_TRANSPOSE)
  {
    return sparseMultiplicationCore(handle, A, (float*)src->data(), (float*)dst->data(), transpose);
  }


  inline IuStatus sparseMultiplication(cusparseHandle_t* handle,
                                       iu::SparseMatrixGpu<float>* A,
                                       iu::VolumeGpu_32f_C1* src,
                                       iu::ImageGpu_32f_C1* dst,
                                       cusparseOperation_t transpose=CUSPARSE_OPERATION_NON_TRANSPOSE)
  {
    return sparseMultiplicationCore(handle, A, (float*)src->data(), (float*)dst->data(), transpose);
  }


} // namespace iuprivate

#endif // SPARSEMULTIPLICATION_H
