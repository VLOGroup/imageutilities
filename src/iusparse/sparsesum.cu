/*
 * Copyright (c) ICG. All rights reserved.
 *
 * Institute for Computer Graphics and Vision
 * Graz University of Technology / Austria
 *
 *
 * This software is distributed WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 * PURPOSE.  See the above copyright notices for more information.
 *
 *
 * Project     : ImageUtilities
 * Module      : Sparse
 * Class       : none
 * Language    : CUDA
 * Description : Implementation of Cuda wrappers for sparse sum functions
 *
 * Author     : 
 * EMail      : 
 *
 */

#ifndef IUSPARSE_SUM_CU
#define IUSPARSE_SUM_CU

#include <iucore/iutextures.cuh>
#include <iucutil.h>
#include "sparsesum.cuh"

namespace iuprivate {

  __global__ void cuSumKernelNO(const float* value, const int* row, float* dst, float add_const, int cnt)
  {
    int ind = blockIdx.x*blockDim.x + threadIdx.x;

    if (ind<cnt)
    {
      float sum = 0.0f;
      for (int i=row[ind]; i<row[ind+1]; i++)
        sum += value[i];
      dst[ind] = sum + add_const;
    }
  }

  __global__ void cuSumKernelABS(const float* value, const int* row, float* dst, float add_const, int cnt)
  {
    int ind = blockIdx.x*blockDim.x + threadIdx.x;

    if (ind<cnt)
    {
      float sum = 0.0f;
      for (int i=row[ind]; i<row[ind+1]; i++)
        sum += abs(value[i]);
      dst[ind] = sum + add_const;
    }
  }

  __global__ void cuSumKernelSQR(const float* value, const int* row, float* dst, float add_const, int cnt)
  {
    int ind = blockIdx.x*blockDim.x + threadIdx.x;

    if (ind<cnt)
    {
      float sum = 0.0f;
      for (int i=row[ind]; i<row[ind+1]; i++)
      {
        float v = value[i];
        sum += v*v;
      }
      dst[ind] = sum + add_const;
    }
  }

  __global__ void cuSumKernelCNT(const float* value, const int* row, float* dst, float add_const, int cnt)
  {
    int ind = blockIdx.x*blockDim.x + threadIdx.x;

    if (ind<cnt)
    {
      float sum = 0.0f;
      for (int i=row[ind]; i<row[ind+1]; i++)
        if (value[i] != 0.0f)
         sum += 1.0f;
      dst[ind] = sum + add_const;
    }
  }

  // Sums up a sparse matrix along the rows
  IuStatus cuSumRow(iu::SparseMatrixGpu<float>* A, float* dst, float add_const, IuSparseSum function)

  {
    // fragmentation
    dim3 dimBlock(256, 1);
    dim3 dimGrid(iu::divUp(A->n_row(), dimBlock.x), 1);

    if (function == IU_NO)
      cuSumKernelNO <<<dimGrid, dimBlock>>> (A->value()->data(), A->row()->data(), dst, add_const, A->n_row());
    else if (function == IU_ABS)
      cuSumKernelABS <<<dimGrid, dimBlock>>> (A->value()->data(), A->row()->data(), dst, add_const, A->n_row());
    else if (function == IU_SQR)
      cuSumKernelSQR <<<dimGrid, dimBlock>>> (A->value()->data(), A->row()->data(), dst, add_const, A->n_row());
    else if (function == IU_CNT)
      cuSumKernelCNT <<<dimGrid, dimBlock>>> (A->value()->data(), A->row()->data(), dst, add_const, A->n_row());
    else
      return IU_NOT_SUPPORTED_ERROR;

    // error check
    IU_CHECK_AND_RETURN_CUDA_ERRORS();
  }

  // Sums up a sparse matrix along the columns
  IuStatus cuSumCol(iu::SparseMatrixGpu<float>* A, float* dst, float add_const, IuSparseSum function)
  {
    // fragmentation
    dim3 dimBlock(256, 1);
    dim3 dimGrid(iu::divUp(A->n_col(), dimBlock.x), 1);

    if (function == IU_NO)
      cuSumKernelNO <<<dimGrid, dimBlock>>> (A->value()->data(), A->col()->data(), dst, add_const, A->n_col());
    else if (function == IU_ABS)
      cuSumKernelABS <<<dimGrid, dimBlock>>> (A->value()->data(), A->col()->data(), dst, add_const, A->n_col());
    else if (function == IU_SQR)
      cuSumKernelSQR <<<dimGrid, dimBlock>>> (A->value()->data(), A->col()->data(), dst, add_const, A->n_col());
    else if (function == IU_CNT)
      cuSumKernelCNT <<<dimGrid, dimBlock>>> (A->value()->data(), A->col()->data(), dst, add_const, A->n_col());
    else
      return IU_NOT_SUPPORTED_ERROR;

    // error check
    IU_CHECK_AND_RETURN_CUDA_ERRORS();
  }

} // namespace iuprivate

#endif // IUSPARSE_SUM_CU

