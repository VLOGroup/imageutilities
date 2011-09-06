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
 * Language    : C++
 * Description : Implementation of sparse sum functions
 *
 * Author     :
 * EMail      :
 *
 */

#include "sparsesum.cuh"
#include "sparsesum.h"

namespace iuprivate {

// ROW ///////////////////////////////////////////////////////////////////

  // sum up sparse matrix in row direction
  IuStatus sumSparseRow(iu::SparseMatrixGpu<float>* A, iu::LinearDeviceMemory_32f_C1* dst, float add_const, IuSparseSum function)
  {
    IuStatus status;

    if (A->n_row() != dst->length())
    {
      printf("ERROR in sumSparseRow: number of rows does not match output size!\n");
      return IU_ERROR;
    }

    A->changeSparseFormat(CSR);

    status = cuSumRow(A, dst->data(), add_const, function);
    if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
    return IU_NO_ERROR;
  }

  IuStatus sumSparseRow(iu::SparseMatrixGpu<float>* A, iu::ImageGpu_32f_C1* dst, float add_const, IuSparseSum function)
  {
    IuStatus status;

    if (A->n_row() != dst->stride()*dst->height())
    {
      printf("ERROR in sumSparseRow: number of rows does not match output size!\n");
      return IU_ERROR;
    }

    A->changeSparseFormat(CSR);

    status = cuSumRow(A, dst->data(), add_const, function);
    if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
    return IU_NO_ERROR;
  }

  IuStatus sumSparseRow(iu::SparseMatrixGpu<float>* A, iu::VolumeGpu_32f_C1* dst, float add_const, IuSparseSum function)
  {
    IuStatus status;

    if (A->n_row() != dst->stride()*dst->height()*dst->depth())
    {
      printf("ERROR in sumSparseRow: number of rows does not match output size!\n");
      return IU_ERROR;
    }

    A->changeSparseFormat(CSR);

    status = cuSumRow(A, dst->data(), add_const, function);
    if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
    return IU_NO_ERROR;
  }



// COLUMN ///////////////////////////////////////////////////////////////

  // sum up sparse matrix in column direction
  IuStatus sumSparseCol(iu::SparseMatrixGpu<float>* A, iu::LinearDeviceMemory_32f_C1* dst, float add_const, IuSparseSum function)
  {
    IuStatus status;

    if (A->n_col() != dst->length())
    {
      printf("ERROR in sumSparseCol: number of columns does not match output size!\n");
      return IU_ERROR;
    }

    A->changeSparseFormat(CSC);
    status = cuSumCol(A, dst->data(), add_const, function);
    if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
    return IU_NO_ERROR;
  }

  // sum up sparse matrix in column direction
  IuStatus sumSparseCol(iu::SparseMatrixGpu<float>* A, iu::ImageGpu_32f_C1* dst, float add_const, IuSparseSum function)
  {
    IuStatus status;

    if (A->n_col() != dst->stride()*dst->height())
    {
      printf("ERROR in sumSparseCol: number of columns does not match output size!\n");
      return IU_ERROR;
    }

    A->changeSparseFormat(CSC);
    status = cuSumCol(A, dst->data(), add_const, function);
    if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
    return IU_NO_ERROR;
  }

    // sum up sparse matrix in column direction
  IuStatus sumSparseCol(iu::SparseMatrixGpu<float>* A, iu::VolumeGpu_32f_C1* dst, float add_const, IuSparseSum function)
  {
    IuStatus status;

    if (A->n_col() != dst->stride()*dst->height()*dst->depth())
    {
      printf("ERROR in sumSparseCol: number of columns does not match output size!\n");
      return IU_ERROR;
    }

    A->changeSparseFormat(CSC);
    status = cuSumCol(A, dst->data(), add_const, function);
    if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
    return IU_NO_ERROR;
  }

} // namespace iu
