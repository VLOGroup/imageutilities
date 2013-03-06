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
  void sumSparseRow(iu::SparseMatrixGpu<float>* A, iu::LinearDeviceMemory_32f_C1* dst, float add_const, IuSparseSum function)
  {
    if (A->n_row() != dst->length())
    {
      throw IuException("number of rows does not match output size!", __FILE__, __FUNCTION__, __LINE__ );
    }

    A->changeSparseFormat(CSR);

    cuSumRow(A, dst->data(), add_const, function);
  }

  void sumSparseRow(iu::SparseMatrixGpu<float>* A, iu::ImageGpu_32f_C1* dst, float add_const, IuSparseSum function)
  {

    if (A->n_row() != dst->stride()*dst->height())
    {
      throw IuException("number of rows does not match output size!", __FILE__, __FUNCTION__, __LINE__ );
    }

    A->changeSparseFormat(CSR);

    cuSumRow(A, dst->data(), add_const, function);
  }

  void sumSparseRow(iu::SparseMatrixGpu<float>* A, iu::VolumeGpu_32f_C1* dst, float add_const, IuSparseSum function)
  {

    if (A->n_row() != dst->stride()*dst->height()*dst->depth())
    {
      throw IuException("number of rows does not match output size!", __FILE__, __FUNCTION__, __LINE__ );
    }

    A->changeSparseFormat(CSR);

    cuSumRow(A, dst->data(), add_const, function);
  }



// COLUMN ///////////////////////////////////////////////////////////////

  // sum up sparse matrix in column direction
  void sumSparseCol(iu::SparseMatrixGpu<float>* A, iu::LinearDeviceMemory_32f_C1* dst, float add_const, IuSparseSum function)
  {
    if (A->n_col() != dst->length())
    {
      throw IuException("number of columns does not match output size!", __FILE__, __FUNCTION__, __LINE__ );
    }

    A->changeSparseFormat(CSC);
    cuSumCol(A, dst->data(), add_const, function);
  }

  // sum up sparse matrix in column direction
  void sumSparseCol(iu::SparseMatrixGpu<float>* A, iu::ImageGpu_32f_C1* dst, float add_const, IuSparseSum function)
  {
    if (A->n_col() != dst->stride()*dst->height())
    {
      throw IuException("number of columns does not match output size!", __FILE__, __FUNCTION__, __LINE__ );
    }

    A->changeSparseFormat(CSC);
    cuSumCol(A, dst->data(), add_const, function);
  }

    // sum up sparse matrix in column direction
  void sumSparseCol(iu::SparseMatrixGpu<float>* A, iu::VolumeGpu_32f_C1* dst, float add_const, IuSparseSum function)
  {
    if (A->n_col() != dst->stride()*dst->height()*dst->depth())
    {
      throw IuException("number of columns does not match output size!", __FILE__, __FUNCTION__, __LINE__ );
    }

    A->changeSparseFormat(CSC);
    cuSumCol(A, dst->data(), add_const, function);
  }

} // namespace iu
