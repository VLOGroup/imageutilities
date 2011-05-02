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

  ///////////////////////////////////////////////////////////////////////////////

  // sum up sparse matrix in row direction
  IuStatus sumSparseRow(iu::SparseMatrixGpu<float>* A, iu::LinearDeviceMemory_32f_C1* dst, IuSparseSum function)
  {
    IuStatus status;
    // TODO: Check if dst is big enough
    A->changeSparseFormat(CSR);
    status = cuSumRow(A, dst->data(), function);
    IU_ASSERT(status == IU_SUCCESS);
    return IU_NO_ERROR;
  }

  IuStatus sumSparseRow(iu::SparseMatrixGpu<float>* A, iu::ImageGpu_32f_C1* dst, IuSparseSum function)
  {
    IuStatus status;
    // TODO: Check if dst is big enough
    A->changeSparseFormat(CSR);
    status = cuSumRow(A, dst->data(), function);
    IU_ASSERT(status == IU_SUCCESS);
    return IU_NO_ERROR;
  }




  // sum up sparse matrix in column direction
  IuStatus sumSparseCol(iu::SparseMatrixGpu<float>* A, iu::LinearDeviceMemory_32f_C1* dst, IuSparseSum function)
  {
    IuStatus status;
    // TODO: Check if dst is big enough
    A->changeSparseFormat(CSC);
    status = cuSumCol(A, dst->data(), function);
    IU_ASSERT(status == IU_SUCCESS);
    return IU_NO_ERROR;
  }

  // sum up sparse matrix in column direction
  IuStatus sumSparseCol(iu::SparseMatrixGpu<float>* A, iu::ImageGpu_32f_C1* dst, IuSparseSum function)
  {
    IuStatus status;
    // TODO: Check if dst is big enough
    A->changeSparseFormat(CSC);
    status = cuSumCol(A, dst->data(), function);
    IU_ASSERT(status == IU_SUCCESS);
    return IU_NO_ERROR;
  }


} // namespace iu
