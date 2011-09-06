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
 * Description : Definition of sparse sum functions
 *
 * Author     : 
 * EMail      : 
 *
 */

#ifndef IUSPARSE_SUM_H
#define IUSPARSE_SUM_H

#include <iucore/coredefs.h>
#include <iucore/memorydefs.h>
#include "sparsematrix_gpu.h"

namespace iuprivate {

/** Sums up rows of a sparse matrix
 */
IuStatus sumSparseRow(iu::SparseMatrixGpu<float>* A, iu::LinearDeviceMemory_32f_C1* dst, float add_const, IuSparseSum function);
IuStatus sumSparseRow(iu::SparseMatrixGpu<float>* A, iu::ImageGpu_32f_C1* dst, float add_const, IuSparseSum function);
IuStatus sumSparseRow(iu::SparseMatrixGpu<float>* A, iu::VolumeGpu_32f_C1* dst, float add_const, IuSparseSum function);

/** Sums up columns of a sparse matrix
 */
IuStatus sumSparseCol(iu::SparseMatrixGpu<float>* A, iu::LinearDeviceMemory_32f_C1* dst, float add_const, IuSparseSum function);
IuStatus sumSparseCol(iu::SparseMatrixGpu<float>* A, iu::ImageGpu_32f_C1* dst, float add_const, IuSparseSum function);
IuStatus sumSparseCol(iu::SparseMatrixGpu<float>* A, iu::VolumeGpu_32f_C1* dst, float add_const, IuSparseSum function);


} // namespace iuprivate

#endif // IUSPARSE_SUM_H
