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
 * Description : Definition of Cuda wrappers for sparse sum functions
 *
 * Author     : 
 * EMail      : 
 *
 */

#ifndef IUSPARSE_SUM_CUH
#define IUSPARSE_SUM_CUH

#include <iucore/coredefs.h>
#include <iucore/memorydefs.h>
#include "sparsematrix_gpu.h"


namespace iuprivate {

IuStatus cuSumRow(iu::SparseMatrixGpu<float>* A, float* dst, float add_const, IuSparseSum function);

IuStatus cuSumCol(iu::SparseMatrixGpu<float>* A, float* dst, float add_const, IuSparseSum function);

} // namespace iuprivate

#endif // IUSPARSE_SUM_CUH
