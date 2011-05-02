#ifndef SPARSEMATRIXDEFS_H
#define SPARSEMATRIXDEFS_H

#include "iudefs.h"
#include "sparsematrix_cpu.h"
#include "sparsematrix_gpu.h"
#include "sparsemultiplication.h"
#include "sparsesum.h"

namespace iu {

/* ****************************************************************************
 *  Sparse Matrix Definitions
 * ****************************************************************************/


/*
  Host
*/
// 32-bit
typedef SparseMatrixCpu<float> SparseMatrixCpu_32f;

/*
  Device
*/
// 32-bit
typedef SparseMatrixGpu<float> SparseMatrixGpu_32f;


} // namespace iu

#endif // SPARSEMATRIXDEFS_H
