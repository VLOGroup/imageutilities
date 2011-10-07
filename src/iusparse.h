#ifndef IUSPARSE_H
#define IUSPARSE_H


#include "iudefs.h"
#include <iusparse/sparsematrixdefs.h>


namespace iu {

/** Sums up rows of a sparse matrix
 */
IUCORE_DLLAPI IuStatus sumSparseRow(iu::SparseMatrixGpu<float>* A, iu::LinearDeviceMemory_32f_C1* dst, float add_const=0.0f, IuSparseSum function=IU_NO);
IUCORE_DLLAPI IuStatus sumSparseRow(iu::SparseMatrixGpu<float>* A, iu::ImageGpu_32f_C1* dst, float add_const=0.0f, IuSparseSum function=IU_NO);
IUCORE_DLLAPI IuStatus sumSparseRow(iu::SparseMatrixGpu<float>* A, iu::VolumeGpu_32f_C1* dst, float add_const=0.0f, IuSparseSum function=IU_NO);

/** Sums up columns of a sparse matrix
 */
IUCORE_DLLAPI IuStatus sumSparseCol(iu::SparseMatrixGpu<float>* A, iu::LinearDeviceMemory_32f_C1* dst, float add_const=0.0f, IuSparseSum function=IU_NO);
IUCORE_DLLAPI IuStatus sumSparseCol(iu::SparseMatrixGpu<float>* A, iu::ImageGpu_32f_C1* dst, float add_const=0.0f, IuSparseSum function=IU_NO);
IUCORE_DLLAPI IuStatus sumSparseCol(iu::SparseMatrixGpu<float>* A, iu::VolumeGpu_32f_C1* dst, float add_const=0.0f, IuSparseSum function=IU_NO);

} // namespace iu

#endif // IUSPARSE_H
