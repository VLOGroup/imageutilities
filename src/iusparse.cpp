
#include "iusparse.h"
#include <iusparse/sparsesum.h>

namespace iu {

/* ***************************************************************************
 *  CONNECTORS
 * ***************************************************************************/

IuStatus sumSparseRow(iu::SparseMatrixGpu<float>* A, iu::LinearDeviceMemory_32f_C1* dst, IuSparseSum function)
{ return iuprivate::sumSparseRow(A, dst, function); }

IuStatus sumSparseRow(iu::SparseMatrixGpu<float>* A, iu::ImageGpu_32f_C1* dst, IuSparseSum function)
{ return iuprivate::sumSparseRow(A, dst, function); }



IuStatus sumSparseCol(iu::SparseMatrixGpu<float>* A, iu::LinearDeviceMemory_32f_C1* dst, IuSparseSum function)
{ return iuprivate::sumSparseCol(A, dst, function); }

IuStatus sumSparseCol(iu::SparseMatrixGpu<float>* A, iu::ImageGpu_32f_C1* dst, IuSparseSum function)
{ return iuprivate::sumSparseCol(A, dst, function); }


} // namespace iu
