
#include "iusparse.h"
#include <iusparse/sparsesum.h>

namespace iu {

/* ***************************************************************************
 *  CONNECTORS
 * ***************************************************************************/

IuStatus sumSparseRow(iu::SparseMatrixGpu<float>* A, iu::LinearDeviceMemory_32f_C1* dst, float add_const, IuSparseSum function)
{ return iuprivate::sumSparseRow(A, dst, add_const, function); }

IuStatus sumSparseRow(iu::SparseMatrixGpu<float>* A, iu::ImageGpu_32f_C1* dst, float add_const, IuSparseSum function)
{ return iuprivate::sumSparseRow(A, dst, add_const, function); }

IuStatus sumSparseRow(iu::SparseMatrixGpu<float>* A, iu::VolumeGpu_32f_C1* dst, float add_const, IuSparseSum function)
{ return iuprivate::sumSparseRow(A, dst, add_const, function); }



IuStatus sumSparseCol(iu::SparseMatrixGpu<float>* A, iu::LinearDeviceMemory_32f_C1* dst, float add_const, IuSparseSum function)
{ return iuprivate::sumSparseCol(A, dst, add_const, function); }

IuStatus sumSparseCol(iu::SparseMatrixGpu<float>* A, iu::ImageGpu_32f_C1* dst, float add_const, IuSparseSum function)
{ return iuprivate::sumSparseCol(A, dst, add_const, function); }

IuStatus sumSparseCol(iu::SparseMatrixGpu<float>* A, iu::VolumeGpu_32f_C1* dst, float add_const, IuSparseSum function)
{ return iuprivate::sumSparseCol(A, dst, add_const, function); }



} // namespace iu
