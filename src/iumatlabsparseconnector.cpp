#include "iumatlabsparseconnector.h"
#include <iusparse/sparsematrixdefs.h>
#include <iumatlab/matlabsparseconnector.h>

namespace iu {

/* ***************************************************************************
 *  MATLAB CONNECTORS
 * ***************************************************************************/

//
IuStatus convertSparseMatrixToCpu(const mxArray* src, iu::SparseMatrixCpu_32f** dst)
{return iuprivate::convertSparseMatrixToCpu_32f(src,dst);}

//
IuStatus convertSparseMatrixToGpu(cusparseHandle_t* handle, const mxArray* src,
                                  iu::SparseMatrixGpu_32f** dst)
{return iuprivate::convertSparseMatrixToGpu_32f(handle, src, dst);}

} // namespace iu
