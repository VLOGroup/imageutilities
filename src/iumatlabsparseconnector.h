#ifndef IUMATLABSPARSECONNECTOR_H
#define IUMATLABSPARSECONNECTOR_H

#include "iudefs.h"
#include <iusparse/sparsematrixdefs.h>
#include <mex.h>

namespace iu {

/** Converts matlab sparse matrix to iu::SparseMatrixCpu format
 * \param src  Matlab array containing complete sparse matrix
 * \param dst  Destination sparse matrix on the host.
 */
IUMATLAB_DLLAPI IuStatus convertSparseMatrixToCpu(const mxArray* src, iu::SparseMatrixCpu_32f** dst);

/** Converts matlab sparse matrix to iu::SparseMatrixGpu format
 * \param handle Handle to the CUDA Sparse matrix library
 * \param src    Matlab array containing complete sparse matrix
 * \param dst    Destination sparse matrix on the device.
 */
IUMATLAB_DLLAPI IuStatus convertSparseMatrixToGpu(cusparseHandle_t* handle, const mxArray* src,
                                                  iu::SparseMatrixGpu_32f** dst);

} // namespace iu

#endif // IUMATLABSPARSECONNECTOR_H
