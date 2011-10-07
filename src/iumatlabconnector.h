#ifndef IUMATLABCONNECTOR_H
#define IUMATLABCONNECTOR_H

#include "iudefs.h"


namespace iu {

/*
  MATLAB connectors
 */

/** Converts matlab memory layout to an ImageCpu memory layout.
 * \param matlab_src_buffer Matlab image buffer.
 * \param width Width of the matlab image.
 * \param height Height of the matlab image.
 * \param dst Destination image on the host.
 */
IUMATLAB_DLLAPI IuStatus convertMatlabToCpu(double* matlab_src_buffer, unsigned int width, unsigned int height,
                                            iu::ImageCpu_32f_C1 *dst);


/** Converts matlab memory layout to an ImageCpu memory layout - float/single version
 * \param matlab_src_buffer Matlab image buffer.
 * \param width Width of the matlab image.
 * \param height Height of the matlab image.
 * \param dst Destination image on the host.
 */
IUMATLAB_DLLAPI IuStatus convertMatlabToCpu(float* matlab_src_buffer, unsigned int width, unsigned int height,
                                            iu::ImageCpu_32f_C1 *dst);

/** Converts matlab memory layout to an ImageCpu memory layout - int version
 * \param matlab_src_buffer Matlab image buffer.
 * \param width Width of the matlab image.
 * \param height Height of the matlab image.
 * \param dst Destination image on the host.
 */
IUMATLAB_DLLAPI IuStatus convertMatlabToCpu(int* matlab_src_buffer, unsigned int width, unsigned int height,
                                            iu::ImageCpu_32s_C1 *dst);

/** Converts matlab memory layout to an ImageGpu memory layout.
 * \param matlab_src_buffer Matlab image buffer.
 * \param width Width of the matlab image.
 * \param height Height of the matlab image.
 * \param dst Destination image on the device.
 */
IUMATLAB_DLLAPI IuStatus convertMatlabToGpu(double* matlab_src_buffer, unsigned int width, unsigned int height,
                                            iu::ImageGpu_32f_C1 *dst);

/** Converts matlab memory layout to an ImageGpu memory layout.
 * \param matlab_src_buffer Matlab image buffer.
 * \param width Width of the matlab image.
 * \param height Height of the matlab image.
 * \param dst Destination image on the device.
 */
IUMATLAB_DLLAPI IuStatus convertMatlabToGpu(int* matlab_src_buffer, unsigned int width, unsigned int height,
                                            iu::ImageGpu_32s_C1 *dst);


/** Converts 3-channel matlab memory layout to an 4-channel ImageCpu memory layout.
 * \param matlab_src_buffer Matlab image buffer.
 * \param width Width of the matlab image.
 * \param height Height of the matlab image.
 * \param dst Destination image on the device.
 */
IUMATLAB_DLLAPI IuStatus convertMatlabC3ToCpuC4(double* matlab_src_buffer, unsigned int width, unsigned int height,
                                                iu::ImageCpu_32f_C4 *dst);

/** Converts 2-channel matlab memory layout to a 2-channel ImageCpu memory layout.
 * \param matlab_src_buffer Matlab image buffer.
 * \param width Width of the matlab image.
 * \param height Height of the matlab image.
 * \param dst Destination image on the device.
 */
IUMATLAB_DLLAPI IuStatus convertMatlabC2ToCpuC2(double *matlab_src_buffer, unsigned int width, unsigned int height,
                                                iu::ImageCpu_32f_C2 *dst);

/** Converts 4-channel matlab memory layout to an 4-channel ImageCpu memory layout.
 * \param matlab_src_buffer Matlab image buffer.
 * \param width Width of the matlab image.
 * \param height Height of the matlab image.
 * \param dst Destination image on the device.
 */
IUMATLAB_DLLAPI IuStatus convertMatlabC4ToCpuC4(double *matlab_src_buffer, unsigned int width, unsigned int height,
                                                iu::ImageCpu_32f_C4 *dst);

/** Converts 3-channel matlab memory layout to an 4-channel ImageGpu memory layout.
 * \param matlab_src_buffer Matlab image buffer.
 * \param width Width of the matlab image.
 * \param height Height of the matlab image.
 * \param dst Destination image on the device.
 */
IUMATLAB_DLLAPI IuStatus convertMatlabC3ToGpuC4(double* matlab_src_buffer, unsigned int width, unsigned int height,
                                                iu::ImageGpu_32f_C4 *dst);


/** Converts 2-channel matlab memory layout to an 2-channel ImageGpu memory layout.
 * \param matlab_src_buffer Matlab image buffer.
 * \param width Width of the matlab image.
 * \param height Height of the matlab image.
 * \param dst Destination image on the device.
 */
IUMATLAB_DLLAPI IuStatus convertMatlabC2ToGpuC2(double *matlab_src_buffer, unsigned int width, unsigned int height,
                                                iu::ImageGpu_32f_C2 *dst);


/** Converts 4-channel matlab memory layout to an 4-channel ImageGpu memory layout.
 * \param matlab_src_buffer Matlab image buffer.
 * \param width Width of the matlab image.
 * \param height Height of the matlab image.
 * \param dst Destination image on the device.
 */
IUMATLAB_DLLAPI IuStatus convertMatlabC4ToGpuC4(double *matlab_src_buffer, unsigned int width, unsigned int height,
                                                iu::ImageGpu_32f_C4 *dst);



/** Converts an 4-channel ImageCpu to 3-channel matlab memory layout memory layout.
 * \param src Source color image.
 * \param dst matlab_dst_buffer image in matlab layout.
 */
IUMATLAB_DLLAPI IuStatus convertCpuC4ToMatlabC3(iu::ImageCpu_32f_C4 *src, double* matlab_dst_buffer);


/** Converts an 4-channel ImageCpu to 4-channel matlab memory layout memory layout.
 * \param src Source color image.
 * \param dst matlab_dst_buffer image in matlab layout.
 */
IUMATLAB_DLLAPI IuStatus convertCpuC4ToMatlabC4(iu::ImageCpu_32f_C4 *src, double* matlab_dst_buffer);

/** Converts an 2-channel ImageCpu to 2-channel matlab memory layout memory layout.
 * \param src Source color image.
 * \param dst matlab_dst_buffer image in matlab layout.
 */
IUMATLAB_DLLAPI IuStatus convertCpuC2ToMatlabC2(iu::ImageCpu_32f_C2 *src, double* matlab_dst_buffer);


/** Converts an 4-channel ImageGpu to 3-channel matlab memory layout memory layout.
 * \param src Source color image.
 * \param dst matlab_dst_buffer image in matlab layout.
 */
IUMATLAB_DLLAPI IuStatus convertGpuC4ToMatlabC3(iu::ImageGpu_32f_C4 *src, double* matlab_dst_buffer);


/** Converts an 2-channel ImageGpu to 2-channel matlab memory layout memory layout.
 * \param src Source color image.
 * \param dst matlab_dst_buffer image in matlab layout.
 */
IUMATLAB_DLLAPI IuStatus convertGpuC2ToMatlabC2(iu::ImageGpu_32f_C2 *src, double* matlab_dst_buffer);


/** Converts an 4-channel ImageGpu to 4-channel matlab memory layout memory layout.
 * \param src Source color image.
 * \param dst matlab_dst_buffer image in matlab layout.
 */
IUMATLAB_DLLAPI IuStatus convertGpuC4ToMatlabC4(iu::ImageGpu_32f_C4 *src, double* matlab_dst_buffer);


/** Converts matlab memory layout to an ImageCpu memory layout.
 * \param matlab_src_buffer Matlab image buffer.
 * \param width Width of the matlab image.
 * \param height Height of the matlab image.
 * \param dst Destination image on the host.
 */
IUMATLAB_DLLAPI IuStatus convertCpuToMatlab(ImageCpu_32f_C1 *src,
                                            double* matlab_dst_buffer, unsigned int width, unsigned int height);

/** Converts matlab memory layout to an ImageCpu memory layout.
 * \param matlab_src_buffer Matlab image buffer.
 * \param width Width of the matlab image.
 * \param height Height of the matlab image.
 * \param dst Destination image on the host.
 */
IUMATLAB_DLLAPI IuStatus convertCpuToMatlab(ImageCpu_8u_C1 *src,
                                            unsigned char* matlab_dst_buffer, unsigned int width, unsigned int height);

/** Converts matlab memory layout to an ImageGpu memory layout.
 * \param matlab_src_buffer Matlab image buffer.
 * \param width Width of the matlab image.
 * \param height Height of the matlab image.
 * \param dst Destination image on the device.
 */
IUMATLAB_DLLAPI IuStatus convertGpuToMatlab(ImageGpu_32f_C1 *src,
                                            double* matlab_dst_buffer, unsigned int width, unsigned int height);

/** Converts matlab memory layout to an ImageGpu memory layout.
 * \param matlab_src_buffer Matlab image buffer.
 * \param width Width of the matlab image.
 * \param height Height of the matlab image.
 * \param dst Destination image on the device.
 */
IUMATLAB_DLLAPI IuStatus convertGpuToMatlab(ImageGpu_8u_C1 *src,
                                            unsigned char* matlab_dst_buffer, unsigned int width, unsigned int height);



} // namespace iu

#endif // IUMATLABCONNECTOR_H
