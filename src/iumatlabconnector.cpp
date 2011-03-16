#include "iumatlabconnector.h"
#include <iumatlab/matlabconnector.h>

namespace iu {

/* ***************************************************************************
 *  MATLAB CONNECTORS
 * ***************************************************************************/

// matlab -> cpu: 32-bit; 1-channel
IuStatus convertMatlabToCpu(double* matlab_src_buffer, unsigned int width, unsigned int height,
                            iu::ImageCpu_32f_C1 *dst)
{ return iuprivate::convertMatlabToCpu(matlab_src_buffer, width, height, dst); }


// matlab(single) -> cpu: 32-bit; 1-channel
IuStatus convertMatlabToCpu(float* matlab_src_buffer, unsigned int width, unsigned int height,
                            iu::ImageCpu_32f_C1 *dst)
{ return iuprivate::convertMatlabToCpu(matlab_src_buffer, width, height, dst); }

// matlab(int) -> cpu: 32-bit; 1-channel
IuStatus convertMatlabToCpu(int* matlab_src_buffer, unsigned int width, unsigned int height,
                            iu::ImageCpu_32s_C1 *dst)
{ return iuprivate::convertMatlabToCpu(matlab_src_buffer, width, height, dst); }


// matlab -> gpu: 32-bit; 1-channel
IuStatus convertMatlabToGpu(double* matlab_src_buffer, unsigned int width, unsigned int height,
                            iu::ImageGpu_32f_C1 *dst)
{ return iuprivate::convertMatlabToGpu(matlab_src_buffer, width, height, dst); }

// matlab -> gpu: 32-bit; 1-channel
IuStatus convertMatlabToGpu(int* matlab_src_buffer, unsigned int width, unsigned int height,
                            iu::ImageGpu_32s_C1 *dst)
{ return iuprivate::convertMatlabToGpu(matlab_src_buffer, width, height, dst); }

// matlab -> cpu: 32-bit; 3-channel -> 4-channel
IuStatus convertMatlabC3ToCpuC4(double* matlab_src_buffer, unsigned int width, unsigned int height,
                                iu::ImageCpu_32f_C4 *dst)
{ return iuprivate::convertMatlabC3ToCpuC4(matlab_src_buffer, width, height, dst); }

// matlab -> cpu: 32-bit; 2-channel -> 2-channel
IuStatus convertMatlabC2ToCpuC2(double* matlab_src_buffer, unsigned int width, unsigned int height,
                                iu::ImageCpu_32f_C2 *dst)
{ return iuprivate::convertMatlabC2ToCpuC2(matlab_src_buffer, width, height, dst); }

// matlab -> cpu: 32-bit; 4-channel -> 4-channel
IuStatus convertMatlabC4ToCpuC4(double* matlab_src_buffer, unsigned int width, unsigned int height,
                                iu::ImageCpu_32f_C4 *dst)
{ return iuprivate::convertMatlabC4ToCpuC4(matlab_src_buffer, width, height, dst); }

// matlab -> gpu: 32-bit; 3-channel -> 4-channel
IuStatus convertMatlabC3ToGpuC4(double* matlab_src_buffer, unsigned int width, unsigned int height,
                                iu::ImageGpu_32f_C4 *dst)
{ return iuprivate::convertMatlabC3ToGpuC4(matlab_src_buffer, width, height, dst); }


// matlab -> gpu: 32-bit; 2-channel -> 2-channel
IuStatus convertMatlabC2ToGpuC2(double* matlab_src_buffer, unsigned int width, unsigned int height,
                                iu::ImageGpu_32f_C2 *dst)
{ return iuprivate::convertMatlabC2ToGpuC2(matlab_src_buffer, width, height, dst); }


// matlab -> gpu: 32-bit; 4-channel -> 4-channel
IuStatus convertMatlabC4ToGpuC4(double* matlab_src_buffer, unsigned int width, unsigned int height,
                                iu::ImageGpu_32f_C4 *dst)
{ return iuprivate::convertMatlabC4ToGpuC4(matlab_src_buffer, width, height, dst); }


// cpu -> matlab: 32-bit; 4-channel -> 3-channel
IuStatus convertCpuC4ToMatlabC3(iu::ImageCpu_32f_C4 *src, double* matlab_src_buffer)
{ return iuprivate::convertCpuC4ToMatlabC3(src, matlab_src_buffer); }

// cpu -> matlab: 32-bit; 2-channel -> 2-channel
IuStatus convertCpuC2ToMatlabC2(iu::ImageCpu_32f_C2 *src, double* matlab_src_buffer)
{ return iuprivate::convertCpuC2ToMatlabC2(src, matlab_src_buffer); }

// cpu -> matlab: 32-bit; 4-channel -> 4-channel
IuStatus convertCpuC4ToMatlabC4(iu::ImageCpu_32f_C4 *src, double* matlab_src_buffer)
{ return iuprivate::convertCpuC4ToMatlabC4(src, matlab_src_buffer); }

// gpu -> matlab: 32-bit; 4-channel -> 3-channel
IuStatus convertGpuC4ToMatlabC3(iu::ImageGpu_32f_C4 *src, double* matlab_src_buffer)
{ return iuprivate::convertGpuC4ToMatlabC3(src, matlab_src_buffer); }


// gpu -> matlab: 32-bit; 2-channel -> 2-channel
IuStatus convertGpuC2ToMatlabC2(iu::ImageGpu_32f_C2 *src, double* matlab_src_buffer)
{ return iuprivate::convertGpuC2ToMatlabC2(src, matlab_src_buffer); }


// gpu -> matlab: 32-bit; 4-channel -> 4-channel
IuStatus convertGpuC4ToMatlabC4(iu::ImageGpu_32f_C4 *src, double* matlab_src_buffer)
{ return iuprivate::convertGpuC4ToMatlabC4(src, matlab_src_buffer); }


// cpu -> matlab: 32-bit; 1-channel
IuStatus convertCpuToMatlab(iu::ImageCpu_32f_C1 *src,
                            double* matlab_dst_buffer, unsigned int width, unsigned int height)
{ return iuprivate::convertCpuToMatlab(src, matlab_dst_buffer, width, height); }

// cpu -> matlab: 8-bit; 1-channel
IuStatus convertCpuToMatlab(iu::ImageCpu_8u_C1 *src,
                            unsigned char* matlab_dst_buffer, unsigned int width, unsigned int height)
{ return iuprivate::convertCpuToMatlab(src, matlab_dst_buffer, width, height); }

// gpu -> matlab: 32-bit; 1-channel
IuStatus convertGpuToMatlab(iu::ImageGpu_32f_C1 *src,
                            double* matlab_dst_buffer, unsigned int width, unsigned int height)
{ return iuprivate::convertGpuToMatlab(src, matlab_dst_buffer, width, height); }

// gpu -> matlab: 8-bit; 1-channel
IuStatus convertGpuToMatlab(iu::ImageGpu_8u_C1 *src,
                            unsigned char* matlab_dst_buffer, unsigned int width, unsigned int height)
{ return iuprivate::convertGpuToMatlab(src, matlab_dst_buffer, width, height); }

} // namespace iu
