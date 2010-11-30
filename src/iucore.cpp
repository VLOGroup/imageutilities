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
 * Module      : Core Module
 * Class       : Wrapper
 * Language    : C
 * Description : Implementation of public interfaces to core module
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#include "iucore.h"
#include <iucore/copy.h>
#include <iucore/setvalue.h>
#include <iucore/clamp.h>
#include <iucore/convert.h>
#include <iuconnector/iumatlabconnector.h>

namespace iu {

/* ***************************************************************************
 * 1D COPY
 * ***************************************************************************/

// 1D copy host -> host;
void copy(const LinearHostMemory_8u_C1* src, LinearHostMemory_8u_C1* dst)
{ iuprivate::copy(src,dst); }
void copy(const LinearHostMemory_16u_C1* src, LinearHostMemory_16u_C1* dst)
{ iuprivate::copy(src,dst); }
void copy(const LinearHostMemory_32f_C1* src, LinearHostMemory_32f_C1* dst)
{ iuprivate::copy(src,dst); }

// 1D copy device -> device;
void copy(const LinearDeviceMemory_8u_C1* src, LinearDeviceMemory_8u_C1* dst){ iuprivate::copy(src,dst); }
void copy(const LinearDeviceMemory_8u_C2* src, LinearDeviceMemory_8u_C2* dst){ iuprivate::copy(src,dst); }
void copy(const LinearDeviceMemory_8u_C3* src, LinearDeviceMemory_8u_C3* dst){ iuprivate::copy(src,dst); }
void copy(const LinearDeviceMemory_8u_C4* src, LinearDeviceMemory_8u_C4* dst){ iuprivate::copy(src,dst); }
void copy(const LinearDeviceMemory_16u_C1* src, LinearDeviceMemory_16u_C1* dst){ iuprivate::copy(src,dst); }
void copy(const LinearDeviceMemory_16u_C2* src, LinearDeviceMemory_16u_C2* dst){ iuprivate::copy(src,dst); }
void copy(const LinearDeviceMemory_16u_C3* src, LinearDeviceMemory_16u_C3* dst){ iuprivate::copy(src,dst); }
void copy(const LinearDeviceMemory_16u_C4* src, LinearDeviceMemory_16u_C4* dst){ iuprivate::copy(src,dst); }
void copy(const LinearDeviceMemory_32f_C1* src, LinearDeviceMemory_32f_C1* dst){ iuprivate::copy(src,dst); }
void copy(const LinearDeviceMemory_32f_C2* src, LinearDeviceMemory_32f_C2* dst){ iuprivate::copy(src,dst); }
void copy(const LinearDeviceMemory_32f_C3* src, LinearDeviceMemory_32f_C3* dst){ iuprivate::copy(src,dst); }
void copy(const LinearDeviceMemory_32f_C4* src, LinearDeviceMemory_32f_C4* dst){ iuprivate::copy(src,dst); }

// 1D copy host -> device;
void copy(const LinearHostMemory_8u_C1* src, LinearDeviceMemory_8u_C1* dst){ iuprivate::copy(src,dst); }
void copy(const LinearHostMemory_8u_C2* src, LinearDeviceMemory_8u_C2* dst){ iuprivate::copy(src,dst); }
void copy(const LinearHostMemory_8u_C3* src, LinearDeviceMemory_8u_C3* dst){ iuprivate::copy(src,dst); }
void copy(const LinearHostMemory_8u_C4* src, LinearDeviceMemory_8u_C4* dst){ iuprivate::copy(src,dst); }
void copy(const LinearHostMemory_16u_C1* src, LinearDeviceMemory_16u_C1* dst){ iuprivate::copy(src,dst); }
void copy(const LinearHostMemory_16u_C2* src, LinearDeviceMemory_16u_C2* dst){ iuprivate::copy(src,dst); }
void copy(const LinearHostMemory_16u_C3* src, LinearDeviceMemory_16u_C3* dst){ iuprivate::copy(src,dst); }
void copy(const LinearHostMemory_16u_C4* src, LinearDeviceMemory_16u_C4* dst){ iuprivate::copy(src,dst); }
void copy(const LinearHostMemory_32f_C1* src, LinearDeviceMemory_32f_C1* dst){ iuprivate::copy(src,dst); }
void copy(const LinearHostMemory_32f_C2* src, LinearDeviceMemory_32f_C2* dst){ iuprivate::copy(src,dst); }
void copy(const LinearHostMemory_32f_C3* src, LinearDeviceMemory_32f_C3* dst){ iuprivate::copy(src,dst); }
void copy(const LinearHostMemory_32f_C4* src, LinearDeviceMemory_32f_C4* dst){ iuprivate::copy(src,dst); }

// 1D copy device -> host;
void copy(const LinearDeviceMemory_8u_C1* src, LinearHostMemory_8u_C1* dst){ iuprivate::copy(src,dst); }
void copy(const LinearDeviceMemory_8u_C2* src, LinearHostMemory_8u_C2* dst){ iuprivate::copy(src,dst); }
void copy(const LinearDeviceMemory_8u_C3* src, LinearHostMemory_8u_C3* dst){ iuprivate::copy(src,dst); }
void copy(const LinearDeviceMemory_8u_C4* src, LinearHostMemory_8u_C4* dst){ iuprivate::copy(src,dst); }
void copy(const LinearDeviceMemory_16u_C1* src, LinearHostMemory_16u_C1* dst){ iuprivate::copy(src,dst); }
void copy(const LinearDeviceMemory_16u_C2* src, LinearHostMemory_16u_C2* dst){ iuprivate::copy(src,dst); }
void copy(const LinearDeviceMemory_16u_C3* src, LinearHostMemory_16u_C3* dst){ iuprivate::copy(src,dst); }
void copy(const LinearDeviceMemory_16u_C4* src, LinearHostMemory_16u_C4* dst){ iuprivate::copy(src,dst); }
void copy(const LinearDeviceMemory_32f_C1* src, LinearHostMemory_32f_C1* dst){ iuprivate::copy(src,dst); }
void copy(const LinearDeviceMemory_32f_C2* src, LinearHostMemory_32f_C2* dst){ iuprivate::copy(src,dst); }
void copy(const LinearDeviceMemory_32f_C3* src, LinearHostMemory_32f_C3* dst){ iuprivate::copy(src,dst); }
void copy(const LinearDeviceMemory_32f_C4* src, LinearHostMemory_32f_C4* dst){ iuprivate::copy(src,dst); }

/* ***************************************************************************
 * 2D COPY
 * ***************************************************************************/

// 2D copy host -> host;
void copy(const ImageCpu_8u_C1* src, ImageCpu_8u_C1* dst) { iuprivate::copy(src, dst); }
void copy(const ImageCpu_8u_C2* src, ImageCpu_8u_C2* dst) { iuprivate::copy(src, dst); }
void copy(const ImageCpu_8u_C3* src, ImageCpu_8u_C3* dst) { iuprivate::copy(src, dst); }
void copy(const ImageCpu_8u_C4* src, ImageCpu_8u_C4* dst) { iuprivate::copy(src, dst); }
void copy(const ImageCpu_32f_C1* src, ImageCpu_32f_C1* dst) { iuprivate::copy(src, dst); }
void copy(const ImageCpu_32f_C2* src, ImageCpu_32f_C2* dst) { iuprivate::copy(src, dst); }
void copy(const ImageCpu_32f_C3* src, ImageCpu_32f_C3* dst) { iuprivate::copy(src, dst); }
void copy(const ImageCpu_32f_C4* src, ImageCpu_32f_C4* dst) { iuprivate::copy(src, dst); }

// 2D copy device -> device;
void copy(const ImageGpu_8u_C1* src, ImageGpu_8u_C1* dst) { iuprivate::copy(src, dst); }
void copy(const ImageGpu_8u_C2* src, ImageGpu_8u_C2* dst) { iuprivate::copy(src, dst); }
void copy(const ImageGpu_8u_C3* src, ImageGpu_8u_C3* dst) { iuprivate::copy(src, dst); }
void copy(const ImageGpu_8u_C4* src, ImageGpu_8u_C4* dst) { iuprivate::copy(src, dst); }
void copy(const ImageGpu_32f_C1* src, ImageGpu_32f_C1* dst) { iuprivate::copy(src, dst); }
void copy(const ImageGpu_32f_C2* src, ImageGpu_32f_C2* dst) { iuprivate::copy(src, dst); }
void copy(const ImageGpu_32f_C3* src, ImageGpu_32f_C3* dst) { iuprivate::copy(src, dst); }
void copy(const ImageGpu_32f_C4* src, ImageGpu_32f_C4* dst) { iuprivate::copy(src, dst); }

// 2D copy host -> device;
void copy(const ImageCpu_8u_C1* src, ImageGpu_8u_C1* dst) { iuprivate::copy(src, dst); }
void copy(const ImageCpu_8u_C2* src, ImageGpu_8u_C2* dst) { iuprivate::copy(src, dst); }
void copy(const ImageCpu_8u_C3* src, ImageGpu_8u_C3* dst) { iuprivate::copy(src, dst); }
void copy(const ImageCpu_8u_C4* src, ImageGpu_8u_C4* dst) { iuprivate::copy(src, dst); }
void copy(const ImageCpu_32f_C1* src, ImageGpu_32f_C1* dst) { iuprivate::copy(src, dst); }
void copy(const ImageCpu_32f_C2* src, ImageGpu_32f_C2* dst) { iuprivate::copy(src, dst); }
void copy(const ImageCpu_32f_C3* src, ImageGpu_32f_C3* dst) { iuprivate::copy(src, dst); }
void copy(const ImageCpu_32f_C4* src, ImageGpu_32f_C4* dst) { iuprivate::copy(src, dst); }

// 2D copy device -> host;
void copy(const ImageGpu_8u_C1* src, ImageCpu_8u_C1* dst) { iuprivate::copy(src, dst); }
void copy(const ImageGpu_8u_C2* src, ImageCpu_8u_C2* dst) { iuprivate::copy(src, dst); }
void copy(const ImageGpu_8u_C3* src, ImageCpu_8u_C3* dst) { iuprivate::copy(src, dst); }
void copy(const ImageGpu_8u_C4* src, ImageCpu_8u_C4* dst) { iuprivate::copy(src, dst); }
void copy(const ImageGpu_32f_C1* src, ImageCpu_32f_C1* dst) { iuprivate::copy(src, dst); }
void copy(const ImageGpu_32f_C2* src, ImageCpu_32f_C2* dst) { iuprivate::copy(src, dst); }
void copy(const ImageGpu_32f_C3* src, ImageCpu_32f_C3* dst) { iuprivate::copy(src, dst); }
void copy(const ImageGpu_32f_C4* src, ImageCpu_32f_C4* dst) { iuprivate::copy(src, dst); }


/* ***************************************************************************
 * 3D COPY
 * ***************************************************************************/

// 3D copy host -> host;
void copy(const VolumeCpu_8u_C1* src, VolumeCpu_8u_C1* dst) { iuprivate::copy(src, dst); }
void copy(const VolumeCpu_8u_C2* src, VolumeCpu_8u_C2* dst) { iuprivate::copy(src, dst); }
void copy(const VolumeCpu_8u_C4* src, VolumeCpu_8u_C4* dst) { iuprivate::copy(src, dst); }
void copy(const VolumeCpu_32f_C1* src, VolumeCpu_32f_C1* dst) { iuprivate::copy(src, dst); }
void copy(const VolumeCpu_32f_C2* src, VolumeCpu_32f_C2* dst) { iuprivate::copy(src, dst); }
void copy(const VolumeCpu_32f_C4* src, VolumeCpu_32f_C4* dst) { iuprivate::copy(src, dst); }

// 3D copy device -> device;
void copy(const VolumeGpu_8u_C1* src, VolumeGpu_8u_C1* dst) { iuprivate::copy(src, dst); }
void copy(const VolumeGpu_8u_C2* src, VolumeGpu_8u_C2* dst) { iuprivate::copy(src, dst); }
void copy(const VolumeGpu_8u_C4* src, VolumeGpu_8u_C4* dst) { iuprivate::copy(src, dst); }
void copy(const VolumeGpu_32f_C1* src, VolumeGpu_32f_C1* dst) { iuprivate::copy(src, dst); }
void copy(const VolumeGpu_32f_C2* src, VolumeGpu_32f_C2* dst) { iuprivate::copy(src, dst); }
void copy(const VolumeGpu_32f_C4* src, VolumeGpu_32f_C4* dst) { iuprivate::copy(src, dst); }

// 3D copy host -> device;
void copy(const VolumeCpu_8u_C1* src, VolumeGpu_8u_C1* dst) { iuprivate::copy(src, dst); }
void copy(const VolumeCpu_8u_C2* src, VolumeGpu_8u_C2* dst) { iuprivate::copy(src, dst); }
void copy(const VolumeCpu_8u_C4* src, VolumeGpu_8u_C4* dst) { iuprivate::copy(src, dst); }
void copy(const VolumeCpu_32f_C1* src, VolumeGpu_32f_C1* dst) { iuprivate::copy(src, dst); }
void copy(const VolumeCpu_32f_C2* src, VolumeGpu_32f_C2* dst) { iuprivate::copy(src, dst); }
void copy(const VolumeCpu_32f_C4* src, VolumeGpu_32f_C4* dst) { iuprivate::copy(src, dst); }

// 3D copy device -> host;
void copy(const VolumeGpu_8u_C1* src, VolumeCpu_8u_C1* dst) { iuprivate::copy(src, dst); }
void copy(const VolumeGpu_8u_C2* src, VolumeCpu_8u_C2* dst) { iuprivate::copy(src, dst); }
void copy(const VolumeGpu_8u_C4* src, VolumeCpu_8u_C4* dst) { iuprivate::copy(src, dst); }
void copy(const VolumeGpu_32f_C1* src, VolumeCpu_32f_C1* dst) { iuprivate::copy(src, dst); }
void copy(const VolumeGpu_32f_C2* src, VolumeCpu_32f_C2* dst) { iuprivate::copy(src, dst); }
void copy(const VolumeGpu_32f_C4* src, VolumeCpu_32f_C4* dst) { iuprivate::copy(src, dst); }


/* ***************************************************************************
     SET
 * ***************************************************************************/

// 1D set value; host; 8-bit
void setValue(const unsigned char& value, LinearHostMemory_8u_C1* srcdst)
{iuprivate::setValue(value, srcdst);}
void setValue(const float& value, LinearHostMemory_32f_C1* srcdst)
{iuprivate::setValue(value, srcdst);}
void setValue(const unsigned char& value, LinearDeviceMemory_8u_C1* srcdst)
{iuprivate::setValue(value, srcdst);}
void setValue(const float& value, LinearDeviceMemory_32f_C1* srcdst)
{iuprivate::setValue(value, srcdst);}

void setValue(const unsigned char &value, ImageCpu_8u_C1* srcdst, const IuRect& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const uchar2 &value, ImageCpu_8u_C2* srcdst, const IuRect& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const uchar3 &value, ImageCpu_8u_C3* srcdst, const IuRect& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const uchar4 &value, ImageCpu_8u_C4* srcdst, const IuRect& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const float &value, ImageCpu_32f_C1* srcdst, const IuRect& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const float2 &value, ImageCpu_32f_C2* srcdst, const IuRect& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const float3 &value, ImageCpu_32f_C3* srcdst, const IuRect& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const float4 &value, ImageCpu_32f_C4* srcdst, const IuRect& roi) {iuprivate::setValue(value, srcdst, roi);}

void setValue(const unsigned char &value, ImageGpu_8u_C1* srcdst, const IuRect& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const uchar2 &value, ImageGpu_8u_C2* srcdst, const IuRect& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const uchar3 &value, ImageGpu_8u_C3* srcdst, const IuRect& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const uchar4 &value, ImageGpu_8u_C4* srcdst, const IuRect& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const float &value, ImageGpu_32f_C1* srcdst, const IuRect& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const float2 &value, ImageGpu_32f_C2* srcdst, const IuRect& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const float3 &value, ImageGpu_32f_C3* srcdst, const IuRect& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const float4 &value, ImageGpu_32f_C4* srcdst, const IuRect& roi) {iuprivate::setValue(value, srcdst, roi);}

void setValue(const unsigned char &value, VolumeCpu_8u_C1* srcdst, const IuCube& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const uchar2 &value, VolumeCpu_8u_C2* srcdst, const IuCube& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const uchar4 &value, VolumeCpu_8u_C4* srcdst, const IuCube& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const float &value, VolumeCpu_32f_C1* srcdst, const IuCube& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const float2 &value, VolumeCpu_32f_C2* srcdst, const IuCube& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const float4 &value, VolumeCpu_32f_C4* srcdst, const IuCube& roi) {iuprivate::setValue(value, srcdst, roi);}

void setValue(const unsigned char &value, VolumeGpu_8u_C1* srcdst, const IuCube& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const uchar2 &value, VolumeGpu_8u_C2* srcdst, const IuCube& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const uchar4 &value, VolumeGpu_8u_C4* srcdst, const IuCube& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const float &value, VolumeGpu_32f_C1* srcdst, const IuCube& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const float2 &value, VolumeGpu_32f_C2* srcdst, const IuCube& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const float4 &value, VolumeGpu_32f_C4* srcdst, const IuCube& roi) {iuprivate::setValue(value, srcdst, roi);}


/* ***************************************************************************
     CLAMP
 * ***************************************************************************/

void clamp(const float& min, const float& max, iu::ImageGpu_32f_C1 *srcdst, const IuRect &roi)
{ iuprivate::clamp(min, max, srcdst, roi); }


/* ***************************************************************************
 *  MEMORY CONVERSIONS
 * ***************************************************************************/

// conversion; device; 32-bit 3-channel -> 32-bit 4-channel
void convert(const ImageGpu_32f_C3* src, const IuRect& src_roi, ImageGpu_32f_C4* dst, const IuRect& dst_roi)
{iuprivate::convert(src, src_roi, dst, dst_roi);}
// conversion; device; 32-bit 4-channel -> 32-bit 3-channel
void convert(const ImageGpu_32f_C4* src, const IuRect& src_roi, ImageGpu_32f_C3* dst, const IuRect& dst_roi)
{iuprivate::convert(src, src_roi, dst, dst_roi);}

// [host] 2D bit depth conversion; 32f_C1 -> 8u_C1;
void convert_32f8u_C1(const iu::ImageCpu_32f_C1* src, iu::ImageCpu_8u_C1* dst,
                       float mul_constant, float add_constant)
{iuprivate::convert_32f8u_C1(src, dst, mul_constant, add_constant);}

// [host] 2D bit depth conversion; 16u_C1 -> 32f_C1;
void convert_16u32f_C1(const iu::ImageCpu_16u_C1* src, iu::ImageCpu_32f_C1 *dst,
                       float mul_constant, float add_constant)
{iuprivate::convert_16u32f_C1(src, dst, mul_constant, add_constant);}

// [device] 2D bit depth conversion: 32f_C1 -> 8u_C1
void convert_32f8u_C1(const iu::ImageGpu_32f_C1* src, const IuRect& src_roi, iu::ImageGpu_8u_C1* dst, const IuRect& dst_roi,
                     float mul_constant, unsigned char add_constant)
{iuprivate::convert_32f8u_C1(src, src_roi, dst, dst_roi, mul_constant, add_constant);}

// [device] 2D bit depth conversion: 8u_C1 -> 32f_C1
void convert_8u32f_C1(const iu::ImageGpu_8u_C1* src, const IuRect& src_roi, iu::ImageGpu_32f_C1* dst, const IuRect& dst_roi,
                     float mul_constant, float add_constant)
{iuprivate::convert_8u32f_C1(src, src_roi, dst, dst_roi, mul_constant, add_constant);}

/* ***************************************************************************
 *  MATLAB CONNECTORS
 * ***************************************************************************/

// matlab -> cpu: 32-bit; 1-channel
IuStatus convertMatlabToCpu(double* matlab_src_buffer, unsigned int width, unsigned int height,
                            iu::ImageCpu_32f_C1 *dst)
{ return iuprivate::convertMatlabToCpu(matlab_src_buffer, width, height, dst); }


// matlab -> gpu: 32-bit; 1-channel
IuStatus convertMatlabToGpu(double* matlab_src_buffer, unsigned int width, unsigned int height,
                            iu::ImageGpu_32f_C1 *dst)
{ return iuprivate::convertMatlabToGpu(matlab_src_buffer, width, height, dst); }

// matlab -> cpu: 32-bit; 3-channel -> 4-channel
IuStatus convertMatlabC3ToCpuC4(double* matlab_src_buffer, unsigned int width, unsigned int height,
                                iu::ImageCpu_32f_C4 *dst)
{ return iuprivate::convertMatlabC3ToCpuC4(matlab_src_buffer, width, height, dst); }

// matlab -> gpu: 32-bit; 3-channel -> 4-channel
IuStatus convertMatlabC3ToGpuC4(double* matlab_src_buffer, unsigned int width, unsigned int height,
                                iu::ImageGpu_32f_C4 *dst)
{ return iuprivate::convertMatlabC3ToGpuC4(matlab_src_buffer, width, height, dst); }

// cpu -> matlab: 32-bit; 4-channel -> 3-channel
IuStatus convertCpuC4ToMatlabC3(iu::ImageCpu_32f_C4 *src, double* matlab_src_buffer)
{ return iuprivate::convertCpuC4ToMatlabC3(src, matlab_src_buffer); }

// gpu -> matlab: 32-bit; 4-channel -> 3-channel
IuStatus convertGpuC4ToMatlabC3(iu::ImageGpu_32f_C4 *src, double* matlab_src_buffer)
{ return iuprivate::convertGpuC4ToMatlabC3(src, matlab_src_buffer); }

// cpu -> matlab: 32-bit; 1-channel
IuStatus convertCpuToMatlab(iu::ImageCpu_32f_C1 *src,
                            double* matlab_dst_buffer, unsigned int width, unsigned int height)
{ return iuprivate::convertCpuToMatlab(src, matlab_dst_buffer, width, height); }

// gpu -> matlab: 32-bit; 1-channel
IuStatus convertGpuToMatlab(iu::ImageGpu_32f_C1 *src,
                            double* matlab_dst_buffer, unsigned int width, unsigned int height)
{ return iuprivate::convertGpuToMatlab(src, matlab_dst_buffer, width, height); }


} // namespace iu
