
#include "iumath.h"
#include "iucore.h"

#include "iumath/arithmetics.cuh"
#include "iumath/statistics.cuh"

namespace iu {
namespace math {

// add constant
void addC(iu::ImageGpu_32f_C1& src, const float& val, iu::ImageGpu_32f_C1& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::ImageGpu_32f_C2& src, const float2& val, iu::ImageGpu_32f_C2& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::ImageGpu_32f_C3& src, const float3& val, iu::ImageGpu_32f_C3& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::ImageGpu_32f_C4& src, const float4& val, iu::ImageGpu_32f_C4& dst) {iuprivate::math::addC(src,val,dst);}

void addC(iu::ImageGpu_32s_C1& src, const int& val, iu::ImageGpu_32s_C1& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::ImageGpu_32u_C1& src, const unsigned int& val, iu::ImageGpu_32u_C1& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::ImageGpu_16u_C1& src, const unsigned short& val, iu::ImageGpu_16u_C1& dst) {iuprivate::math::addC(src,val,dst);}

void addC(iu::ImageGpu_8u_C1& src, const unsigned char& val, iu::ImageGpu_8u_C1& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::ImageGpu_8u_C2& src, const uchar2& val, iu::ImageGpu_8u_C2& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::ImageGpu_8u_C3& src, const uchar3& val, iu::ImageGpu_8u_C3& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::ImageGpu_8u_C4& src, const uchar4& val, iu::ImageGpu_8u_C4& dst) {iuprivate::math::addC(src,val,dst);}

void addC(iu::VolumeGpu_32f_C1& src, const float& val, iu::VolumeGpu_32f_C1& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::VolumeGpu_32f_C2& src, const float2& val, iu::VolumeGpu_32f_C2& dst) {iuprivate::math::addC(src,val,dst);}

void addC(iu::LinearDeviceMemory_32f_C1& src, const float& val, iu::LinearDeviceMemory_32f_C1& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory_32f_C2& src, const float2& val, iu::LinearDeviceMemory_32f_C2& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory_32f_C3& src, const float3& val, iu::LinearDeviceMemory_32f_C3& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory_32f_C4& src, const float4& val, iu::LinearDeviceMemory_32f_C4& dst) {iuprivate::math::addC(src,val,dst);}

void addC(iu::LinearDeviceMemory_32s_C1& src, const int& val, iu::LinearDeviceMemory_32s_C1& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory_32u_C1& src, const unsigned int& val, iu::LinearDeviceMemory_32u_C1& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory_16u_C1& src, const unsigned short& val, iu::LinearDeviceMemory_16u_C1& dst) {iuprivate::math::addC(src,val,dst);}

void addC(iu::LinearDeviceMemory_8u_C1& src, const unsigned char& val, iu::LinearDeviceMemory_8u_C1& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory_8u_C2& src, const uchar2& val, iu::LinearDeviceMemory_8u_C2& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory_8u_C3& src, const uchar3& val, iu::LinearDeviceMemory_8u_C3& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory_8u_C4& src, const uchar4& val, iu::LinearDeviceMemory_8u_C4& dst) {iuprivate::math::addC(src,val,dst);}

void addC(iu::ImageCpu_32f_C1& src, const float& val, iu::ImageCpu_32f_C1& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::ImageCpu_32f_C2& src, const float2& val, iu::ImageCpu_32f_C2& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::ImageCpu_32f_C3& src, const float3& val, iu::ImageCpu_32f_C3& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::ImageCpu_32f_C4& src, const float4& val, iu::ImageCpu_32f_C4& dst) {iuprivate::math::addC(src,val,dst);}

void addC(iu::ImageCpu_32s_C1& src, const int& val, iu::ImageCpu_32s_C1& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::ImageCpu_32u_C1& src, const unsigned int& val, iu::ImageCpu_32u_C1& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::ImageCpu_16u_C1& src, const unsigned short& val, iu::ImageCpu_16u_C1& dst) {iuprivate::math::addC(src,val,dst);}

void addC(iu::ImageCpu_8u_C1& src, const unsigned char& val, iu::ImageCpu_8u_C1& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::ImageCpu_8u_C2& src, const uchar2& val, iu::ImageCpu_8u_C2& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::ImageCpu_8u_C3& src, const uchar3& val, iu::ImageCpu_8u_C3& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::ImageCpu_8u_C4& src, const uchar4& val, iu::ImageCpu_8u_C4& dst) {iuprivate::math::addC(src,val,dst);}

void addC(iu::VolumeCpu_32f_C1& src, const float& val, iu::VolumeCpu_32f_C1& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::VolumeCpu_32f_C2& src, const float2& val, iu::VolumeCpu_32f_C2& dst) {iuprivate::math::addC(src,val,dst);}

void addC(iu::LinearHostMemory_32f_C1& src, const float& val, iu::LinearHostMemory_32f_C1& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory_32f_C2& src, const float2& val, iu::LinearHostMemory_32f_C2& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory_32f_C3& src, const float3& val, iu::LinearHostMemory_32f_C3& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory_32f_C4& src, const float4& val, iu::LinearHostMemory_32f_C4& dst) {iuprivate::math::addC(src,val,dst);}

void addC(iu::LinearHostMemory_32s_C1& src, const int& val, iu::LinearHostMemory_32s_C1& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory_32u_C1& src, const unsigned int& val, iu::LinearHostMemory_32u_C1& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory_16u_C1& src, const unsigned short& val, iu::LinearHostMemory_16u_C1& dst) {iuprivate::math::addC(src,val,dst);}

void addC(iu::LinearHostMemory_8u_C1& src, const unsigned char& val, iu::LinearHostMemory_8u_C1& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory_8u_C2& src, const uchar2& val, iu::LinearHostMemory_8u_C2& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory_8u_C3& src, const uchar3& val, iu::LinearHostMemory_8u_C3& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory_8u_C4& src, const uchar4& val, iu::LinearHostMemory_8u_C4& dst) {iuprivate::math::addC(src,val,dst);}

// multiply constant
void mulC(iu::ImageGpu_32f_C1& src, const float& val, iu::ImageGpu_32f_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::ImageGpu_32f_C2& src, const float2& val, iu::ImageGpu_32f_C2& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::ImageGpu_32f_C3& src, const float3& val, iu::ImageGpu_32f_C3& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::ImageGpu_32f_C4& src, const float4& val, iu::ImageGpu_32f_C4& dst) {iuprivate::math::mulC(src,val,dst);}

void mulC(iu::ImageGpu_32s_C1& src, const int& val, iu::ImageGpu_32s_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::ImageGpu_32u_C1& src, const unsigned int& val, iu::ImageGpu_32u_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::ImageGpu_16u_C1& src, const unsigned short& val, iu::ImageGpu_16u_C1& dst) {iuprivate::math::mulC(src,val,dst);}

void mulC(iu::ImageGpu_8u_C1& src, const unsigned char& val, iu::ImageGpu_8u_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::ImageGpu_8u_C2& src, const uchar2& val, iu::ImageGpu_8u_C2& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::ImageGpu_8u_C3& src, const uchar3& val, iu::ImageGpu_8u_C3& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::ImageGpu_8u_C4& src, const uchar4& val, iu::ImageGpu_8u_C4& dst) {iuprivate::math::mulC(src,val,dst);}

void mulC(iu::VolumeGpu_32f_C1& src, const float& val, iu::VolumeGpu_32f_C1& dst) {iuprivate::math::mulC(src,val,dst);}

void mulC(iu::LinearDeviceMemory_32f_C1& src, const float& val, iu::LinearDeviceMemory_32f_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory_32f_C2& src, const float2& val, iu::LinearDeviceMemory_32f_C2& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory_32f_C3& src, const float3& val, iu::LinearDeviceMemory_32f_C3& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory_32f_C4& src, const float4& val, iu::LinearDeviceMemory_32f_C4& dst) {iuprivate::math::mulC(src,val,dst);}

void mulC(iu::LinearDeviceMemory_32s_C1& src, const int& val, iu::LinearDeviceMemory_32s_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory_32u_C1& src, const unsigned int& val, iu::LinearDeviceMemory_32u_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory_16u_C1& src, const unsigned short& val, iu::LinearDeviceMemory_16u_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory_8u_C1& src, const unsigned char& val, iu::LinearDeviceMemory_8u_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory_8u_C2& src, const uchar2& val, iu::LinearDeviceMemory_8u_C2& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory_8u_C3& src, const uchar3& val, iu::LinearDeviceMemory_8u_C3& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory_8u_C4& src, const uchar4& val, iu::LinearDeviceMemory_8u_C4& dst) {iuprivate::math::mulC(src,val,dst);}

void mulC(iu::ImageCpu_32f_C1& src, const float& val, iu::ImageCpu_32f_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::ImageCpu_32f_C2& src, const float2& val, iu::ImageCpu_32f_C2& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::ImageCpu_32f_C3& src, const float3& val, iu::ImageCpu_32f_C3& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::ImageCpu_32f_C4& src, const float4& val, iu::ImageCpu_32f_C4& dst) {iuprivate::math::mulC(src,val,dst);}

void mulC(iu::ImageCpu_32s_C1& src, const int& val, iu::ImageCpu_32s_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::ImageCpu_32u_C1& src, const unsigned int& val, iu::ImageCpu_32u_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::ImageCpu_16u_C1& src, const unsigned short& val, iu::ImageCpu_16u_C1& dst) {iuprivate::math::mulC(src,val,dst);}

void mulC(iu::ImageCpu_8u_C1& src, const unsigned char& val, iu::ImageCpu_8u_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::ImageCpu_8u_C2& src, const uchar2& val, iu::ImageCpu_8u_C2& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::ImageCpu_8u_C3& src, const uchar3& val, iu::ImageCpu_8u_C3& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::ImageCpu_8u_C4& src, const uchar4& val, iu::ImageCpu_8u_C4& dst) {iuprivate::math::mulC(src,val,dst);}

void mulC(iu::VolumeCpu_32f_C1& src, const float& val, iu::VolumeCpu_32f_C1& dst) {iuprivate::math::mulC(src,val,dst);}

void mulC(iu::LinearHostMemory_32f_C1& src, const float& val, iu::LinearHostMemory_32f_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory_32f_C2& src, const float2& val, iu::LinearHostMemory_32f_C2& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory_32f_C3& src, const float3& val, iu::LinearHostMemory_32f_C3& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory_32f_C4& src, const float4& val, iu::LinearHostMemory_32f_C4& dst) {iuprivate::math::mulC(src,val,dst);}

void mulC(iu::LinearHostMemory_32s_C1& src, const int& val, iu::LinearHostMemory_32s_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory_32u_C1& src, const unsigned int& val, iu::LinearHostMemory_32u_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory_16u_C1& src, const unsigned short& val, iu::LinearHostMemory_16u_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory_8u_C1& src, const unsigned char& val, iu::LinearHostMemory_8u_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory_8u_C2& src, const uchar2& val, iu::LinearHostMemory_8u_C2& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory_8u_C3& src, const uchar3& val, iu::LinearHostMemory_8u_C3& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory_8u_C4& src, const uchar4& val, iu::LinearHostMemory_8u_C4& dst) {iuprivate::math::mulC(src,val,dst);}

// pointwise weighted add
void addWeighted(iu::ImageGpu_32f_C1& src1, const float& weight1,
                 iu::ImageGpu_32f_C1& src2, const float& weight2,iu::ImageGpu_32f_C1& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::ImageGpu_32f_C2& src1, const float2& weight1,
                 iu::ImageGpu_32f_C2& src2, const float2& weight2,iu::ImageGpu_32f_C2& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::ImageGpu_32f_C3& src1, const float3& weight1,
                 iu::ImageGpu_32f_C3& src2, const float3& weight2,iu::ImageGpu_32f_C3& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::ImageGpu_32f_C4& src1, const float4& weight1,
                 iu::ImageGpu_32f_C4& src2, const float4& weight2,iu::ImageGpu_32f_C4& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::ImageGpu_8u_C1& src1, const unsigned char& weight1,
                 iu::ImageGpu_8u_C1& src2, const unsigned char& weight2,iu::ImageGpu_8u_C1& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::ImageGpu_8u_C4& src1, const uchar4 &weight1,
                 iu::ImageGpu_8u_C4& src2, const uchar4& weight2, iu::ImageGpu_8u_C4& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}

void addWeighted(iu::VolumeGpu_32f_C1& src1, const float& weight1,
                 iu::VolumeGpu_32f_C1& src2, const float& weight2,iu::VolumeGpu_32f_C1& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}

void addWeighted(iu::LinearDeviceMemory_32f_C1& src1, const float& weight1,
                 iu::LinearDeviceMemory_32f_C1& src2, const float& weight2,iu::LinearDeviceMemory_32f_C1& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory_32f_C2& src1, const float2& weight1,
                 iu::LinearDeviceMemory_32f_C2& src2, const float2& weight2,iu::LinearDeviceMemory_32f_C2& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory_32f_C3& src1, const float3& weight1,
                 iu::LinearDeviceMemory_32f_C3& src2, const float3& weight2,iu::LinearDeviceMemory_32f_C3& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory_32f_C4& src1, const float4& weight1,
                 iu::LinearDeviceMemory_32f_C4& src2, const float4& weight2,iu::LinearDeviceMemory_32f_C4& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory_8u_C1& src1, const unsigned char& weight1,
                 iu::LinearDeviceMemory_8u_C1& src2, const unsigned char& weight2,iu::LinearDeviceMemory_8u_C1& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory_8u_C4& src1, const uchar4 &weight1,
                 iu::LinearDeviceMemory_8u_C4& src2, const uchar4& weight2, iu::LinearDeviceMemory_8u_C4& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}

void addWeighted(iu::ImageCpu_32f_C1& src1, const float& weight1,
                 iu::ImageCpu_32f_C1& src2, const float& weight2,iu::ImageCpu_32f_C1& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::ImageCpu_32f_C2& src1, const float2& weight1,
                 iu::ImageCpu_32f_C2& src2, const float2& weight2,iu::ImageCpu_32f_C2& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::ImageCpu_32f_C3& src1, const float3& weight1,
                 iu::ImageCpu_32f_C3& src2, const float3& weight2,iu::ImageCpu_32f_C3& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::ImageCpu_32f_C4& src1, const float4& weight1,
                 iu::ImageCpu_32f_C4& src2, const float4& weight2,iu::ImageCpu_32f_C4& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::ImageCpu_8u_C1& src1, const unsigned char& weight1,
                 iu::ImageCpu_8u_C1& src2, const unsigned char& weight2,iu::ImageCpu_8u_C1& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::ImageCpu_8u_C4& src1, const uchar4 &weight1,
                 iu::ImageCpu_8u_C4& src2, const uchar4& weight2, iu::ImageCpu_8u_C4& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}

void addWeighted(iu::VolumeCpu_32f_C1& src1, const float& weight1,
                 iu::VolumeCpu_32f_C1& src2, const float& weight2,iu::VolumeCpu_32f_C1& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}

void addWeighted(iu::LinearHostMemory_32f_C1& src1, const float& weight1,
                 iu::LinearHostMemory_32f_C1& src2, const float& weight2,iu::LinearHostMemory_32f_C1& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory_32f_C2& src1, const float2& weight1,
                 iu::LinearHostMemory_32f_C2& src2, const float2& weight2,iu::LinearHostMemory_32f_C2& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory_32f_C3& src1, const float3& weight1,
                 iu::LinearHostMemory_32f_C3& src2, const float3& weight2,iu::LinearHostMemory_32f_C3& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory_32f_C4& src1, const float4& weight1,
                 iu::LinearHostMemory_32f_C4& src2, const float4& weight2,iu::LinearHostMemory_32f_C4& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory_8u_C1& src1, const unsigned char& weight1,
                 iu::LinearHostMemory_8u_C1& src2, const unsigned char& weight2,iu::LinearHostMemory_8u_C1& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory_8u_C4& src1, const uchar4 &weight1,
                 iu::LinearHostMemory_8u_C4& src2, const uchar4& weight2, iu::LinearHostMemory_8u_C4& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}

// pointwise multiply
void mul(iu::ImageGpu_32f_C1& src1, iu::ImageGpu_32f_C1& src2, iu::ImageGpu_32f_C1& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::ImageGpu_32f_C2& src1, iu::ImageGpu_32f_C2& src2, iu::ImageGpu_32f_C2& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::ImageGpu_32f_C3& src1, iu::ImageGpu_32f_C3& src2, iu::ImageGpu_32f_C3& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::ImageGpu_32f_C4& src1, iu::ImageGpu_32f_C4& src2, iu::ImageGpu_32f_C4& dst) {iuprivate::math::mul(src1,src2,dst);}

void mul(iu::ImageGpu_8u_C1& src1, iu::ImageGpu_8u_C1& src2, iu::ImageGpu_8u_C1& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::ImageGpu_8u_C4& src1, iu::ImageGpu_8u_C4& src2, iu::ImageGpu_8u_C4& dst) {iuprivate::math::mul(src1,src2,dst);}

void mul(iu::VolumeGpu_32f_C1& src1, iu::VolumeGpu_32f_C1& src2, iu::VolumeGpu_32f_C1& dst) {iuprivate::math::mul(src1,src2,dst);}

void mul(iu::LinearDeviceMemory_32f_C1& src1, iu::LinearDeviceMemory_32f_C1& src2, iu::LinearDeviceMemory_32f_C1& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory_32f_C2& src1, iu::LinearDeviceMemory_32f_C2& src2, iu::LinearDeviceMemory_32f_C2& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory_32f_C3& src1, iu::LinearDeviceMemory_32f_C3& src2, iu::LinearDeviceMemory_32f_C3& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory_32f_C4& src1, iu::LinearDeviceMemory_32f_C4& src2, iu::LinearDeviceMemory_32f_C4& dst) {iuprivate::math::mul(src1,src2,dst);}

void mul(iu::LinearDeviceMemory_8u_C1& src1, iu::LinearDeviceMemory_8u_C1& src2, iu::LinearDeviceMemory_8u_C1& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory_8u_C4& src1, iu::LinearDeviceMemory_8u_C4& src2, iu::LinearDeviceMemory_8u_C4& dst) {iuprivate::math::mul(src1,src2,dst);}

void mul(iu::ImageCpu_32f_C1& src1, iu::ImageCpu_32f_C1& src2, iu::ImageCpu_32f_C1& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::ImageCpu_32f_C2& src1, iu::ImageCpu_32f_C2& src2, iu::ImageCpu_32f_C2& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::ImageCpu_32f_C3& src1, iu::ImageCpu_32f_C3& src2, iu::ImageCpu_32f_C3& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::ImageCpu_32f_C4& src1, iu::ImageCpu_32f_C4& src2, iu::ImageCpu_32f_C4& dst) {iuprivate::math::mul(src1,src2,dst);}

void mul(iu::ImageCpu_8u_C1& src1, iu::ImageCpu_8u_C1& src2, iu::ImageCpu_8u_C1& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::ImageCpu_8u_C4& src1, iu::ImageCpu_8u_C4& src2, iu::ImageCpu_8u_C4& dst) {iuprivate::math::mul(src1,src2,dst);}

void mul(iu::VolumeCpu_32f_C1& src1, iu::VolumeCpu_32f_C1& src2, iu::VolumeCpu_32f_C1& dst) {iuprivate::math::mul(src1,src2,dst);}

void mul(iu::LinearHostMemory_32f_C1& src1, iu::LinearHostMemory_32f_C1& src2, iu::LinearHostMemory_32f_C1& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory_32f_C2& src1, iu::LinearHostMemory_32f_C2& src2, iu::LinearHostMemory_32f_C2& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory_32f_C3& src1, iu::LinearHostMemory_32f_C3& src2, iu::LinearHostMemory_32f_C3& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory_32f_C4& src1, iu::LinearHostMemory_32f_C4& src2, iu::LinearHostMemory_32f_C4& dst) {iuprivate::math::mul(src1,src2,dst);}

void mul(iu::LinearHostMemory_8u_C1& src1, iu::LinearHostMemory_8u_C1& src2, iu::LinearHostMemory_8u_C1& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory_8u_C4& src1, iu::LinearHostMemory_8u_C4& src2, iu::LinearHostMemory_8u_C4& dst) {iuprivate::math::mul(src1,src2,dst);}

// set value
void fill(iu::ImageGpu_32f_C1& dst, float value) {iuprivate::math::fill(dst,value);}
void fill(iu::ImageGpu_32f_C2& dst, float2 value) {iuprivate::math::fill(dst,value);}
void fill(iu::ImageGpu_32f_C4& dst, float4 value) {iuprivate::math::fill(dst,value);}
void fill(iu::ImageGpu_8u_C1& dst, unsigned char value) {iuprivate::math::fill(dst,value);}
void fill(iu::ImageGpu_8u_C2& dst, uchar2 value) {iuprivate::math::fill(dst,value);}
void fill(iu::ImageGpu_8u_C4& dst, uchar4 value) {iuprivate::math::fill(dst,value);}

void fill(iu::ImageCpu_32f_C1& dst, float value) {iuprivate::math::fill(dst,value);}
void fill(iu::ImageCpu_32f_C2& dst, float2 value) {iuprivate::math::fill(dst,value);}
void fill(iu::ImageCpu_32f_C4& dst, float4 value) {iuprivate::math::fill(dst,value);}
void fill(iu::ImageCpu_8u_C1& dst, unsigned char value) {iuprivate::math::fill(dst,value);}
void fill(iu::ImageCpu_8u_C2& dst, uchar2 value) {iuprivate::math::fill(dst,value);}
void fill(iu::ImageCpu_8u_C4& dst, uchar4 value) {iuprivate::math::fill(dst,value);}

void fill(iu::LinearDeviceMemory_32f_C1& dst, float value) {iuprivate::math::fill(dst,value);}

void fill(iu::LinearHostMemory_32f_C1& dst, float value) {iuprivate::math::fill(dst,value);}

void fill(iu::VolumeGpu_32f_C1& dst, float value) {iuprivate::math::fill(dst,value);}
void fill(iu::VolumeGpu_32f_C2& dst, float2 value) {iuprivate::math::fill(dst,value);}

void fill(iu::VolumeCpu_32f_C1& dst, float value) {iuprivate::math::fill(dst,value);}
void fill(iu::VolumeCpu_32f_C2& dst, float2 value) {iuprivate::math::fill(dst,value);}

// min-max
void minMax(iu::ImageGpu_32f_C1& src, float& minVal, float& maxVal) {iuprivate::math::minMax(src,minVal,maxVal);}
void minMax(iu::VolumeGpu_32f_C1& src, float& minVal, float& maxVal) {iuprivate::math::minMax(src,minVal,maxVal);}

void minMax(iu::ImageCpu_32f_C1& src, float& minVal, float& maxVal) {iuprivate::math::minMax(src,minVal,maxVal);}
void minMax(iu::VolumeCpu_32f_C1& src, float& minVal, float& maxVal) {iuprivate::math::minMax(src,minVal,maxVal);}

void minMax(iu::LinearDeviceMemory_32f_C1& src, float& minVal, float& maxVal, unsigned int& minIdx, unsigned int& maxIdx) {iuprivate::math::minMax(src,minVal,maxVal,minIdx,maxIdx);}
void minMax(iu::LinearHostMemory_32f_C1& src, float& minVal, float& maxVal, unsigned int& minIdx, unsigned int& maxIdx) {iuprivate::math::minMax(src,minVal,maxVal,minIdx,maxIdx);}

//sum
void summation(iu::ImageGpu_32f_C1& src, float& sum) {iuprivate::math::summation(src,0.f,sum);}
void summation(iu::VolumeGpu_32f_C1& src, float& sum) {iuprivate::math::summation(src,0.f,sum);}
void summation(iu::LinearDeviceMemory_32f_C1& src, float& sum) {iuprivate::math::summation(src,0.f,sum);}

void summation(iu::ImageCpu_32f_C1& src, float& sum) {iuprivate::math::summation(src,0.f,sum);}
void summation(iu::VolumeCpu_32f_C1& src, float& sum) {iuprivate::math::summation(src,0.f,sum);}
void summation(iu::LinearHostMemory_32f_C1& src, float& sum) {iuprivate::math::summation(src,0.f,sum);}


// L1-norm
IUCORE_DLLAPI void normDiffL1(iu::ImageGpu_32f_C1& src, iu::ImageGpu_32f_C1& ref, float& norm) {iuprivate::math::normDiffL1(src,ref,norm);}
IUCORE_DLLAPI void normDiffL1(iu::ImageGpu_32f_C1& src, float& ref, float& norm) {iuprivate::math::normDiffL1(src,ref,norm);}
IUCORE_DLLAPI void normDiffL1(iu::VolumeGpu_32f_C1& src, iu::VolumeGpu_32f_C1& ref, float& norm) {iuprivate::math::normDiffL1(src,ref,norm);}
IUCORE_DLLAPI void normDiffL1(iu::VolumeGpu_32f_C1& src, float& ref, float& norm) {iuprivate::math::normDiffL1(src,ref,norm);}

IUCORE_DLLAPI void normDiffL1(iu::ImageCpu_32f_C1& src, iu::ImageCpu_32f_C1& ref, float& norm) {iuprivate::math::normDiffL1(src,ref,norm);}
IUCORE_DLLAPI void normDiffL1(iu::ImageCpu_32f_C1& src, float& ref, float& norm) {iuprivate::math::normDiffL1(src,ref,norm);}
IUCORE_DLLAPI void normDiffL1(iu::VolumeCpu_32f_C1& src, iu::VolumeCpu_32f_C1& ref, float& norm) {iuprivate::math::normDiffL1(src,ref,norm);}
IUCORE_DLLAPI void normDiffL1(iu::VolumeCpu_32f_C1& src, float& ref, float& norm) {iuprivate::math::normDiffL1(src,ref,norm);}


// L2-norm
IUCORE_DLLAPI void normDiffL2(iu::ImageGpu_32f_C1& src, iu::ImageGpu_32f_C1& ref, float& norm) {iuprivate::math::normDiffL2(src,ref,norm);}
IUCORE_DLLAPI void normDiffL2(iu::ImageGpu_32f_C1& src, float& ref, float& norm) {iuprivate::math::normDiffL2(src,ref,norm);}
IUCORE_DLLAPI void normDiffL2(iu::VolumeGpu_32f_C1& src, iu::VolumeGpu_32f_C1& ref, float& norm) {iuprivate::math::normDiffL2(src,ref,norm);}
IUCORE_DLLAPI void normDiffL2(iu::VolumeGpu_32f_C1& src, float& ref, float& norm) {iuprivate::math::normDiffL2(src,ref,norm);}

IUCORE_DLLAPI void normDiffL2(iu::ImageCpu_32f_C1& src, iu::ImageCpu_32f_C1& ref, float& norm) {iuprivate::math::normDiffL2(src,ref,norm);}
IUCORE_DLLAPI void normDiffL2(iu::ImageCpu_32f_C1& src, float& ref, float& norm) {iuprivate::math::normDiffL2(src,ref,norm);}
IUCORE_DLLAPI void normDiffL2(iu::VolumeCpu_32f_C1& src, iu::VolumeCpu_32f_C1& ref, float& norm) {iuprivate::math::normDiffL2(src,ref,norm);}
IUCORE_DLLAPI void normDiffL2(iu::VolumeCpu_32f_C1& src, float& ref, float& norm) {iuprivate::math::normDiffL2(src,ref,norm);}

// MSE
IUCORE_DLLAPI void mse(iu::ImageGpu_32f_C1& src, iu::ImageGpu_32f_C1& ref, float& mse) {iuprivate::math::mse(src,ref,mse);}
IUCORE_DLLAPI void mse(iu::VolumeGpu_32f_C1& src, iu::VolumeGpu_32f_C1& ref, float& mse) {iuprivate::math::mse(src,ref,mse);}

IUCORE_DLLAPI void mse(iu::ImageCpu_32f_C1& src, iu::ImageCpu_32f_C1& ref, float& mse) {iuprivate::math::mse(src,ref,mse);}
IUCORE_DLLAPI void mse(iu::VolumeCpu_32f_C1& src, iu::VolumeCpu_32f_C1& ref, float& mse) {iuprivate::math::mse(src,ref,mse);}

} //namespace math
} //namespace iu

