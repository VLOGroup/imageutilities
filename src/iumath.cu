
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

// multiply constant
void mulC(iu::ImageGpu_32f_C1& src, const float& val, iu::ImageGpu_32f_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::ImageGpu_32f_C2& src, const float2& val, iu::ImageGpu_32f_C2& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::ImageGpu_32f_C3& src, const float3& val, iu::ImageGpu_32f_C3& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::ImageGpu_32f_C4& src, const float4& val, iu::ImageGpu_32f_C4& dst) {iuprivate::math::mulC(src,val,dst);}

void mulC(iu::ImageCpu_32f_C1& src, const float& val, iu::ImageCpu_32f_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::ImageCpu_32f_C2& src, const float2& val, iu::ImageCpu_32f_C2& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::ImageCpu_32f_C3& src, const float3& val, iu::ImageCpu_32f_C3& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::ImageCpu_32f_C4& src, const float4& val, iu::ImageCpu_32f_C4& dst) {iuprivate::math::mulC(src,val,dst);}


void mulC(iu::ImageGpu_32s_C1& src, const int& val, iu::ImageGpu_32s_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::ImageGpu_32u_C1& src, const unsigned int& val, iu::ImageGpu_32u_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::ImageGpu_16u_C1& src, const unsigned short& val, iu::ImageGpu_16u_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::ImageGpu_8u_C1& src, const unsigned char& val, iu::ImageGpu_8u_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::ImageGpu_8u_C2& src, const uchar2& val, iu::ImageGpu_8u_C2& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::ImageGpu_8u_C3& src, const uchar3& val, iu::ImageGpu_8u_C3& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::ImageGpu_8u_C4& src, const uchar4& val, iu::ImageGpu_8u_C4& dst) {iuprivate::math::mulC(src,val,dst);}

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

// pointwise multiply
void mul(iu::ImageGpu_32f_C1& src1, iu::ImageGpu_32f_C1& src2, iu::ImageGpu_32f_C1& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::ImageGpu_32f_C2& src1, iu::ImageGpu_32f_C2& src2, iu::ImageGpu_32f_C2& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::ImageGpu_32f_C3& src1, iu::ImageGpu_32f_C3& src2, iu::ImageGpu_32f_C3& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::ImageGpu_32f_C4& src1, iu::ImageGpu_32f_C4& src2, iu::ImageGpu_32f_C4& dst) {iuprivate::math::mul(src1,src2,dst);}

void mul(iu::ImageGpu_8u_C1& src1, iu::ImageGpu_8u_C1& src2, iu::ImageGpu_8u_C1& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::ImageGpu_8u_C4& src1, iu::ImageGpu_8u_C4& src2, iu::ImageGpu_8u_C4& dst) {iuprivate::math::mul(src1,src2,dst);}

void mul(iu::LinearDeviceMemory_32f_C1& src1, iu::LinearDeviceMemory_32f_C1& src2, iu::LinearDeviceMemory_32f_C1& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory_32f_C2& src1, iu::LinearDeviceMemory_32f_C2& src2, iu::LinearDeviceMemory_32f_C2& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory_32f_C3& src1, iu::LinearDeviceMemory_32f_C3& src2, iu::LinearDeviceMemory_32f_C3& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory_32f_C4& src1, iu::LinearDeviceMemory_32f_C4& src2, iu::LinearDeviceMemory_32f_C4& dst) {iuprivate::math::mul(src1,src2,dst);}

void mul(iu::LinearDeviceMemory_8u_C1& src1, iu::LinearDeviceMemory_8u_C1& src2, iu::LinearDeviceMemory_8u_C1& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory_8u_C4& src1, iu::LinearDeviceMemory_8u_C4& src2, iu::LinearDeviceMemory_8u_C4& dst) {iuprivate::math::mul(src1,src2,dst);}

// set value
void fill(iu::ImageGpu_32f_C1& dst, float value) {iuprivate::math::fill(dst,value);}
void fill(iu::ImageGpu_32f_C2& dst, float2 value) {iuprivate::math::fill(dst,value);}
void fill(iu::ImageGpu_32f_C4& dst, float4 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearDeviceMemory_32f_C1& dst, float value) {iuprivate::math::fill(dst,value);}

void fill(iu::ImageCpu_32f_C1& dst, float value) {iuprivate::math::fill(dst,value);}
void fill(iu::ImageCpu_32f_C2& dst, float2 value) {iuprivate::math::fill(dst,value);}
void fill(iu::ImageCpu_32f_C4& dst, float4 value) {iuprivate::math::fill(dst,value);}
void fill(iu::ImageCpu_8u_C1& dst, unsigned char value) {iuprivate::math::fill(dst,value);}
void fill(iu::ImageCpu_8u_C2& dst, uchar2 value) {iuprivate::math::fill(dst,value);}
void fill(iu::ImageCpu_8u_C4& dst, uchar4 value) {iuprivate::math::fill(dst,value);}

void fill(iu::LinearHostMemory_32f_C1& dst, float value) {iuprivate::math::fill(dst,value);}

// min-max
void minMax(iu::ImageGpu_32f_C1& src, float& minVal, float& maxVal) {iuprivate::math::minMax(src,minVal,maxVal);}
void minMax(iu::LinearDeviceMemory_32f_C1& src, float& minVal, float& maxVal, unsigned int& minIdx, unsigned int& maxIdx) {iuprivate::math::minMax(src,minVal,maxVal,minIdx,maxIdx);}

//sum
void summation(iu::ImageGpu_32f_C1& src, float& sum) {iuprivate::math::summation(src,0.f,sum);}
void summation(iu::ImageGpu_32f_C2& src, float2& sum) {iuprivate::math::summation(src,make_float2(0.f),sum);}
void summation(iu::LinearDeviceMemory_32f_C1& src, float& sum) {iuprivate::math::summation(src,0.f,sum);}
} //namespace math
} //namespace iu

