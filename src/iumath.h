#ifndef IUMATH_H
#define IUMATH_H

#include "iudefs.h"

namespace iu {
namespace math {

//---------------------------------------------------------------------------------------------------
// ARITHMETICS

// add constant
IUCORE_DLLAPI void addC(iu::ImageGpu_32f_C1& src, const float& val, iu::ImageGpu_32f_C1& dst);
IUCORE_DLLAPI void addC(iu::ImageGpu_32f_C2& src, const float2& val, iu::ImageGpu_32f_C2& dst);
IUCORE_DLLAPI void addC(iu::ImageGpu_32f_C3& src, const float3& val, iu::ImageGpu_32f_C3& dst);
IUCORE_DLLAPI void addC(iu::ImageGpu_32f_C4& src, const float4& val, iu::ImageGpu_32f_C4& dst);

IUCORE_DLLAPI void addC(iu::ImageGpu_8u_C1& src, const unsigned char& val, iu::ImageGpu_8u_C1& dst);
IUCORE_DLLAPI void addC(iu::ImageGpu_8u_C2& src, const uchar2& val, iu::ImageGpu_8u_C2& dst);
IUCORE_DLLAPI void addC(iu::ImageGpu_8u_C3& src, const uchar3& val, iu::ImageGpu_8u_C3& dst);
IUCORE_DLLAPI void addC(iu::ImageGpu_8u_C4& src, const uchar4& val, iu::ImageGpu_8u_C4& dst);

IUCORE_DLLAPI void addC(iu::ImageGpu_32s_C1& src, const int& val, iu::ImageGpu_32s_C1& dst);
IUCORE_DLLAPI void addC(iu::ImageGpu_32u_C1& src, const unsigned int& val, iu::ImageGpu_32u_C1& dst);
IUCORE_DLLAPI void addC(iu::ImageGpu_16u_C1& src, const unsigned short& val, iu::ImageGpu_16u_C1& dst);

IUCORE_DLLAPI void addC(iu::LinearDeviceMemory_32f_C1& src, const float& val, iu::LinearDeviceMemory_32f_C1& dst);
IUCORE_DLLAPI void addC(iu::LinearDeviceMemory_32f_C2& src, const float2& val, iu::LinearDeviceMemory_32f_C2& dst);
IUCORE_DLLAPI void addC(iu::LinearDeviceMemory_32f_C3& src, const float3& val, iu::LinearDeviceMemory_32f_C3& dst);
IUCORE_DLLAPI void addC(iu::LinearDeviceMemory_32f_C4& src, const float4& val, iu::LinearDeviceMemory_32f_C4& dst);

IUCORE_DLLAPI void addC(iu::LinearDeviceMemory_8u_C1& src, const unsigned char& val, iu::LinearDeviceMemory_8u_C1& dst);
IUCORE_DLLAPI void addC(iu::LinearDeviceMemory_8u_C2& src, const uchar2& val, iu::LinearDeviceMemory_8u_C2& dst);
IUCORE_DLLAPI void addC(iu::LinearDeviceMemory_8u_C3& src, const uchar3& val, iu::LinearDeviceMemory_8u_C3& dst);
IUCORE_DLLAPI void addC(iu::LinearDeviceMemory_8u_C4& src, const uchar4& val, iu::LinearDeviceMemory_8u_C4& dst);

IUCORE_DLLAPI void addC(iu::LinearDeviceMemory_32s_C1& src, const int& val, iu::LinearDeviceMemory_32s_C1& dst);
IUCORE_DLLAPI void addC(iu::LinearDeviceMemory_32u_C1& src, const unsigned int& val, iu::LinearDeviceMemory_32u_C1& dst);
IUCORE_DLLAPI void addC(iu::LinearDeviceMemory_16u_C1& src, const unsigned short& val, iu::LinearDeviceMemory_16u_C1& dst);

// multiply constant
IUCORE_DLLAPI void mulC(iu::ImageGpu_32f_C1& src, const float& val, iu::ImageGpu_32f_C1& dst);
IUCORE_DLLAPI void mulC(iu::ImageGpu_32f_C2& src, const float2& val, iu::ImageGpu_32f_C2& dst);
IUCORE_DLLAPI void mulC(iu::ImageGpu_32f_C3& src, const float3& val, iu::ImageGpu_32f_C3& dst);
IUCORE_DLLAPI void mulC(iu::ImageGpu_32f_C4& src, const float4& val, iu::ImageGpu_32f_C4& dst);

IUCORE_DLLAPI void mulC(iu::ImageCpu_32f_C1& src, const float& val, iu::ImageCpu_32f_C1& dst);
IUCORE_DLLAPI void mulC(iu::ImageCpu_32f_C2& src, const float2& val, iu::ImageCpu_32f_C2& dst);
IUCORE_DLLAPI void mulC(iu::ImageCpu_32f_C3& src, const float3& val, iu::ImageCpu_32f_C3& dst);
IUCORE_DLLAPI void mulC(iu::ImageCpu_32f_C4& src, const float4& val, iu::ImageCpu_32f_C4& dst);


IUCORE_DLLAPI void mulC(iu::ImageGpu_8u_C1& src, const unsigned char& val, iu::ImageGpu_8u_C1& dst);
IUCORE_DLLAPI void mulC(iu::ImageGpu_8u_C2& src, const uchar2& val, iu::ImageGpu_8u_C2& dst);
IUCORE_DLLAPI void mulC(iu::ImageGpu_8u_C3& src, const uchar3& val, iu::ImageGpu_8u_C3& dst);
IUCORE_DLLAPI void mulC(iu::ImageGpu_8u_C4& src, const uchar4& val, iu::ImageGpu_8u_C4& dst);

IUCORE_DLLAPI void mulC(iu::ImageGpu_8u_C1& src, const unsigned char& val, iu::ImageGpu_8u_C1& dst);
IUCORE_DLLAPI void mulC(iu::ImageGpu_32s_C1& src, const int& val, iu::ImageGpu_32s_C1& dst);
IUCORE_DLLAPI void mulC(iu::ImageGpu_32u_C1& src, const unsigned int& val, iu::ImageGpu_32u_C1& dst);
IUCORE_DLLAPI void mulC(iu::ImageGpu_16u_C1& src, const unsigned short& val, iu::ImageGpu_16u_C1& dst);

IUCORE_DLLAPI void mulC(iu::LinearDeviceMemory_32f_C1& src, const float& val, iu::LinearDeviceMemory_32f_C1& dst);
IUCORE_DLLAPI void mulC(iu::LinearDeviceMemory_32f_C2& src, const float2& val, iu::LinearDeviceMemory_32f_C2& dst);
IUCORE_DLLAPI void mulC(iu::LinearDeviceMemory_32f_C3& src, const float3& val, iu::LinearDeviceMemory_32f_C3& dst);
IUCORE_DLLAPI void mulC(iu::LinearDeviceMemory_32f_C4& src, const float4& val, iu::LinearDeviceMemory_32f_C4& dst);

IUCORE_DLLAPI void mulC(iu::LinearDeviceMemory_8u_C1& src, const unsigned char& val, iu::LinearDeviceMemory_8u_C1& dst);
IUCORE_DLLAPI void mulC(iu::LinearDeviceMemory_8u_C2& src, const uchar2& val, iu::LinearDeviceMemory_8u_C2& dst);
IUCORE_DLLAPI void mulC(iu::LinearDeviceMemory_8u_C3& src, const uchar3& val, iu::LinearDeviceMemory_8u_C3& dst);
IUCORE_DLLAPI void mulC(iu::LinearDeviceMemory_8u_C4& src, const uchar4& val, iu::LinearDeviceMemory_8u_C4& dst);

IUCORE_DLLAPI void mulC(iu::LinearDeviceMemory_8u_C1& src, const unsigned char& val, iu::LinearDeviceMemory_8u_C1& dst);
IUCORE_DLLAPI void mulC(iu::LinearDeviceMemory_32s_C1& src, const int& val, iu::LinearDeviceMemory_32s_C1& dst);
IUCORE_DLLAPI void mulC(iu::LinearDeviceMemory_32u_C1& src, const unsigned int& val, iu::LinearDeviceMemory_32u_C1& dst);
IUCORE_DLLAPI void mulC(iu::LinearDeviceMemory_16u_C1& src, const unsigned short& val, iu::LinearDeviceMemory_16u_C1& dst);

// pointwise weighted add
IUCORE_DLLAPI void addWeighted(iu::ImageGpu_32f_C1& src1, const float& weight1,
                 iu::ImageGpu_32f_C1& src2, const float& weight2,iu::ImageGpu_32f_C1& dst);
IUCORE_DLLAPI void addWeighted(iu::ImageGpu_32f_C2& src1, const float2& weight1,
                 iu::ImageGpu_32f_C2& src2, const float2& weight2,iu::ImageGpu_32f_C2& dst);
IUCORE_DLLAPI void addWeighted(iu::ImageGpu_32f_C3& src1, const float3& weight1,
                 iu::ImageGpu_32f_C3& src2, const float3& weight2,iu::ImageGpu_32f_C3& dst);
IUCORE_DLLAPI void addWeighted(iu::ImageGpu_32f_C4& src1, const float4& weight1,
                 iu::ImageGpu_32f_C4& src2, const float4& weight2,iu::ImageGpu_32f_C4& dst);
IUCORE_DLLAPI void addWeighted(iu::ImageGpu_8u_C1& src1, const unsigned char& weight1,
                 iu::ImageGpu_8u_C1& src2, const unsigned char& weight2,iu::ImageGpu_8u_C1& dst);
IUCORE_DLLAPI void addWeighted(iu::ImageGpu_8u_C4& src1, const uchar4& weight1,
                 iu::ImageGpu_8u_C4& src2, const uchar4& weight2,iu::ImageGpu_8u_C4& dst);

IUCORE_DLLAPI void addWeighted(iu::LinearDeviceMemory_32f_C1& src1, const float& weight1,
                 iu::LinearDeviceMemory_32f_C1& src2, const float& weight2,iu::LinearDeviceMemory_32f_C1& dst);
IUCORE_DLLAPI void addWeighted(iu::LinearDeviceMemory_32f_C2& src1, const float2& weight1,
                 iu::LinearDeviceMemory_32f_C2& src2, const float2& weight2,iu::LinearDeviceMemory_32f_C2& dst);
IUCORE_DLLAPI void addWeighted(iu::LinearDeviceMemory_32f_C3& src1, const float3& weight1,
                 iu::LinearDeviceMemory_32f_C3& src2, const float3& weight2,iu::LinearDeviceMemory_32f_C3& dst);
IUCORE_DLLAPI void addWeighted(iu::LinearDeviceMemory_32f_C4& src1, const float4& weight1,
                 iu::LinearDeviceMemory_32f_C4& src2, const float4& weight2,iu::LinearDeviceMemory_32f_C4& dst);
IUCORE_DLLAPI void addWeighted(iu::LinearDeviceMemory_8u_C1& src1, const unsigned char& weight1,
                 iu::LinearDeviceMemory_8u_C1& src2, const unsigned char& weight2,iu::LinearDeviceMemory_8u_C1& dst);
IUCORE_DLLAPI void addWeighted(iu::LinearDeviceMemory_8u_C4& src1, const uchar4& weight1,
                 iu::LinearDeviceMemory_8u_C4& src2, const uchar4& weight2,iu::LinearDeviceMemory_8u_C4& dst);

// pointwise multiply
IUCORE_DLLAPI void mul(iu::ImageGpu_32f_C1& src1, iu::ImageGpu_32f_C1& src2, iu::ImageGpu_32f_C1& dst);
IUCORE_DLLAPI void mul(iu::ImageGpu_32f_C2& src1, iu::ImageGpu_32f_C2& src2, iu::ImageGpu_32f_C2& dst);
IUCORE_DLLAPI void mul(iu::ImageGpu_32f_C3& src1, iu::ImageGpu_32f_C3& src2, iu::ImageGpu_32f_C3& dst);
IUCORE_DLLAPI void mul(iu::ImageGpu_32f_C4& src1, iu::ImageGpu_32f_C4& src2, iu::ImageGpu_32f_C4& dst);

IUCORE_DLLAPI void mul(iu::ImageGpu_8u_C1& src1, iu::ImageGpu_8u_C1& src2, iu::ImageGpu_8u_C1& dst);
IUCORE_DLLAPI void mul(iu::ImageGpu_8u_C4& src1, iu::ImageGpu_8u_C4& src2, iu::ImageGpu_8u_C4& dst);

IUCORE_DLLAPI void mul(iu::LinearDeviceMemory_32f_C1& src1, iu::LinearDeviceMemory_32f_C1& src2, iu::LinearDeviceMemory_32f_C1& dst);
IUCORE_DLLAPI void mul(iu::LinearDeviceMemory_32f_C2& src1, iu::LinearDeviceMemory_32f_C2& src2, iu::LinearDeviceMemory_32f_C2& dst);
IUCORE_DLLAPI void mul(iu::LinearDeviceMemory_32f_C3& src1, iu::LinearDeviceMemory_32f_C3& src2, iu::LinearDeviceMemory_32f_C3& dst);
IUCORE_DLLAPI void mul(iu::LinearDeviceMemory_32f_C4& src1, iu::LinearDeviceMemory_32f_C4& src2, iu::LinearDeviceMemory_32f_C4& dst);

IUCORE_DLLAPI void mul(iu::LinearDeviceMemory_8u_C1& src1, iu::LinearDeviceMemory_8u_C1& src2, iu::LinearDeviceMemory_8u_C1& dst);
IUCORE_DLLAPI void mul(iu::LinearDeviceMemory_8u_C4& src1, iu::LinearDeviceMemory_8u_C4& src2, iu::LinearDeviceMemory_8u_C4& dst);

// set value
IUCORE_DLLAPI void fill(iu::ImageGpu_32f_C1& dst, float value);
IUCORE_DLLAPI void fill(iu::ImageGpu_32f_C2& dst, float2 value);
IUCORE_DLLAPI void fill(iu::ImageGpu_32f_C4& dst, float4 value);
IUCORE_DLLAPI void fill(iu::LinearDeviceMemory_32f_C1& dst, float value);

IUCORE_DLLAPI void fill(iu::ImageCpu_32f_C1& dst, float value);
IUCORE_DLLAPI void fill(iu::ImageCpu_32f_C2& dst, float2 value);
IUCORE_DLLAPI void fill(iu::ImageCpu_32f_C4& dst, float4 value);
IUCORE_DLLAPI void fill(iu::ImageCpu_8u_C1& dst, unsigned char value);
IUCORE_DLLAPI void fill(iu::ImageCpu_8u_C2& dst, uchar2 value);
IUCORE_DLLAPI void fill(iu::ImageCpu_8u_C4& dst, uchar4 value);

IUCORE_DLLAPI void fill(iu::LinearHostMemory_32f_C1& dst, float value);

//---------------------------------------------------------------------------------------------------
// STATISTICS
IUCORE_DLLAPI void minMax(iu::ImageGpu_32f_C1& src, float& minVal, float& maxVal);
IUCORE_DLLAPI void minMax(iu::LinearDeviceMemory_32f_C1& src, float& minVal, float& maxVal, unsigned int& minIdx, unsigned int& maxIdx);

IUCORE_DLLAPI void summation(iu::ImageGpu_32f_C1& src, float& sum);
IUCORE_DLLAPI void summation(iu::ImageGpu_32f_C2& src, float2& sum);
IUCORE_DLLAPI void summation(iu::LinearDeviceMemory_32f_C1& src, float& sum);
} // namespace math
} // namespace iu

#endif
