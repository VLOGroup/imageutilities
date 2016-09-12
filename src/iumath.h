#pragma once

#include "iumath/iumathapi.h"
#include "iudefs.h"

namespace iu {
/// Math related functions
namespace math {

/** \defgroup Math iumath
 * \brief Provides basic mathematics on arrays and images
 */

/** \defgroup MathArithmetics Arithmetics
 \ingroup Math
 \brief Pointwise image arithmetics
 * \{
 */
//---------------------------------------------------------------------------------------------------
// ARITHMETICS

// add constant
/** Add a constant to an array (can be called in-place)
 * \param src Source array
 * \param val Value to add
 * \param[out] dst Destination array
 */
IUMATH_DLLAPI void addC(iu::ImageGpu_32f_C1& src, const float& val, iu::ImageGpu_32f_C1& dst);
IUMATH_DLLAPI void addC(iu::ImageGpu_32f_C2& src, const float2& val, iu::ImageGpu_32f_C2& dst);
IUMATH_DLLAPI void addC(iu::ImageGpu_32f_C3& src, const float3& val, iu::ImageGpu_32f_C3& dst);
IUMATH_DLLAPI void addC(iu::ImageGpu_32f_C4& src, const float4& val, iu::ImageGpu_32f_C4& dst);

IUMATH_DLLAPI void addC(iu::ImageGpu_64f_C1& src, const double& val, iu::ImageGpu_64f_C1& dst);
IUMATH_DLLAPI void addC(iu::ImageGpu_64f_C2& src, const double2& val, iu::ImageGpu_64f_C2& dst);
IUMATH_DLLAPI void addC(iu::ImageGpu_64f_C3& src, const double3& val, iu::ImageGpu_64f_C3& dst);
IUMATH_DLLAPI void addC(iu::ImageGpu_64f_C4& src, const double4& val, iu::ImageGpu_64f_C4& dst);

IUMATH_DLLAPI void addC(iu::ImageGpu_8u_C1& src, const unsigned char& val, iu::ImageGpu_8u_C1& dst);
IUMATH_DLLAPI void addC(iu::ImageGpu_8u_C2& src, const uchar2& val, iu::ImageGpu_8u_C2& dst);
IUMATH_DLLAPI void addC(iu::ImageGpu_8u_C3& src, const uchar3& val, iu::ImageGpu_8u_C3& dst);
IUMATH_DLLAPI void addC(iu::ImageGpu_8u_C4& src, const uchar4& val, iu::ImageGpu_8u_C4& dst);

IUMATH_DLLAPI void addC(iu::ImageGpu_32s_C1& src, const int& val, iu::ImageGpu_32s_C1& dst);
IUMATH_DLLAPI void addC(iu::ImageGpu_32u_C1& src, const unsigned int& val, iu::ImageGpu_32u_C1& dst);
IUMATH_DLLAPI void addC(iu::ImageGpu_16u_C1& src, const unsigned short& val, iu::ImageGpu_16u_C1& dst);

IUMATH_DLLAPI void addC(iu::VolumeGpu_32f_C1& src, const float& val, iu::VolumeGpu_32f_C1& dst);
IUMATH_DLLAPI void addC(iu::VolumeGpu_32f_C2& src, const float2& val, iu::VolumeGpu_32f_C2& dst);
IUMATH_DLLAPI void addC(iu::VolumeGpu_32f_C3& src, const float3& val, iu::VolumeGpu_32f_C3& dst);
IUMATH_DLLAPI void addC(iu::VolumeGpu_32f_C4& src, const float4& val, iu::VolumeGpu_32f_C4& dst);

IUMATH_DLLAPI void addC(iu::VolumeGpu_64f_C1& src, const double& val, iu::VolumeGpu_64f_C1& dst);
IUMATH_DLLAPI void addC(iu::VolumeGpu_64f_C2& src, const double2& val, iu::VolumeGpu_64f_C2& dst);
IUMATH_DLLAPI void addC(iu::VolumeGpu_64f_C3& src, const double3& val, iu::VolumeGpu_64f_C3& dst);
IUMATH_DLLAPI void addC(iu::VolumeGpu_64f_C4& src, const double4& val, iu::VolumeGpu_64f_C4& dst);

IUMATH_DLLAPI void addC(iu::LinearDeviceMemory_32f_C1& src, const float& val, iu::LinearDeviceMemory_32f_C1& dst);
IUMATH_DLLAPI void addC(iu::LinearDeviceMemory_32f_C2& src, const float2& val, iu::LinearDeviceMemory_32f_C2& dst);
IUMATH_DLLAPI void addC(iu::LinearDeviceMemory_32f_C3& src, const float3& val, iu::LinearDeviceMemory_32f_C3& dst);
IUMATH_DLLAPI void addC(iu::LinearDeviceMemory_32f_C4& src, const float4& val, iu::LinearDeviceMemory_32f_C4& dst);

IUMATH_DLLAPI void addC(iu::LinearDeviceMemory_64f_C1& src, const double& val, iu::LinearDeviceMemory_64f_C1& dst);
IUMATH_DLLAPI void addC(iu::LinearDeviceMemory_64f_C2& src, const double2& val, iu::LinearDeviceMemory_64f_C2& dst);
IUMATH_DLLAPI void addC(iu::LinearDeviceMemory_64f_C3& src, const double3& val, iu::LinearDeviceMemory_64f_C3& dst);
IUMATH_DLLAPI void addC(iu::LinearDeviceMemory_64f_C4& src, const double4& val, iu::LinearDeviceMemory_64f_C4& dst);

IUMATH_DLLAPI void addC(iu::LinearDeviceMemory_8u_C1& src, const unsigned char& val, iu::LinearDeviceMemory_8u_C1& dst);
IUMATH_DLLAPI void addC(iu::LinearDeviceMemory_8u_C2& src, const uchar2& val, iu::LinearDeviceMemory_8u_C2& dst);
IUMATH_DLLAPI void addC(iu::LinearDeviceMemory_8u_C3& src, const uchar3& val, iu::LinearDeviceMemory_8u_C3& dst);
IUMATH_DLLAPI void addC(iu::LinearDeviceMemory_8u_C4& src, const uchar4& val, iu::LinearDeviceMemory_8u_C4& dst);

IUMATH_DLLAPI void addC(iu::LinearDeviceMemory_32s_C1& src, const int& val, iu::LinearDeviceMemory_32s_C1& dst);
IUMATH_DLLAPI void addC(iu::LinearDeviceMemory_32u_C1& src, const unsigned int& val, iu::LinearDeviceMemory_32u_C1& dst);
IUMATH_DLLAPI void addC(iu::LinearDeviceMemory_16u_C1& src, const unsigned short& val, iu::LinearDeviceMemory_16u_C1& dst);

IUMATH_DLLAPI void addC(iu::LinearDeviceMemory<float, 2>& src, const float& val, iu::LinearDeviceMemory<float, 2>& dst);
IUMATH_DLLAPI void addC(iu::LinearDeviceMemory<float2, 2>& src, const float2& val, iu::LinearDeviceMemory<float2, 2>& dst);
IUMATH_DLLAPI void addC(iu::LinearDeviceMemory<float3, 2>& src, const float3& val, iu::LinearDeviceMemory<float3, 2>& dst);
IUMATH_DLLAPI void addC(iu::LinearDeviceMemory<float4, 2>& src, const float4& val, iu::LinearDeviceMemory<float4, 2>& dst);

IUMATH_DLLAPI void addC(iu::LinearDeviceMemory<float, 3>& src, const float& val, iu::LinearDeviceMemory<float, 3>& dst);
IUMATH_DLLAPI void addC(iu::LinearDeviceMemory<float2, 3>& src, const float2& val, iu::LinearDeviceMemory<float2, 3>& dst);
IUMATH_DLLAPI void addC(iu::LinearDeviceMemory<float3, 3>& src, const float3& val, iu::LinearDeviceMemory<float3, 3>& dst);
IUMATH_DLLAPI void addC(iu::LinearDeviceMemory<float4, 3>& src, const float4& val, iu::LinearDeviceMemory<float4, 3>& dst);

IUMATH_DLLAPI void addC(iu::LinearDeviceMemory<float, 4>& src, const float& val, iu::LinearDeviceMemory<float, 4>& dst);
IUMATH_DLLAPI void addC(iu::LinearDeviceMemory<float2, 4>& src, const float2& val, iu::LinearDeviceMemory<float2, 4>& dst);
IUMATH_DLLAPI void addC(iu::LinearDeviceMemory<float3, 4>& src, const float3& val, iu::LinearDeviceMemory<float3, 4>& dst);
IUMATH_DLLAPI void addC(iu::LinearDeviceMemory<float4, 4>& src, const float4& val, iu::LinearDeviceMemory<float4, 4>& dst);

IUMATH_DLLAPI void addC(iu::LinearDeviceMemory<float, 5>& src, const float& val, iu::LinearDeviceMemory<float, 5>& dst);
IUMATH_DLLAPI void addC(iu::LinearDeviceMemory<float2, 5>& src, const float2& val, iu::LinearDeviceMemory<float2, 5>& dst);
IUMATH_DLLAPI void addC(iu::LinearDeviceMemory<float3, 5>& src, const float3& val, iu::LinearDeviceMemory<float3, 5>& dst);
IUMATH_DLLAPI void addC(iu::LinearDeviceMemory<float4, 5>& src, const float4& val, iu::LinearDeviceMemory<float4, 5>& dst);

IUMATH_DLLAPI void addC(iu::LinearDeviceMemory<double, 2>& src, const double& val, iu::LinearDeviceMemory<double, 2>& dst);
IUMATH_DLLAPI void addC(iu::LinearDeviceMemory<double2, 2>& src, const double2& val, iu::LinearDeviceMemory<double2, 2>& dst);
IUMATH_DLLAPI void addC(iu::LinearDeviceMemory<double3, 2>& src, const double3& val, iu::LinearDeviceMemory<double3, 2>& dst);
IUMATH_DLLAPI void addC(iu::LinearDeviceMemory<double4, 2>& src, const double4& val, iu::LinearDeviceMemory<double4, 2>& dst);

IUMATH_DLLAPI void addC(iu::LinearDeviceMemory<double, 3>& src, const double& val, iu::LinearDeviceMemory<double, 3>& dst);
IUMATH_DLLAPI void addC(iu::LinearDeviceMemory<double2, 3>& src, const double2& val, iu::LinearDeviceMemory<double2, 3>& dst);
IUMATH_DLLAPI void addC(iu::LinearDeviceMemory<double3, 3>& src, const double3& val, iu::LinearDeviceMemory<double3, 3>& dst);
IUMATH_DLLAPI void addC(iu::LinearDeviceMemory<double4, 3>& src, const double4& val, iu::LinearDeviceMemory<double4, 3>& dst);

IUMATH_DLLAPI void addC(iu::LinearDeviceMemory<double, 4>& src, const double& val, iu::LinearDeviceMemory<double, 4>& dst);
IUMATH_DLLAPI void addC(iu::LinearDeviceMemory<double2, 4>& src, const double2& val, iu::LinearDeviceMemory<double2, 4>& dst);
IUMATH_DLLAPI void addC(iu::LinearDeviceMemory<double3, 4>& src, const double3& val, iu::LinearDeviceMemory<double3, 4>& dst);
IUMATH_DLLAPI void addC(iu::LinearDeviceMemory<double4, 4>& src, const double4& val, iu::LinearDeviceMemory<double4, 4>& dst);

IUMATH_DLLAPI void addC(iu::LinearDeviceMemory<double, 5>& src, const double& val, iu::LinearDeviceMemory<double, 5>& dst);
IUMATH_DLLAPI void addC(iu::LinearDeviceMemory<double2, 5>& src, const double2& val, iu::LinearDeviceMemory<double2, 5>& dst);
IUMATH_DLLAPI void addC(iu::LinearDeviceMemory<double3, 5>& src, const double3& val, iu::LinearDeviceMemory<double3, 5>& dst);
IUMATH_DLLAPI void addC(iu::LinearDeviceMemory<double4, 5>& src, const double4& val, iu::LinearDeviceMemory<double4, 5>& dst);

IUMATH_DLLAPI void addC(iu::ImageCpu_32f_C1& src, const float& val, iu::ImageCpu_32f_C1& dst);
IUMATH_DLLAPI void addC(iu::ImageCpu_32f_C2& src, const float2& val, iu::ImageCpu_32f_C2& dst);
IUMATH_DLLAPI void addC(iu::ImageCpu_32f_C3& src, const float3& val, iu::ImageCpu_32f_C3& dst);
IUMATH_DLLAPI void addC(iu::ImageCpu_32f_C4& src, const float4& val, iu::ImageCpu_32f_C4& dst);

IUMATH_DLLAPI void addC(iu::ImageCpu_64f_C1& src, const double& val, iu::ImageCpu_64f_C1& dst);
IUMATH_DLLAPI void addC(iu::ImageCpu_64f_C2& src, const double2& val, iu::ImageCpu_64f_C2& dst);
IUMATH_DLLAPI void addC(iu::ImageCpu_64f_C3& src, const double3& val, iu::ImageCpu_64f_C3& dst);
IUMATH_DLLAPI void addC(iu::ImageCpu_64f_C4& src, const double4& val, iu::ImageCpu_64f_C4& dst);

IUMATH_DLLAPI void addC(iu::ImageCpu_8u_C1& src, const unsigned char& val, iu::ImageCpu_8u_C1& dst);
IUMATH_DLLAPI void addC(iu::ImageCpu_8u_C2& src, const uchar2& val, iu::ImageCpu_8u_C2& dst);
IUMATH_DLLAPI void addC(iu::ImageCpu_8u_C3& src, const uchar3& val, iu::ImageCpu_8u_C3& dst);
IUMATH_DLLAPI void addC(iu::ImageCpu_8u_C4& src, const uchar4& val, iu::ImageCpu_8u_C4& dst);

IUMATH_DLLAPI void addC(iu::ImageCpu_32s_C1& src, const int& val, iu::ImageCpu_32s_C1& dst);
IUMATH_DLLAPI void addC(iu::ImageCpu_32u_C1& src, const unsigned int& val, iu::ImageCpu_32u_C1& dst);
IUMATH_DLLAPI void addC(iu::ImageCpu_16u_C1& src, const unsigned short& val, iu::ImageCpu_16u_C1& dst);

IUMATH_DLLAPI void addC(iu::VolumeCpu_32f_C1& src, const float& val, iu::VolumeCpu_32f_C1& dst);
IUMATH_DLLAPI void addC(iu::VolumeCpu_32f_C2& src, const float2& val, iu::VolumeCpu_32f_C2& dst);
IUMATH_DLLAPI void addC(iu::VolumeCpu_32f_C3& src, const float3& val, iu::VolumeCpu_32f_C3& dst);
IUMATH_DLLAPI void addC(iu::VolumeCpu_32f_C4& src, const float4& val, iu::VolumeCpu_32f_C4& dst);

IUMATH_DLLAPI void addC(iu::VolumeCpu_64f_C1& src, const double& val, iu::VolumeCpu_64f_C1& dst);
IUMATH_DLLAPI void addC(iu::VolumeCpu_64f_C2& src, const double2& val, iu::VolumeCpu_64f_C2& dst);
IUMATH_DLLAPI void addC(iu::VolumeCpu_64f_C3& src, const double3& val, iu::VolumeCpu_64f_C3& dst);
IUMATH_DLLAPI void addC(iu::VolumeCpu_64f_C4& src, const double4& val, iu::VolumeCpu_64f_C4& dst);

IUMATH_DLLAPI void addC(iu::LinearHostMemory_32f_C1& src, const float& val, iu::LinearHostMemory_32f_C1& dst);
IUMATH_DLLAPI void addC(iu::LinearHostMemory_32f_C2& src, const float2& val, iu::LinearHostMemory_32f_C2& dst);
IUMATH_DLLAPI void addC(iu::LinearHostMemory_32f_C3& src, const float3& val, iu::LinearHostMemory_32f_C3& dst);
IUMATH_DLLAPI void addC(iu::LinearHostMemory_32f_C4& src, const float4& val, iu::LinearHostMemory_32f_C4& dst);

IUMATH_DLLAPI void addC(iu::LinearHostMemory_64f_C1& src, const double& val, iu::LinearHostMemory_64f_C1& dst);
IUMATH_DLLAPI void addC(iu::LinearHostMemory_64f_C2& src, const double2& val, iu::LinearHostMemory_64f_C2& dst);
IUMATH_DLLAPI void addC(iu::LinearHostMemory_64f_C3& src, const double3& val, iu::LinearHostMemory_64f_C3& dst);
IUMATH_DLLAPI void addC(iu::LinearHostMemory_64f_C4& src, const double4& val, iu::LinearHostMemory_64f_C4& dst);

IUMATH_DLLAPI void addC(iu::LinearHostMemory_8u_C1& src, const unsigned char& val, iu::LinearHostMemory_8u_C1& dst);
IUMATH_DLLAPI void addC(iu::LinearHostMemory_8u_C2& src, const uchar2& val, iu::LinearHostMemory_8u_C2& dst);
IUMATH_DLLAPI void addC(iu::LinearHostMemory_8u_C3& src, const uchar3& val, iu::LinearHostMemory_8u_C3& dst);
IUMATH_DLLAPI void addC(iu::LinearHostMemory_8u_C4& src, const uchar4& val, iu::LinearHostMemory_8u_C4& dst);

IUMATH_DLLAPI void addC(iu::LinearHostMemory_32s_C1& src, const int& val, iu::LinearHostMemory_32s_C1& dst);
IUMATH_DLLAPI void addC(iu::LinearHostMemory_32u_C1& src, const unsigned int& val, iu::LinearHostMemory_32u_C1& dst);
IUMATH_DLLAPI void addC(iu::LinearHostMemory_16u_C1& src, const unsigned short& val, iu::LinearHostMemory_16u_C1& dst);

IUMATH_DLLAPI void addC(iu::LinearHostMemory<float, 2>& src, const float& val, iu::LinearHostMemory<float, 2>& dst);
IUMATH_DLLAPI void addC(iu::LinearHostMemory<float2, 2>& src, const float2& val, iu::LinearHostMemory<float2, 2>& dst);
IUMATH_DLLAPI void addC(iu::LinearHostMemory<float3, 2>& src, const float3& val, iu::LinearHostMemory<float3, 2>& dst);
IUMATH_DLLAPI void addC(iu::LinearHostMemory<float4, 2>& src, const float4& val, iu::LinearHostMemory<float4, 2>& dst);

IUMATH_DLLAPI void addC(iu::LinearHostMemory<float, 3>& src, const float& val, iu::LinearHostMemory<float, 3>& dst);
IUMATH_DLLAPI void addC(iu::LinearHostMemory<float2, 3>& src, const float2& val, iu::LinearHostMemory<float2, 3>& dst);
IUMATH_DLLAPI void addC(iu::LinearHostMemory<float3, 3>& src, const float3& val, iu::LinearHostMemory<float3, 3>& dst);
IUMATH_DLLAPI void addC(iu::LinearHostMemory<float4, 3>& src, const float4& val, iu::LinearHostMemory<float4, 3>& dst);

IUMATH_DLLAPI void addC(iu::LinearHostMemory<float, 4>& src, const float& val, iu::LinearHostMemory<float, 4>& dst);
IUMATH_DLLAPI void addC(iu::LinearHostMemory<float2, 4>& src, const float2& val, iu::LinearHostMemory<float2, 4>& dst);
IUMATH_DLLAPI void addC(iu::LinearHostMemory<float3, 4>& src, const float3& val, iu::LinearHostMemory<float3, 4>& dst);
IUMATH_DLLAPI void addC(iu::LinearHostMemory<float4, 4>& src, const float4& val, iu::LinearHostMemory<float4, 4>& dst);

IUMATH_DLLAPI void addC(iu::LinearHostMemory<float, 5>& src, const float& val, iu::LinearHostMemory<float, 5>& dst);
IUMATH_DLLAPI void addC(iu::LinearHostMemory<float2, 5>& src, const float2& val, iu::LinearHostMemory<float2, 5>& dst);
IUMATH_DLLAPI void addC(iu::LinearHostMemory<float3, 5>& src, const float3& val, iu::LinearHostMemory<float3, 5>& dst);
IUMATH_DLLAPI void addC(iu::LinearHostMemory<float4, 5>& src, const float4& val, iu::LinearHostMemory<float4, 5>& dst);

IUMATH_DLLAPI void addC(iu::LinearHostMemory<double, 2>& src, const double& val, iu::LinearHostMemory<double, 2>& dst);
IUMATH_DLLAPI void addC(iu::LinearHostMemory<double2, 2>& src, const double2& val, iu::LinearHostMemory<double2, 2>& dst);
IUMATH_DLLAPI void addC(iu::LinearHostMemory<double3, 2>& src, const double3& val, iu::LinearHostMemory<double3, 2>& dst);
IUMATH_DLLAPI void addC(iu::LinearHostMemory<double4, 2>& src, const double4& val, iu::LinearHostMemory<double4, 2>& dst);

IUMATH_DLLAPI void addC(iu::LinearHostMemory<double, 3>& src, const double& val, iu::LinearHostMemory<double, 3>& dst);
IUMATH_DLLAPI void addC(iu::LinearHostMemory<double2, 3>& src, const double2& val, iu::LinearHostMemory<double2, 3>& dst);
IUMATH_DLLAPI void addC(iu::LinearHostMemory<double3, 3>& src, const double3& val, iu::LinearHostMemory<double3, 3>& dst);
IUMATH_DLLAPI void addC(iu::LinearHostMemory<double4, 3>& src, const double4& val, iu::LinearHostMemory<double4, 3>& dst);

IUMATH_DLLAPI void addC(iu::LinearHostMemory<double, 4>& src, const double& val, iu::LinearHostMemory<double, 4>& dst);
IUMATH_DLLAPI void addC(iu::LinearHostMemory<double2, 4>& src, const double2& val, iu::LinearHostMemory<double2, 4>& dst);
IUMATH_DLLAPI void addC(iu::LinearHostMemory<double3, 4>& src, const double3& val, iu::LinearHostMemory<double3, 4>& dst);
IUMATH_DLLAPI void addC(iu::LinearHostMemory<double4, 4>& src, const double4& val, iu::LinearHostMemory<double4, 4>& dst);

IUMATH_DLLAPI void addC(iu::LinearHostMemory<double, 5>& src, const double& val, iu::LinearHostMemory<double, 5>& dst);
IUMATH_DLLAPI void addC(iu::LinearHostMemory<double2, 5>& src, const double2& val, iu::LinearHostMemory<double2, 5>& dst);
IUMATH_DLLAPI void addC(iu::LinearHostMemory<double3, 5>& src, const double3& val, iu::LinearHostMemory<double3, 5>& dst);
IUMATH_DLLAPI void addC(iu::LinearHostMemory<double4, 5>& src, const double4& val, iu::LinearHostMemory<double4, 5>& dst);

// multiply constant
/** Multiply a constant to an array (can be called in-place)
 * \param src Source array
 * \param val Value to add
 * \param[out] dst Destination array
 */
IUMATH_DLLAPI void mulC(iu::ImageGpu_32f_C1& src, const float& val, iu::ImageGpu_32f_C1& dst);
IUMATH_DLLAPI void mulC(iu::ImageGpu_32f_C2& src, const float2& val, iu::ImageGpu_32f_C2& dst);
IUMATH_DLLAPI void mulC(iu::ImageGpu_32f_C3& src, const float3& val, iu::ImageGpu_32f_C3& dst);
IUMATH_DLLAPI void mulC(iu::ImageGpu_32f_C4& src, const float4& val, iu::ImageGpu_32f_C4& dst);

IUMATH_DLLAPI void mulC(iu::ImageGpu_64f_C1& src, const double& val, iu::ImageGpu_64f_C1& dst);
IUMATH_DLLAPI void mulC(iu::ImageGpu_64f_C2& src, const double2& val, iu::ImageGpu_64f_C2& dst);
IUMATH_DLLAPI void mulC(iu::ImageGpu_64f_C3& src, const double3& val, iu::ImageGpu_64f_C3& dst);
IUMATH_DLLAPI void mulC(iu::ImageGpu_64f_C4& src, const double4& val, iu::ImageGpu_64f_C4& dst);

IUMATH_DLLAPI void mulC(iu::ImageGpu_8u_C1& src, const unsigned char& val, iu::ImageGpu_8u_C1& dst);
IUMATH_DLLAPI void mulC(iu::ImageGpu_8u_C2& src, const uchar2& val, iu::ImageGpu_8u_C2& dst);
IUMATH_DLLAPI void mulC(iu::ImageGpu_8u_C3& src, const uchar3& val, iu::ImageGpu_8u_C3& dst);
IUMATH_DLLAPI void mulC(iu::ImageGpu_8u_C4& src, const uchar4& val, iu::ImageGpu_8u_C4& dst);

IUMATH_DLLAPI void mulC(iu::ImageGpu_8u_C1& src, const unsigned char& val, iu::ImageGpu_8u_C1& dst);
IUMATH_DLLAPI void mulC(iu::ImageGpu_32s_C1& src, const int& val, iu::ImageGpu_32s_C1& dst);
IUMATH_DLLAPI void mulC(iu::ImageGpu_32u_C1& src, const unsigned int& val, iu::ImageGpu_32u_C1& dst);
IUMATH_DLLAPI void mulC(iu::ImageGpu_16u_C1& src, const unsigned short& val, iu::ImageGpu_16u_C1& dst);

IUMATH_DLLAPI void mulC(iu::VolumeGpu_32f_C1& src, const float& val, iu::VolumeGpu_32f_C1& dst);
IUMATH_DLLAPI void mulC(iu::VolumeGpu_32f_C2& src, const float2& val, iu::VolumeGpu_32f_C2& dst);
IUMATH_DLLAPI void mulC(iu::VolumeGpu_32f_C3& src, const float3& val, iu::VolumeGpu_32f_C3& dst);
IUMATH_DLLAPI void mulC(iu::VolumeGpu_32f_C4& src, const float4& val, iu::VolumeGpu_32f_C4& dst);

IUMATH_DLLAPI void mulC(iu::VolumeGpu_64f_C1& src, const double& val, iu::VolumeGpu_64f_C1& dst);
IUMATH_DLLAPI void mulC(iu::VolumeGpu_64f_C2& src, const double2& val, iu::VolumeGpu_64f_C2& dst);
IUMATH_DLLAPI void mulC(iu::VolumeGpu_64f_C3& src, const double3& val, iu::VolumeGpu_64f_C3& dst);
IUMATH_DLLAPI void mulC(iu::VolumeGpu_64f_C4& src, const double4& val, iu::VolumeGpu_64f_C4& dst);

IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory_32f_C1& src, const float& val, iu::LinearDeviceMemory_32f_C1& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory_32f_C2& src, const float2& val, iu::LinearDeviceMemory_32f_C2& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory_32f_C3& src, const float3& val, iu::LinearDeviceMemory_32f_C3& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory_32f_C4& src, const float4& val, iu::LinearDeviceMemory_32f_C4& dst);

IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory_64f_C1& src, const double& val, iu::LinearDeviceMemory_64f_C1& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory_64f_C2& src, const double2& val, iu::LinearDeviceMemory_64f_C2& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory_64f_C3& src, const double3& val, iu::LinearDeviceMemory_64f_C3& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory_64f_C4& src, const double4& val, iu::LinearDeviceMemory_64f_C4& dst);

IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory_8u_C1& src, const unsigned char& val, iu::LinearDeviceMemory_8u_C1& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory_8u_C2& src, const uchar2& val, iu::LinearDeviceMemory_8u_C2& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory_8u_C3& src, const uchar3& val, iu::LinearDeviceMemory_8u_C3& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory_8u_C4& src, const uchar4& val, iu::LinearDeviceMemory_8u_C4& dst);

IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory_8u_C1& src, const unsigned char& val, iu::LinearDeviceMemory_8u_C1& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory_32s_C1& src, const int& val, iu::LinearDeviceMemory_32s_C1& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory_32u_C1& src, const unsigned int& val, iu::LinearDeviceMemory_32u_C1& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory_16u_C1& src, const unsigned short& val, iu::LinearDeviceMemory_16u_C1& dst);

IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory<float, 2>& src, const float& val, iu::LinearDeviceMemory<float, 2>& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory<float2, 2>& src, const float2& val, iu::LinearDeviceMemory<float2, 2>& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory<float3, 2>& src, const float3& val, iu::LinearDeviceMemory<float3, 2>& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory<float4, 2>& src, const float4& val, iu::LinearDeviceMemory<float4, 2>& dst);

IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory<float, 3>& src, const float& val, iu::LinearDeviceMemory<float, 3>& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory<float2, 3>& src, const float2& val, iu::LinearDeviceMemory<float2, 3>& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory<float3, 3>& src, const float3& val, iu::LinearDeviceMemory<float3, 3>& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory<float4, 3>& src, const float4& val, iu::LinearDeviceMemory<float4, 3>& dst);

IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory<float, 4>& src, const float& val, iu::LinearDeviceMemory<float, 4>& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory<float2, 4>& src, const float2& val, iu::LinearDeviceMemory<float2, 4>& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory<float3, 4>& src, const float3& val, iu::LinearDeviceMemory<float3, 4>& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory<float4, 4>& src, const float4& val, iu::LinearDeviceMemory<float4, 4>& dst);

IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory<float, 5>& src, const float& val, iu::LinearDeviceMemory<float, 5>& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory<float2, 5>& src, const float2& val, iu::LinearDeviceMemory<float2, 5>& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory<float3, 5>& src, const float3& val, iu::LinearDeviceMemory<float3, 5>& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory<float4, 5>& src, const float4& val, iu::LinearDeviceMemory<float4, 5>& dst);

IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory<double, 2>& src, const double& val, iu::LinearDeviceMemory<double, 2>& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory<double2, 2>& src, const double2& val, iu::LinearDeviceMemory<double2, 2>& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory<double3, 2>& src, const double3& val, iu::LinearDeviceMemory<double3, 2>& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory<double4, 2>& src, const double4& val, iu::LinearDeviceMemory<double4, 2>& dst);

IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory<double, 3>& src, const double& val, iu::LinearDeviceMemory<double, 3>& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory<double2, 3>& src, const double2& val, iu::LinearDeviceMemory<double2, 3>& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory<double3, 3>& src, const double3& val, iu::LinearDeviceMemory<double3, 3>& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory<double4, 3>& src, const double4& val, iu::LinearDeviceMemory<double4, 3>& dst);

IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory<double, 4>& src, const double& val, iu::LinearDeviceMemory<double, 4>& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory<double2, 4>& src, const double2& val, iu::LinearDeviceMemory<double2, 4>& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory<double3, 4>& src, const double3& val, iu::LinearDeviceMemory<double3, 4>& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory<double4, 4>& src, const double4& val, iu::LinearDeviceMemory<double4, 4>& dst);

IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory<double, 5>& src, const double& val, iu::LinearDeviceMemory<double, 5>& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory<double2, 5>& src, const double2& val, iu::LinearDeviceMemory<double2, 5>& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory<double3, 5>& src, const double3& val, iu::LinearDeviceMemory<double3, 5>& dst);
IUMATH_DLLAPI void mulC(iu::LinearDeviceMemory<double4, 5>& src, const double4& val, iu::LinearDeviceMemory<double4, 5>& dst);

IUMATH_DLLAPI void mulC(iu::ImageCpu_32f_C1& src, const float& val, iu::ImageCpu_32f_C1& dst);
IUMATH_DLLAPI void mulC(iu::ImageCpu_32f_C2& src, const float2& val, iu::ImageCpu_32f_C2& dst);
IUMATH_DLLAPI void mulC(iu::ImageCpu_32f_C3& src, const float3& val, iu::ImageCpu_32f_C3& dst);
IUMATH_DLLAPI void mulC(iu::ImageCpu_32f_C4& src, const float4& val, iu::ImageCpu_32f_C4& dst);

IUMATH_DLLAPI void mulC(iu::ImageCpu_64f_C1& src, const double& val, iu::ImageCpu_64f_C1& dst);
IUMATH_DLLAPI void mulC(iu::ImageCpu_64f_C2& src, const double2& val, iu::ImageCpu_64f_C2& dst);
IUMATH_DLLAPI void mulC(iu::ImageCpu_64f_C3& src, const double3& val, iu::ImageCpu_64f_C3& dst);
IUMATH_DLLAPI void mulC(iu::ImageCpu_64f_C4& src, const double4& val, iu::ImageCpu_64f_C4& dst);

IUMATH_DLLAPI void mulC(iu::ImageCpu_8u_C1& src, const unsigned char& val, iu::ImageCpu_8u_C1& dst);
IUMATH_DLLAPI void mulC(iu::ImageCpu_8u_C2& src, const uchar2& val, iu::ImageCpu_8u_C2& dst);
IUMATH_DLLAPI void mulC(iu::ImageCpu_8u_C3& src, const uchar3& val, iu::ImageCpu_8u_C3& dst);
IUMATH_DLLAPI void mulC(iu::ImageCpu_8u_C4& src, const uchar4& val, iu::ImageCpu_8u_C4& dst);

IUMATH_DLLAPI void mulC(iu::ImageCpu_8u_C1& src, const unsigned char& val, iu::ImageCpu_8u_C1& dst);
IUMATH_DLLAPI void mulC(iu::ImageCpu_32s_C1& src, const int& val, iu::ImageCpu_32s_C1& dst);
IUMATH_DLLAPI void mulC(iu::ImageCpu_32u_C1& src, const unsigned int& val, iu::ImageCpu_32u_C1& dst);
IUMATH_DLLAPI void mulC(iu::ImageCpu_16u_C1& src, const unsigned short& val, iu::ImageCpu_16u_C1& dst);

IUMATH_DLLAPI void mulC(iu::VolumeCpu_32f_C1& src, const float& val, iu::VolumeCpu_32f_C1& dst);
IUMATH_DLLAPI void mulC(iu::VolumeCpu_32f_C2& src, const float2& val, iu::VolumeCpu_32f_C2& dst);
IUMATH_DLLAPI void mulC(iu::VolumeCpu_32f_C3& src, const float3& val, iu::VolumeCpu_32f_C3& dst);
IUMATH_DLLAPI void mulC(iu::VolumeCpu_32f_C4& src, const float4& val, iu::VolumeCpu_32f_C4& dst);

IUMATH_DLLAPI void mulC(iu::VolumeCpu_64f_C1& src, const double& val, iu::VolumeCpu_64f_C1& dst);
IUMATH_DLLAPI void mulC(iu::VolumeCpu_64f_C2& src, const double2& val, iu::VolumeCpu_64f_C2& dst);
IUMATH_DLLAPI void mulC(iu::VolumeCpu_64f_C3& src, const double3& val, iu::VolumeCpu_64f_C3& dst);
IUMATH_DLLAPI void mulC(iu::VolumeCpu_64f_C4& src, const double4& val, iu::VolumeCpu_64f_C4& dst);

IUMATH_DLLAPI void mulC(iu::LinearHostMemory_32f_C1& src, const float& val, iu::LinearHostMemory_32f_C1& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory_32f_C2& src, const float2& val, iu::LinearHostMemory_32f_C2& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory_32f_C3& src, const float3& val, iu::LinearHostMemory_32f_C3& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory_32f_C4& src, const float4& val, iu::LinearHostMemory_32f_C4& dst);

IUMATH_DLLAPI void mulC(iu::LinearHostMemory_64f_C1& src, const double& val, iu::LinearHostMemory_64f_C1& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory_64f_C2& src, const double2& val, iu::LinearHostMemory_64f_C2& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory_64f_C3& src, const double3& val, iu::LinearHostMemory_64f_C3& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory_64f_C4& src, const double4& val, iu::LinearHostMemory_64f_C4& dst);

IUMATH_DLLAPI void mulC(iu::LinearHostMemory_8u_C1& src, const unsigned char& val, iu::LinearHostMemory_8u_C1& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory_8u_C2& src, const uchar2& val, iu::LinearHostMemory_8u_C2& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory_8u_C3& src, const uchar3& val, iu::LinearHostMemory_8u_C3& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory_8u_C4& src, const uchar4& val, iu::LinearHostMemory_8u_C4& dst);

IUMATH_DLLAPI void mulC(iu::LinearHostMemory_8u_C1& src, const unsigned char& val, iu::LinearHostMemory_8u_C1& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory_32s_C1& src, const int& val, iu::LinearHostMemory_32s_C1& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory_32u_C1& src, const unsigned int& val, iu::LinearHostMemory_32u_C1& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory_16u_C1& src, const unsigned short& val, iu::LinearHostMemory_16u_C1& dst);

IUMATH_DLLAPI void mulC(iu::LinearHostMemory<float, 2>& src, const float& val, iu::LinearHostMemory<float, 2>& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory<float2, 2>& src, const float2& val, iu::LinearHostMemory<float2, 2>& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory<float3, 2>& src, const float3& val, iu::LinearHostMemory<float3, 2>& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory<float4, 2>& src, const float4& val, iu::LinearHostMemory<float4, 2>& dst);

IUMATH_DLLAPI void mulC(iu::LinearHostMemory<float, 3>& src, const float& val, iu::LinearHostMemory<float, 3>& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory<float2, 3>& src, const float2& val, iu::LinearHostMemory<float2, 3>& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory<float3, 3>& src, const float3& val, iu::LinearHostMemory<float3, 3>& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory<float4, 3>& src, const float4& val, iu::LinearHostMemory<float4, 3>& dst);

IUMATH_DLLAPI void mulC(iu::LinearHostMemory<float, 4>& src, const float& val, iu::LinearHostMemory<float, 4>& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory<float2, 4>& src, const float2& val, iu::LinearHostMemory<float2, 4>& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory<float3, 4>& src, const float3& val, iu::LinearHostMemory<float3, 4>& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory<float4, 4>& src, const float4& val, iu::LinearHostMemory<float4, 4>& dst);

IUMATH_DLLAPI void mulC(iu::LinearHostMemory<float, 5>& src, const float& val, iu::LinearHostMemory<float, 5>& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory<float2, 5>& src, const float2& val, iu::LinearHostMemory<float2, 5>& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory<float3, 5>& src, const float3& val, iu::LinearHostMemory<float3, 5>& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory<float4, 5>& src, const float4& val, iu::LinearHostMemory<float4, 5>& dst);

IUMATH_DLLAPI void mulC(iu::LinearHostMemory<double, 2>& src, const double& val, iu::LinearHostMemory<double, 2>& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory<double2, 2>& src, const double2& val, iu::LinearHostMemory<double2, 2>& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory<double3, 2>& src, const double3& val, iu::LinearHostMemory<double3, 2>& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory<double4, 2>& src, const double4& val, iu::LinearHostMemory<double4, 2>& dst);

IUMATH_DLLAPI void mulC(iu::LinearHostMemory<double, 3>& src, const double& val, iu::LinearHostMemory<double, 3>& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory<double2, 3>& src, const double2& val, iu::LinearHostMemory<double2, 3>& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory<double3, 3>& src, const double3& val, iu::LinearHostMemory<double3, 3>& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory<double4, 3>& src, const double4& val, iu::LinearHostMemory<double4, 3>& dst);

IUMATH_DLLAPI void mulC(iu::LinearHostMemory<double, 4>& src, const double& val, iu::LinearHostMemory<double, 4>& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory<double2, 4>& src, const double2& val, iu::LinearHostMemory<double2, 4>& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory<double3, 4>& src, const double3& val, iu::LinearHostMemory<double3, 4>& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory<double4, 4>& src, const double4& val, iu::LinearHostMemory<double4, 4>& dst);

IUMATH_DLLAPI void mulC(iu::LinearHostMemory<double, 5>& src, const double& val, iu::LinearHostMemory<double, 5>& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory<double2, 5>& src, const double2& val, iu::LinearHostMemory<double2, 5>& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory<double3, 5>& src, const double3& val, iu::LinearHostMemory<double3, 5>& dst);
IUMATH_DLLAPI void mulC(iu::LinearHostMemory<double4, 5>& src, const double4& val, iu::LinearHostMemory<double4, 5>& dst);

// pointwise weighted add
/** Add an array to another array with weighting factors (dst = weight1*src1 + weight2*src2) (can be called in-place)
 * \param src1 First source array
 * \param weight1 First weight
 * \param src2 Second source array
 * \param weight2 Second weight
 * \param[out] dst Destination array
 */
IUMATH_DLLAPI void addWeighted(iu::ImageGpu_32f_C1& src1, const float& weight1,
                 iu::ImageGpu_32f_C1& src2, const float& weight2,iu::ImageGpu_32f_C1& dst);
IUMATH_DLLAPI void addWeighted(iu::ImageGpu_32f_C2& src1, const float2& weight1,
                 iu::ImageGpu_32f_C2& src2, const float2& weight2,iu::ImageGpu_32f_C2& dst);
IUMATH_DLLAPI void addWeighted(iu::ImageGpu_32f_C3& src1, const float3& weight1,
                 iu::ImageGpu_32f_C3& src2, const float3& weight2,iu::ImageGpu_32f_C3& dst);
IUMATH_DLLAPI void addWeighted(iu::ImageGpu_32f_C4& src1, const float4& weight1,
                 iu::ImageGpu_32f_C4& src2, const float4& weight2,iu::ImageGpu_32f_C4& dst);

IUMATH_DLLAPI void addWeighted(iu::ImageGpu_64f_C1& src1, const double& weight1,
                 iu::ImageGpu_64f_C1& src2, const double& weight2,iu::ImageGpu_64f_C1& dst);
IUMATH_DLLAPI void addWeighted(iu::ImageGpu_64f_C2& src1, const double2& weight1,
                 iu::ImageGpu_64f_C2& src2, const double2& weight2,iu::ImageGpu_64f_C2& dst);
IUMATH_DLLAPI void addWeighted(iu::ImageGpu_64f_C3& src1, const double3& weight1,
                 iu::ImageGpu_64f_C3& src2, const double3& weight2,iu::ImageGpu_64f_C3& dst);
IUMATH_DLLAPI void addWeighted(iu::ImageGpu_64f_C4& src1, const double4& weight1,
                 iu::ImageGpu_64f_C4& src2, const double4& weight2,iu::ImageGpu_64f_C4& dst);

IUMATH_DLLAPI void addWeighted(iu::ImageGpu_8u_C1& src1, const unsigned char& weight1,
                 iu::ImageGpu_8u_C1& src2, const unsigned char& weight2,iu::ImageGpu_8u_C1& dst);
IUMATH_DLLAPI void addWeighted(iu::ImageGpu_8u_C4& src1, const uchar4& weight1,
                 iu::ImageGpu_8u_C4& src2, const uchar4& weight2,iu::ImageGpu_8u_C4& dst);

IUMATH_DLLAPI void addWeighted(iu::VolumeGpu_32f_C1& src1, const float& weight1,
                 iu::VolumeGpu_32f_C1& src2, const float& weight2,iu::VolumeGpu_32f_C1& dst);
IUMATH_DLLAPI void addWeighted(iu::VolumeGpu_32f_C2& src1, const float2& weight1,
                 iu::VolumeGpu_32f_C2& src2, const float2& weight2,iu::VolumeGpu_32f_C2& dst);
IUMATH_DLLAPI void addWeighted(iu::VolumeGpu_64f_C1& src1, const double& weight1,
                 iu::VolumeGpu_64f_C1& src2, const double& weight2,iu::VolumeGpu_64f_C1& dst);
IUMATH_DLLAPI void addWeighted(iu::VolumeGpu_64f_C2& src1, const double2& weight1,
                 iu::VolumeGpu_64f_C2& src2, const double2& weight2,iu::VolumeGpu_64f_C2& dst);

IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory_32f_C1& src1, const float& weight1,
                 iu::LinearDeviceMemory_32f_C1& src2, const float& weight2,iu::LinearDeviceMemory_32f_C1& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory_32f_C2& src1, const float2& weight1,
                 iu::LinearDeviceMemory_32f_C2& src2, const float2& weight2,iu::LinearDeviceMemory_32f_C2& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory_32f_C3& src1, const float3& weight1,
                 iu::LinearDeviceMemory_32f_C3& src2, const float3& weight2,iu::LinearDeviceMemory_32f_C3& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory_32f_C4& src1, const float4& weight1,
                 iu::LinearDeviceMemory_32f_C4& src2, const float4& weight2,iu::LinearDeviceMemory_32f_C4& dst);

IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory_64f_C1& src1, const double& weight1,
                 iu::LinearDeviceMemory_64f_C1& src2, const double& weight2,iu::LinearDeviceMemory_64f_C1& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory_64f_C2& src1, const double2& weight1,
                 iu::LinearDeviceMemory_64f_C2& src2, const double2& weight2,iu::LinearDeviceMemory_64f_C2& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory_64f_C3& src1, const double3& weight1,
                 iu::LinearDeviceMemory_64f_C3& src2, const double3& weight2,iu::LinearDeviceMemory_64f_C3& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory_64f_C4& src1, const double4& weight1,
                 iu::LinearDeviceMemory_64f_C4& src2, const double4& weight2,iu::LinearDeviceMemory_64f_C4& dst);

IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory_8u_C1& src1, const unsigned char& weight1,
                 iu::LinearDeviceMemory_8u_C1& src2, const unsigned char& weight2,iu::LinearDeviceMemory_8u_C1& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory_8u_C4& src1, const uchar4& weight1,
                 iu::LinearDeviceMemory_8u_C4& src2, const uchar4& weight2,iu::LinearDeviceMemory_8u_C4& dst);

IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory<float, 2>& src1, const float& weight1,
                 iu::LinearDeviceMemory<float, 2>& src2, const float& weight2,iu::LinearDeviceMemory<float, 2>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory<float2, 2>& src1, const float2& weight1,
                 iu::LinearDeviceMemory<float2, 2>& src2, const float2& weight2,iu::LinearDeviceMemory<float2, 2>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory<float3, 2>& src1, const float3& weight1,
                 iu::LinearDeviceMemory<float3, 2>& src2, const float3& weight2,iu::LinearDeviceMemory<float3, 2>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory<float4, 2>& src1, const float4& weight1,
                 iu::LinearDeviceMemory<float4, 2>& src2, const float4& weight2,iu::LinearDeviceMemory<float4, 2>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory<float, 3>& src1, const float& weight1,
                 iu::LinearDeviceMemory<float, 3>& src2, const float& weight2,iu::LinearDeviceMemory<float, 3>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory<float2, 3>& src1, const float2& weight1,
                 iu::LinearDeviceMemory<float2, 3>& src2, const float2& weight2,iu::LinearDeviceMemory<float2, 3>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory<float3, 3>& src1, const float3& weight1,
                 iu::LinearDeviceMemory<float3, 3>& src2, const float3& weight2,iu::LinearDeviceMemory<float3, 3>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory<float4, 3>& src1, const float4& weight1,
                 iu::LinearDeviceMemory<float4, 3>& src2, const float4& weight2,iu::LinearDeviceMemory<float4, 3>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory<float, 4>& src1, const float& weight1,
                 iu::LinearDeviceMemory<float, 4>& src2, const float& weight2,iu::LinearDeviceMemory<float, 4>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory<float2, 4>& src1, const float2& weight1,
                 iu::LinearDeviceMemory<float2, 4>& src2, const float2& weight2,iu::LinearDeviceMemory<float2, 4>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory<float3, 4>& src1, const float3& weight1,
                 iu::LinearDeviceMemory<float3, 4>& src2, const float3& weight2,iu::LinearDeviceMemory<float3, 4>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory<float4, 4>& src1, const float4& weight1,
                 iu::LinearDeviceMemory<float4, 4>& src2, const float4& weight2,iu::LinearDeviceMemory<float4, 4>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory<float, 5>& src1, const float& weight1,
                 iu::LinearDeviceMemory<float, 5>& src2, const float& weight2,iu::LinearDeviceMemory<float, 5>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory<float2, 5>& src1, const float2& weight1,
                 iu::LinearDeviceMemory<float2, 5>& src2, const float2& weight2,iu::LinearDeviceMemory<float2, 5>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory<float3, 5>& src1, const float3& weight1,
                 iu::LinearDeviceMemory<float3, 5>& src2, const float3& weight2,iu::LinearDeviceMemory<float3, 5>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory<float4, 5>& src1, const float4& weight1,
                 iu::LinearDeviceMemory<float4, 5>& src2, const float4& weight2,iu::LinearDeviceMemory<float4, 5>& dst);

IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory<double, 2>& src1, const double& weight1,
                 iu::LinearDeviceMemory<double, 2>& src2, const double& weight2,iu::LinearDeviceMemory<double, 2>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory<double2, 2>& src1, const double2& weight1,
                 iu::LinearDeviceMemory<double2, 2>& src2, const double2& weight2,iu::LinearDeviceMemory<double2, 2>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory<double3, 2>& src1, const double3& weight1,
                 iu::LinearDeviceMemory<double3, 2>& src2, const double3& weight2,iu::LinearDeviceMemory<double3, 2>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory<double4, 2>& src1, const double4& weight1,
                 iu::LinearDeviceMemory<double4, 2>& src2, const double4& weight2,iu::LinearDeviceMemory<double4, 2>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory<double, 3>& src1, const double& weight1,
                 iu::LinearDeviceMemory<double, 3>& src2, const double& weight2,iu::LinearDeviceMemory<double, 3>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory<double2, 3>& src1, const double2& weight1,
                 iu::LinearDeviceMemory<double2, 3>& src2, const double2& weight2,iu::LinearDeviceMemory<double2, 3>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory<double3, 3>& src1, const double3& weight1,
                 iu::LinearDeviceMemory<double3, 3>& src2, const double3& weight2,iu::LinearDeviceMemory<double3, 3>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory<double4, 3>& src1, const double4& weight1,
                 iu::LinearDeviceMemory<double4, 3>& src2, const double4& weight2,iu::LinearDeviceMemory<double4, 3>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory<double, 4>& src1, const double& weight1,
                 iu::LinearDeviceMemory<double, 4>& src2, const double& weight2,iu::LinearDeviceMemory<double, 4>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory<double2, 4>& src1, const double2& weight1,
                 iu::LinearDeviceMemory<double2, 4>& src2, const double2& weight2,iu::LinearDeviceMemory<double2, 4>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory<double3, 4>& src1, const double3& weight1,
                 iu::LinearDeviceMemory<double3, 4>& src2, const double3& weight2,iu::LinearDeviceMemory<double3, 4>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory<double4, 4>& src1, const double4& weight1,
                 iu::LinearDeviceMemory<double4, 4>& src2, const double4& weight2,iu::LinearDeviceMemory<double4, 4>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory<double, 5>& src1, const double& weight1,
                 iu::LinearDeviceMemory<double, 5>& src2, const double& weight2,iu::LinearDeviceMemory<double, 5>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory<double2, 5>& src1, const double2& weight1,
                 iu::LinearDeviceMemory<double2, 5>& src2, const double2& weight2,iu::LinearDeviceMemory<double2, 5>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory<double3, 5>& src1, const double3& weight1,
                 iu::LinearDeviceMemory<double3, 5>& src2, const double3& weight2,iu::LinearDeviceMemory<double3, 5>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearDeviceMemory<double4, 5>& src1, const double4& weight1,
                 iu::LinearDeviceMemory<double4, 5>& src2, const double4& weight2,iu::LinearDeviceMemory<double4, 5>& dst);

IUMATH_DLLAPI void addWeighted(iu::ImageCpu_32f_C1& src1, const float& weight1,
                 iu::ImageCpu_32f_C1& src2, const float& weight2,iu::ImageCpu_32f_C1& dst);
IUMATH_DLLAPI void addWeighted(iu::ImageCpu_32f_C2& src1, const float2& weight1,
                 iu::ImageCpu_32f_C2& src2, const float2& weight2,iu::ImageCpu_32f_C2& dst);
IUMATH_DLLAPI void addWeighted(iu::ImageCpu_32f_C3& src1, const float3& weight1,
                 iu::ImageCpu_32f_C3& src2, const float3& weight2,iu::ImageCpu_32f_C3& dst);
IUMATH_DLLAPI void addWeighted(iu::ImageCpu_32f_C4& src1, const float4& weight1,
                 iu::ImageCpu_32f_C4& src2, const float4& weight2,iu::ImageCpu_32f_C4& dst);

IUMATH_DLLAPI void addWeighted(iu::ImageCpu_64f_C1& src1, const double& weight1,
                 iu::ImageCpu_64f_C1& src2, const double& weight2,iu::ImageCpu_64f_C1& dst);
IUMATH_DLLAPI void addWeighted(iu::ImageCpu_64f_C2& src1, const double2& weight1,
                 iu::ImageCpu_64f_C2& src2, const double2& weight2,iu::ImageCpu_64f_C2& dst);
IUMATH_DLLAPI void addWeighted(iu::ImageCpu_64f_C3& src1, const double3& weight1,
                 iu::ImageCpu_64f_C3& src2, const double3& weight2,iu::ImageCpu_64f_C3& dst);
IUMATH_DLLAPI void addWeighted(iu::ImageCpu_64f_C4& src1, const double4& weight1,
                 iu::ImageCpu_64f_C4& src2, const double4& weight2,iu::ImageCpu_64f_C4& dst);

IUMATH_DLLAPI void addWeighted(iu::ImageCpu_8u_C1& src1, const unsigned char& weight1,
                 iu::ImageCpu_8u_C1& src2, const unsigned char& weight2,iu::ImageCpu_8u_C1& dst);
IUMATH_DLLAPI void addWeighted(iu::ImageCpu_8u_C4& src1, const uchar4& weight1,
                 iu::ImageCpu_8u_C4& src2, const uchar4& weight2,iu::ImageCpu_8u_C4& dst);

IUMATH_DLLAPI void addWeighted(iu::VolumeCpu_32f_C1& src1, const float& weight1,
                 iu::VolumeCpu_32f_C1& src2, const float& weight2,iu::VolumeCpu_32f_C1& dst);
IUMATH_DLLAPI void addWeighted(iu::VolumeCpu_32f_C2& src1, const float2& weight1,
                 iu::VolumeCpu_32f_C2& src2, const float2& weight2,iu::VolumeCpu_32f_C2& dst);
IUMATH_DLLAPI void addWeighted(iu::VolumeCpu_64f_C1& src1, const double& weight1,
                 iu::VolumeCpu_64f_C1& src2, const double& weight2,iu::VolumeCpu_64f_C1& dst);
IUMATH_DLLAPI void addWeighted(iu::VolumeCpu_64f_C2& src1, const double2& weight1,
                 iu::VolumeCpu_64f_C2& src2, const double2& weight2,iu::VolumeCpu_64f_C2& dst);

IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory_32f_C1& src1, const float& weight1,
                 iu::LinearHostMemory_32f_C1& src2, const float& weight2,iu::LinearHostMemory_32f_C1& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory_32f_C2& src1, const float2& weight1,
                 iu::LinearHostMemory_32f_C2& src2, const float2& weight2,iu::LinearHostMemory_32f_C2& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory_32f_C3& src1, const float3& weight1,
                 iu::LinearHostMemory_32f_C3& src2, const float3& weight2,iu::LinearHostMemory_32f_C3& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory_32f_C4& src1, const float4& weight1,
                 iu::LinearHostMemory_32f_C4& src2, const float4& weight2,iu::LinearHostMemory_32f_C4& dst);

IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory_64f_C1& src1, const double& weight1,
                 iu::LinearHostMemory_64f_C1& src2, const double& weight2,iu::LinearHostMemory_64f_C1& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory_64f_C2& src1, const double2& weight1,
                 iu::LinearHostMemory_64f_C2& src2, const double2& weight2,iu::LinearHostMemory_64f_C2& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory_64f_C3& src1, const double3& weight1,
                 iu::LinearHostMemory_64f_C3& src2, const double3& weight2,iu::LinearHostMemory_64f_C3& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory_64f_C4& src1, const double4& weight1,
                 iu::LinearHostMemory_64f_C4& src2, const double4& weight2,iu::LinearHostMemory_64f_C4& dst);

IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory_8u_C1& src1, const unsigned char& weight1,
                 iu::LinearHostMemory_8u_C1& src2, const unsigned char& weight2,iu::LinearHostMemory_8u_C1& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory_8u_C4& src1, const uchar4& weight1,
                 iu::LinearHostMemory_8u_C4& src2, const uchar4& weight2,iu::LinearHostMemory_8u_C4& dst);

IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory<float, 2>& src1, const float& weight1,
                 iu::LinearHostMemory<float, 2>& src2, const float& weight2,iu::LinearHostMemory<float, 2>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory<float2, 2>& src1, const float2& weight1,
                 iu::LinearHostMemory<float2, 2>& src2, const float2& weight2,iu::LinearHostMemory<float2, 2>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory<float3, 2>& src1, const float3& weight1,
                 iu::LinearHostMemory<float3, 2>& src2, const float3& weight2,iu::LinearHostMemory<float3, 2>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory<float4, 2>& src1, const float4& weight1,
                 iu::LinearHostMemory<float4, 2>& src2, const float4& weight2,iu::LinearHostMemory<float4, 2>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory<float, 3>& src1, const float& weight1,
                 iu::LinearHostMemory<float, 3>& src2, const float& weight2,iu::LinearHostMemory<float, 3>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory<float2, 3>& src1, const float2& weight1,
                 iu::LinearHostMemory<float2, 3>& src2, const float2& weight2,iu::LinearHostMemory<float2, 3>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory<float3, 3>& src1, const float3& weight1,
                 iu::LinearHostMemory<float3, 3>& src2, const float3& weight2,iu::LinearHostMemory<float3, 3>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory<float4, 3>& src1, const float4& weight1,
                 iu::LinearHostMemory<float4, 3>& src2, const float4& weight2,iu::LinearHostMemory<float4, 3>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory<float, 4>& src1, const float& weight1,
                 iu::LinearHostMemory<float, 4>& src2, const float& weight2,iu::LinearHostMemory<float, 4>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory<float2, 4>& src1, const float2& weight1,
                 iu::LinearHostMemory<float2, 4>& src2, const float2& weight2,iu::LinearHostMemory<float2, 4>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory<float3, 4>& src1, const float3& weight1,
                 iu::LinearHostMemory<float3, 4>& src2, const float3& weight2,iu::LinearHostMemory<float3, 4>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory<float4, 4>& src1, const float4& weight1,
                 iu::LinearHostMemory<float4, 4>& src2, const float4& weight2,iu::LinearHostMemory<float4, 4>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory<float, 5>& src1, const float& weight1,
                 iu::LinearHostMemory<float, 5>& src2, const float& weight2,iu::LinearHostMemory<float, 5>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory<float2, 5>& src1, const float2& weight1,
                 iu::LinearHostMemory<float2, 5>& src2, const float2& weight2,iu::LinearHostMemory<float2, 5>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory<float3, 5>& src1, const float3& weight1,
                 iu::LinearHostMemory<float3, 5>& src2, const float3& weight2,iu::LinearHostMemory<float3, 5>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory<float4, 5>& src1, const float4& weight1,
                 iu::LinearHostMemory<float4, 5>& src2, const float4& weight2,iu::LinearHostMemory<float4, 5>& dst);

IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory<double, 2>& src1, const double& weight1,
                 iu::LinearHostMemory<double, 2>& src2, const double& weight2,iu::LinearHostMemory<double, 2>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory<double2, 2>& src1, const double2& weight1,
                 iu::LinearHostMemory<double2, 2>& src2, const double2& weight2,iu::LinearHostMemory<double2, 2>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory<double3, 2>& src1, const double3& weight1,
                 iu::LinearHostMemory<double3, 2>& src2, const double3& weight2,iu::LinearHostMemory<double3, 2>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory<double4, 2>& src1, const double4& weight1,
                 iu::LinearHostMemory<double4, 2>& src2, const double4& weight2,iu::LinearHostMemory<double4, 2>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory<double, 3>& src1, const double& weight1,
                 iu::LinearHostMemory<double, 3>& src2, const double& weight2,iu::LinearHostMemory<double, 3>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory<double2, 3>& src1, const double2& weight1,
                 iu::LinearHostMemory<double2, 3>& src2, const double2& weight2,iu::LinearHostMemory<double2, 3>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory<double3, 3>& src1, const double3& weight1,
                 iu::LinearHostMemory<double3, 3>& src2, const double3& weight2,iu::LinearHostMemory<double3, 3>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory<double4, 3>& src1, const double4& weight1,
                 iu::LinearHostMemory<double4, 3>& src2, const double4& weight2,iu::LinearHostMemory<double4, 3>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory<double, 4>& src1, const double& weight1,
                 iu::LinearHostMemory<double, 4>& src2, const double& weight2,iu::LinearHostMemory<double, 4>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory<double2, 4>& src1, const double2& weight1,
                 iu::LinearHostMemory<double2, 4>& src2, const double2& weight2,iu::LinearHostMemory<double2, 4>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory<double3, 4>& src1, const double3& weight1,
                 iu::LinearHostMemory<double3, 4>& src2, const double3& weight2,iu::LinearHostMemory<double3, 4>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory<double4, 4>& src1, const double4& weight1,
                 iu::LinearHostMemory<double4, 4>& src2, const double4& weight2,iu::LinearHostMemory<double4, 4>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory<double, 5>& src1, const double& weight1,
                 iu::LinearHostMemory<double, 5>& src2, const double& weight2,iu::LinearHostMemory<double, 5>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory<double2, 5>& src1, const double2& weight1,
                 iu::LinearHostMemory<double2, 5>& src2, const double2& weight2,iu::LinearHostMemory<double2, 5>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory<double3, 5>& src1, const double3& weight1,
                 iu::LinearHostMemory<double3, 5>& src2, const double3& weight2,iu::LinearHostMemory<double3, 5>& dst);
IUMATH_DLLAPI void addWeighted(iu::LinearHostMemory<double4, 5>& src1, const double4& weight1,
                 iu::LinearHostMemory<double4, 5>& src2, const double4& weight2,iu::LinearHostMemory<double4, 5>& dst);

// pointwise multiply
/** Multiply an array to another array pointwise (can be called in-place)
 * \param src1 First source array
 * \param src2 Second source array
 * \param[out] dst Destination array
 */
IUMATH_DLLAPI void mul(iu::ImageGpu_32f_C1& src1, iu::ImageGpu_32f_C1& src2, iu::ImageGpu_32f_C1& dst);
IUMATH_DLLAPI void mul(iu::ImageGpu_32f_C2& src1, iu::ImageGpu_32f_C2& src2, iu::ImageGpu_32f_C2& dst);
IUMATH_DLLAPI void mul(iu::ImageGpu_32f_C3& src1, iu::ImageGpu_32f_C3& src2, iu::ImageGpu_32f_C3& dst);
IUMATH_DLLAPI void mul(iu::ImageGpu_32f_C4& src1, iu::ImageGpu_32f_C4& src2, iu::ImageGpu_32f_C4& dst);

IUMATH_DLLAPI void mul(iu::ImageGpu_64f_C1& src1, iu::ImageGpu_64f_C1& src2, iu::ImageGpu_64f_C1& dst);
IUMATH_DLLAPI void mul(iu::ImageGpu_64f_C2& src1, iu::ImageGpu_64f_C2& src2, iu::ImageGpu_64f_C2& dst);
IUMATH_DLLAPI void mul(iu::ImageGpu_64f_C3& src1, iu::ImageGpu_64f_C3& src2, iu::ImageGpu_64f_C3& dst);
IUMATH_DLLAPI void mul(iu::ImageGpu_64f_C4& src1, iu::ImageGpu_64f_C4& src2, iu::ImageGpu_64f_C4& dst);

IUMATH_DLLAPI void mul(iu::ImageGpu_8u_C1& src1, iu::ImageGpu_8u_C1& src2, iu::ImageGpu_8u_C1& dst);
IUMATH_DLLAPI void mul(iu::ImageGpu_8u_C4& src1, iu::ImageGpu_8u_C4& src2, iu::ImageGpu_8u_C4& dst);

IUMATH_DLLAPI void mul(iu::VolumeGpu_32f_C1& src1, iu::VolumeGpu_32f_C1& src2, iu::VolumeGpu_32f_C1& dst);

IUMATH_DLLAPI void mul(iu::VolumeGpu_64f_C1& src1, iu::VolumeGpu_64f_C1& src2, iu::VolumeGpu_64f_C1& dst);

IUMATH_DLLAPI void mul(iu::LinearDeviceMemory_32f_C1& src1, iu::LinearDeviceMemory_32f_C1& src2, iu::LinearDeviceMemory_32f_C1& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory_32f_C2& src1, iu::LinearDeviceMemory_32f_C2& src2, iu::LinearDeviceMemory_32f_C2& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory_32f_C3& src1, iu::LinearDeviceMemory_32f_C3& src2, iu::LinearDeviceMemory_32f_C3& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory_32f_C4& src1, iu::LinearDeviceMemory_32f_C4& src2, iu::LinearDeviceMemory_32f_C4& dst);


IUMATH_DLLAPI void mul(iu::LinearDeviceMemory<float, 2>& src1, iu::LinearDeviceMemory<float, 2>& src2, iu::LinearDeviceMemory<float, 2>& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory<float2, 2>& src1, iu::LinearDeviceMemory<float2, 2>& src2, iu::LinearDeviceMemory<float2, 2>& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory<float3, 2>& src1, iu::LinearDeviceMemory<float3, 2>& src2, iu::LinearDeviceMemory<float3, 2>& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory<float4, 2>& src1, iu::LinearDeviceMemory<float4, 2>& src2, iu::LinearDeviceMemory<float4, 2>& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory<float, 3>& src1, iu::LinearDeviceMemory<float, 3>& src2, iu::LinearDeviceMemory<float, 3>& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory<float2, 3>& src1, iu::LinearDeviceMemory<float2, 3>& src2, iu::LinearDeviceMemory<float2, 3>& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory<float3, 3>& src1, iu::LinearDeviceMemory<float3, 3>& src2, iu::LinearDeviceMemory<float3, 3>& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory<float4, 3>& src1, iu::LinearDeviceMemory<float4, 3>& src2, iu::LinearDeviceMemory<float4, 3>& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory<float, 4>& src1, iu::LinearDeviceMemory<float, 4>& src2, iu::LinearDeviceMemory<float, 4>& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory<float2, 4>& src1, iu::LinearDeviceMemory<float2, 4>& src2, iu::LinearDeviceMemory<float2, 4>& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory<float3, 4>& src1, iu::LinearDeviceMemory<float3, 4>& src2, iu::LinearDeviceMemory<float3, 4>& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory<float4, 4>& src1, iu::LinearDeviceMemory<float4, 4>& src2, iu::LinearDeviceMemory<float4, 4>& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory<float, 5>& src1, iu::LinearDeviceMemory<float, 5>& src2, iu::LinearDeviceMemory<float, 5>& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory<float2, 5>& src1, iu::LinearDeviceMemory<float2, 5>& src2, iu::LinearDeviceMemory<float2, 5>& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory<float3, 5>& src1, iu::LinearDeviceMemory<float3, 5>& src2, iu::LinearDeviceMemory<float3, 5>& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory<float4, 5>& src1, iu::LinearDeviceMemory<float4, 5>& src2, iu::LinearDeviceMemory<float4, 5>& dst);

IUMATH_DLLAPI void mul(iu::LinearDeviceMemory<double, 2>& src1, iu::LinearDeviceMemory<double, 2>& src2, iu::LinearDeviceMemory<double, 2>& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory<double2, 2>& src1, iu::LinearDeviceMemory<double2, 2>& src2, iu::LinearDeviceMemory<double2, 2>& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory<double3, 2>& src1, iu::LinearDeviceMemory<double3, 2>& src2, iu::LinearDeviceMemory<double3, 2>& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory<double4, 2>& src1, iu::LinearDeviceMemory<double4, 2>& src2, iu::LinearDeviceMemory<double4, 2>& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory<double, 3>& src1, iu::LinearDeviceMemory<double, 3>& src2, iu::LinearDeviceMemory<double, 3>& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory<double2, 3>& src1, iu::LinearDeviceMemory<double2, 3>& src2, iu::LinearDeviceMemory<double2, 3>& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory<double3, 3>& src1, iu::LinearDeviceMemory<double3, 3>& src2, iu::LinearDeviceMemory<double3, 3>& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory<double4, 3>& src1, iu::LinearDeviceMemory<double4, 3>& src2, iu::LinearDeviceMemory<double4, 3>& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory<double, 4>& src1, iu::LinearDeviceMemory<double, 4>& src2, iu::LinearDeviceMemory<double, 4>& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory<double2, 4>& src1, iu::LinearDeviceMemory<double2, 4>& src2, iu::LinearDeviceMemory<double2, 4>& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory<double3, 4>& src1, iu::LinearDeviceMemory<double3, 4>& src2, iu::LinearDeviceMemory<double3, 4>& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory<double4, 4>& src1, iu::LinearDeviceMemory<double4, 4>& src2, iu::LinearDeviceMemory<double4, 4>& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory<double, 5>& src1, iu::LinearDeviceMemory<double, 5>& src2, iu::LinearDeviceMemory<double, 5>& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory<double2, 5>& src1, iu::LinearDeviceMemory<double2, 5>& src2, iu::LinearDeviceMemory<double2, 5>& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory<double3, 5>& src1, iu::LinearDeviceMemory<double3, 5>& src2, iu::LinearDeviceMemory<double3, 5>& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory<double4, 5>& src1, iu::LinearDeviceMemory<double4, 5>& src2, iu::LinearDeviceMemory<double4, 5>& dst);

IUMATH_DLLAPI void mul(iu::LinearDeviceMemory_64f_C1& src1, iu::LinearDeviceMemory_64f_C1& src2, iu::LinearDeviceMemory_64f_C1& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory_64f_C2& src1, iu::LinearDeviceMemory_64f_C2& src2, iu::LinearDeviceMemory_64f_C2& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory_64f_C3& src1, iu::LinearDeviceMemory_64f_C3& src2, iu::LinearDeviceMemory_64f_C3& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory_64f_C4& src1, iu::LinearDeviceMemory_64f_C4& src2, iu::LinearDeviceMemory_64f_C4& dst);


IUMATH_DLLAPI void mul(iu::LinearDeviceMemory_8u_C1& src1, iu::LinearDeviceMemory_8u_C1& src2, iu::LinearDeviceMemory_8u_C1& dst);
IUMATH_DLLAPI void mul(iu::LinearDeviceMemory_8u_C4& src1, iu::LinearDeviceMemory_8u_C4& src2, iu::LinearDeviceMemory_8u_C4& dst);

IUMATH_DLLAPI void mul(iu::ImageCpu_32f_C1& src1, iu::ImageCpu_32f_C1& src2, iu::ImageCpu_32f_C1& dst);
IUMATH_DLLAPI void mul(iu::ImageCpu_32f_C2& src1, iu::ImageCpu_32f_C2& src2, iu::ImageCpu_32f_C2& dst);
IUMATH_DLLAPI void mul(iu::ImageCpu_32f_C3& src1, iu::ImageCpu_32f_C3& src2, iu::ImageCpu_32f_C3& dst);
IUMATH_DLLAPI void mul(iu::ImageCpu_32f_C4& src1, iu::ImageCpu_32f_C4& src2, iu::ImageCpu_32f_C4& dst);

IUMATH_DLLAPI void mul(iu::ImageCpu_64f_C1& src1, iu::ImageCpu_64f_C1& src2, iu::ImageCpu_64f_C1& dst);
IUMATH_DLLAPI void mul(iu::ImageCpu_64f_C2& src1, iu::ImageCpu_64f_C2& src2, iu::ImageCpu_64f_C2& dst);
IUMATH_DLLAPI void mul(iu::ImageCpu_64f_C3& src1, iu::ImageCpu_64f_C3& src2, iu::ImageCpu_64f_C3& dst);
IUMATH_DLLAPI void mul(iu::ImageCpu_64f_C4& src1, iu::ImageCpu_64f_C4& src2, iu::ImageCpu_64f_C4& dst);

IUMATH_DLLAPI void mul(iu::ImageCpu_8u_C1& src1, iu::ImageCpu_8u_C1& src2, iu::ImageCpu_8u_C1& dst);
IUMATH_DLLAPI void mul(iu::ImageCpu_8u_C4& src1, iu::ImageCpu_8u_C4& src2, iu::ImageCpu_8u_C4& dst);

IUMATH_DLLAPI void mul(iu::VolumeCpu_32f_C1& src1, iu::VolumeCpu_32f_C1& src2, iu::VolumeCpu_32f_C1& dst);

IUMATH_DLLAPI void mul(iu::VolumeCpu_64f_C1& src1, iu::VolumeCpu_64f_C1& src2, iu::VolumeCpu_64f_C1& dst);

IUMATH_DLLAPI void mul(iu::LinearHostMemory_32f_C1& src1, iu::LinearHostMemory_32f_C1& src2, iu::LinearHostMemory_32f_C1& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory_32f_C2& src1, iu::LinearHostMemory_32f_C2& src2, iu::LinearHostMemory_32f_C2& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory_32f_C3& src1, iu::LinearHostMemory_32f_C3& src2, iu::LinearHostMemory_32f_C3& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory_32f_C4& src1, iu::LinearHostMemory_32f_C4& src2, iu::LinearHostMemory_32f_C4& dst);

IUMATH_DLLAPI void mul(iu::LinearHostMemory<float, 2>& src1, iu::LinearHostMemory<float, 2>& src2, iu::LinearHostMemory<float, 2>& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory<float2, 2>& src1, iu::LinearHostMemory<float2, 2>& src2, iu::LinearHostMemory<float2, 2>& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory<float3, 2>& src1, iu::LinearHostMemory<float3, 2>& src2, iu::LinearHostMemory<float3, 2>& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory<float4, 2>& src1, iu::LinearHostMemory<float4, 2>& src2, iu::LinearHostMemory<float4, 2>& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory<float, 3>& src1, iu::LinearHostMemory<float, 3>& src2, iu::LinearHostMemory<float, 3>& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory<float2, 3>& src1, iu::LinearHostMemory<float2, 3>& src2, iu::LinearHostMemory<float2, 3>& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory<float3, 3>& src1, iu::LinearHostMemory<float3, 3>& src2, iu::LinearHostMemory<float3, 3>& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory<float4, 3>& src1, iu::LinearHostMemory<float4, 3>& src2, iu::LinearHostMemory<float4, 3>& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory<float, 4>& src1, iu::LinearHostMemory<float, 4>& src2, iu::LinearHostMemory<float, 4>& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory<float2, 4>& src1, iu::LinearHostMemory<float2, 4>& src2, iu::LinearHostMemory<float2, 4>& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory<float3, 4>& src1, iu::LinearHostMemory<float3, 4>& src2, iu::LinearHostMemory<float3, 4>& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory<float4, 4>& src1, iu::LinearHostMemory<float4, 4>& src2, iu::LinearHostMemory<float4, 4>& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory<float, 5>& src1, iu::LinearHostMemory<float, 5>& src2, iu::LinearHostMemory<float, 5>& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory<float2, 5>& src1, iu::LinearHostMemory<float2, 5>& src2, iu::LinearHostMemory<float2, 5>& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory<float3, 5>& src1, iu::LinearHostMemory<float3, 5>& src2, iu::LinearHostMemory<float3, 5>& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory<float4, 5>& src1, iu::LinearHostMemory<float4, 5>& src2, iu::LinearHostMemory<float4, 5>& dst);

IUMATH_DLLAPI void mul(iu::LinearHostMemory<double, 2>& src1, iu::LinearHostMemory<double, 2>& src2, iu::LinearHostMemory<double, 2>& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory<double2, 2>& src1, iu::LinearHostMemory<double2, 2>& src2, iu::LinearHostMemory<double2, 2>& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory<double3, 2>& src1, iu::LinearHostMemory<double3, 2>& src2, iu::LinearHostMemory<double3, 2>& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory<double4, 2>& src1, iu::LinearHostMemory<double4, 2>& src2, iu::LinearHostMemory<double4, 2>& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory<double, 3>& src1, iu::LinearHostMemory<double, 3>& src2, iu::LinearHostMemory<double, 3>& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory<double2, 3>& src1, iu::LinearHostMemory<double2, 3>& src2, iu::LinearHostMemory<double2, 3>& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory<double3, 3>& src1, iu::LinearHostMemory<double3, 3>& src2, iu::LinearHostMemory<double3, 3>& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory<double4, 3>& src1, iu::LinearHostMemory<double4, 3>& src2, iu::LinearHostMemory<double4, 3>& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory<double, 4>& src1, iu::LinearHostMemory<double, 4>& src2, iu::LinearHostMemory<double, 4>& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory<double2, 4>& src1, iu::LinearHostMemory<double2, 4>& src2, iu::LinearHostMemory<double2, 4>& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory<double3, 4>& src1, iu::LinearHostMemory<double3, 4>& src2, iu::LinearHostMemory<double3, 4>& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory<double4, 4>& src1, iu::LinearHostMemory<double4, 4>& src2, iu::LinearHostMemory<double4, 4>& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory<double, 5>& src1, iu::LinearHostMemory<double, 5>& src2, iu::LinearHostMemory<double, 5>& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory<double2, 5>& src1, iu::LinearHostMemory<double2, 5>& src2, iu::LinearHostMemory<double2, 5>& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory<double3, 5>& src1, iu::LinearHostMemory<double3, 5>& src2, iu::LinearHostMemory<double3, 5>& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory<double4, 5>& src1, iu::LinearHostMemory<double4, 5>& src2, iu::LinearHostMemory<double4, 5>& dst);

IUMATH_DLLAPI void mul(iu::LinearHostMemory_64f_C1& src1, iu::LinearHostMemory_64f_C1& src2, iu::LinearHostMemory_64f_C1& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory_64f_C2& src1, iu::LinearHostMemory_64f_C2& src2, iu::LinearHostMemory_64f_C2& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory_64f_C3& src1, iu::LinearHostMemory_64f_C3& src2, iu::LinearHostMemory_64f_C3& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory_64f_C4& src1, iu::LinearHostMemory_64f_C4& src2, iu::LinearHostMemory_64f_C4& dst);


IUMATH_DLLAPI void mul(iu::LinearHostMemory_8u_C1& src1, iu::LinearHostMemory_8u_C1& src2, iu::LinearHostMemory_8u_C1& dst);
IUMATH_DLLAPI void mul(iu::LinearHostMemory_8u_C4& src1, iu::LinearHostMemory_8u_C4& src2, iu::LinearHostMemory_8u_C4& dst);

// set value
/** Set array to a specified value
 * \param dst Destination array
 * \param value Value to set
 */
IUMATH_DLLAPI void fill(iu::ImageGpu_32f_C1& dst, float value);
IUMATH_DLLAPI void fill(iu::ImageGpu_32f_C2& dst, float2 value);
IUMATH_DLLAPI void fill(iu::ImageGpu_32f_C4& dst, float4 value);
IUMATH_DLLAPI void fill(iu::ImageGpu_64f_C1& dst, double value);
IUMATH_DLLAPI void fill(iu::ImageGpu_64f_C2& dst, double2 value);
IUMATH_DLLAPI void fill(iu::ImageGpu_64f_C4& dst, double4 value);
IUMATH_DLLAPI void fill(iu::ImageGpu_32u_C1& dst, unsigned int value);
IUMATH_DLLAPI void fill(iu::ImageGpu_8u_C1& dst, unsigned char value);
IUMATH_DLLAPI void fill(iu::ImageGpu_8u_C2& dst, uchar2 value);
IUMATH_DLLAPI void fill(iu::ImageGpu_8u_C4& dst, uchar4 value);

IUMATH_DLLAPI void fill(iu::ImageCpu_32f_C1& dst, float value);
IUMATH_DLLAPI void fill(iu::ImageCpu_32f_C2& dst, float2 value);
IUMATH_DLLAPI void fill(iu::ImageCpu_32f_C4& dst, float4 value);
IUMATH_DLLAPI void fill(iu::ImageCpu_64f_C1& dst, double value);
IUMATH_DLLAPI void fill(iu::ImageCpu_64f_C2& dst, double2 value);
IUMATH_DLLAPI void fill(iu::ImageCpu_64f_C4& dst, double4 value);
IUMATH_DLLAPI void fill(iu::ImageCpu_8u_C1& dst, unsigned char value);
IUMATH_DLLAPI void fill(iu::ImageCpu_8u_C2& dst, uchar2 value);
IUMATH_DLLAPI void fill(iu::ImageCpu_8u_C4& dst, uchar4 value);

IUMATH_DLLAPI void fill(iu::LinearDeviceMemory_32f_C1& dst, float value);
IUMATH_DLLAPI void fill(iu::LinearDeviceMemory_32f_C2& dst, float2 value);
IUMATH_DLLAPI void fill(iu::LinearDeviceMemory_32f_C3& dst, float3 value);

IUMATH_DLLAPI void fill(iu::LinearDeviceMemory_64f_C1& dst, double value);
IUMATH_DLLAPI void fill(iu::LinearDeviceMemory_64f_C2& dst, double2 value);
IUMATH_DLLAPI void fill(iu::LinearDeviceMemory_64f_C3& dst, double3 value);

IUMATH_DLLAPI void fill(iu::LinearHostMemory_32f_C1& dst, float value);
IUMATH_DLLAPI void fill(iu::LinearHostMemory_32f_C2& dst, float2 value);
IUMATH_DLLAPI void fill(iu::LinearHostMemory_32f_C3& dst, float3 value);

IUMATH_DLLAPI void fill(iu::LinearHostMemory_64f_C1& dst, double value);
IUMATH_DLLAPI void fill(iu::LinearHostMemory_64f_C2& dst, double2 value);
IUMATH_DLLAPI void fill(iu::LinearHostMemory_64f_C3& dst, double3 value);

IUMATH_DLLAPI void fill(iu::VolumeGpu_32f_C1& dst, float value);
IUMATH_DLLAPI void fill(iu::VolumeGpu_32f_C2& dst, float2 value);
IUMATH_DLLAPI void fill(iu::VolumeGpu_32f_C3& dst, float3 value);

IUMATH_DLLAPI void fill(iu::VolumeGpu_64f_C1& dst, double value);
IUMATH_DLLAPI void fill(iu::VolumeGpu_64f_C2& dst, double2 value);
IUMATH_DLLAPI void fill(iu::VolumeGpu_64f_C3& dst, double3 value);

IUMATH_DLLAPI void fill(iu::VolumeCpu_32f_C1& dst, float value);
IUMATH_DLLAPI void fill(iu::VolumeCpu_32f_C2& dst, float2 value);
IUMATH_DLLAPI void fill(iu::VolumeCpu_32f_C3& dst, float3 value);

IUMATH_DLLAPI void fill(iu::VolumeCpu_64f_C1& dst, double value);
IUMATH_DLLAPI void fill(iu::VolumeCpu_64f_C2& dst, double2 value);
IUMATH_DLLAPI void fill(iu::VolumeCpu_64f_C3& dst, double3 value);

IUMATH_DLLAPI void fill(iu::LinearDeviceMemory<float, 2>& dst, float value);
IUMATH_DLLAPI void fill(iu::LinearDeviceMemory<float, 3>& dst, float value);
IUMATH_DLLAPI void fill(iu::LinearDeviceMemory<float, 4>& dst, float value);
IUMATH_DLLAPI void fill(iu::LinearDeviceMemory<float, 5>& dst, float value);
IUMATH_DLLAPI void fill(iu::LinearDeviceMemory<float2, 2>& dst, float2 value);
IUMATH_DLLAPI void fill(iu::LinearDeviceMemory<float2, 3>& dst, float2 value);
IUMATH_DLLAPI void fill(iu::LinearDeviceMemory<float2, 4>& dst, float2 value);
IUMATH_DLLAPI void fill(iu::LinearDeviceMemory<float2, 5>& dst, float2 value);
IUMATH_DLLAPI void fill(iu::LinearDeviceMemory<float3, 2>& dst, float3 value);
IUMATH_DLLAPI void fill(iu::LinearDeviceMemory<float3, 3>& dst, float3 value);
IUMATH_DLLAPI void fill(iu::LinearDeviceMemory<float3, 4>& dst, float3 value);
IUMATH_DLLAPI void fill(iu::LinearDeviceMemory<float3, 5>& dst, float3 value);
IUMATH_DLLAPI void fill(iu::LinearDeviceMemory<float4, 2>& dst, float4 value);
IUMATH_DLLAPI void fill(iu::LinearDeviceMemory<float4, 3>& dst, float4 value);
IUMATH_DLLAPI void fill(iu::LinearDeviceMemory<float4, 4>& dst, float4 value);
IUMATH_DLLAPI void fill(iu::LinearDeviceMemory<float4, 5>& dst, float4 value);

IUMATH_DLLAPI void fill(iu::LinearHostMemory<float, 2>& dst, float value);
IUMATH_DLLAPI void fill(iu::LinearHostMemory<float, 3>& dst, float value);
IUMATH_DLLAPI void fill(iu::LinearHostMemory<float, 4>& dst, float value);
IUMATH_DLLAPI void fill(iu::LinearHostMemory<float, 5>& dst, float value);
IUMATH_DLLAPI void fill(iu::LinearHostMemory<float2, 2>& dst, float2 value);
IUMATH_DLLAPI void fill(iu::LinearHostMemory<float2, 3>& dst, float2 value);
IUMATH_DLLAPI void fill(iu::LinearHostMemory<float2, 4>& dst, float2 value);
IUMATH_DLLAPI void fill(iu::LinearHostMemory<float2, 5>& dst, float2 value);
IUMATH_DLLAPI void fill(iu::LinearHostMemory<float3, 2>& dst, float3 value);
IUMATH_DLLAPI void fill(iu::LinearHostMemory<float3, 3>& dst, float3 value);
IUMATH_DLLAPI void fill(iu::LinearHostMemory<float3, 4>& dst, float3 value);
IUMATH_DLLAPI void fill(iu::LinearHostMemory<float3, 5>& dst, float3 value);
IUMATH_DLLAPI void fill(iu::LinearHostMemory<float4, 2>& dst, float4 value);
IUMATH_DLLAPI void fill(iu::LinearHostMemory<float4, 3>& dst, float4 value);
IUMATH_DLLAPI void fill(iu::LinearHostMemory<float4, 4>& dst, float4 value);
IUMATH_DLLAPI void fill(iu::LinearHostMemory<float4, 5>& dst, float4 value);

IUMATH_DLLAPI void fill(iu::LinearDeviceMemory<double, 2>& dst, double value);
IUMATH_DLLAPI void fill(iu::LinearDeviceMemory<double, 3>& dst, double value);
IUMATH_DLLAPI void fill(iu::LinearDeviceMemory<double, 4>& dst, double value);
IUMATH_DLLAPI void fill(iu::LinearDeviceMemory<double, 5>& dst, double value);
IUMATH_DLLAPI void fill(iu::LinearDeviceMemory<double2, 2>& dst, double2 value);
IUMATH_DLLAPI void fill(iu::LinearDeviceMemory<double2, 3>& dst, double2 value);
IUMATH_DLLAPI void fill(iu::LinearDeviceMemory<double2, 4>& dst, double2 value);
IUMATH_DLLAPI void fill(iu::LinearDeviceMemory<double2, 5>& dst, double2 value);
IUMATH_DLLAPI void fill(iu::LinearDeviceMemory<double3, 2>& dst, double3 value);
IUMATH_DLLAPI void fill(iu::LinearDeviceMemory<double3, 3>& dst, double3 value);
IUMATH_DLLAPI void fill(iu::LinearDeviceMemory<double3, 4>& dst, double3 value);
IUMATH_DLLAPI void fill(iu::LinearDeviceMemory<double3, 5>& dst, double3 value);
IUMATH_DLLAPI void fill(iu::LinearDeviceMemory<double4, 2>& dst, double4 value);
IUMATH_DLLAPI void fill(iu::LinearDeviceMemory<double4, 3>& dst, double4 value);
IUMATH_DLLAPI void fill(iu::LinearDeviceMemory<double4, 4>& dst, double4 value);
IUMATH_DLLAPI void fill(iu::LinearDeviceMemory<double4, 5>& dst, double4 value);

IUMATH_DLLAPI void fill(iu::LinearHostMemory<double, 2>& dst, double value);
IUMATH_DLLAPI void fill(iu::LinearHostMemory<double, 3>& dst, double value);
IUMATH_DLLAPI void fill(iu::LinearHostMemory<double, 4>& dst, double value);
IUMATH_DLLAPI void fill(iu::LinearHostMemory<double, 5>& dst, double value);
IUMATH_DLLAPI void fill(iu::LinearHostMemory<double2, 2>& dst, double2 value);
IUMATH_DLLAPI void fill(iu::LinearHostMemory<double2, 3>& dst, double2 value);
IUMATH_DLLAPI void fill(iu::LinearHostMemory<double2, 4>& dst, double2 value);
IUMATH_DLLAPI void fill(iu::LinearHostMemory<double2, 5>& dst, double2 value);
IUMATH_DLLAPI void fill(iu::LinearHostMemory<double3, 2>& dst, double3 value);
IUMATH_DLLAPI void fill(iu::LinearHostMemory<double3, 3>& dst, double3 value);
IUMATH_DLLAPI void fill(iu::LinearHostMemory<double3, 4>& dst, double3 value);
IUMATH_DLLAPI void fill(iu::LinearHostMemory<double3, 5>& dst, double3 value);
IUMATH_DLLAPI void fill(iu::LinearHostMemory<double4, 2>& dst, double4 value);
IUMATH_DLLAPI void fill(iu::LinearHostMemory<double4, 3>& dst, double4 value);
IUMATH_DLLAPI void fill(iu::LinearHostMemory<double4, 4>& dst, double4 value);
IUMATH_DLLAPI void fill(iu::LinearHostMemory<double4, 5>& dst, double4 value);


/** Split planes of a two channel image (e.g. complex image)
 * \param[in] src  Combined image (e.g. complex image)
 * \param[out] dst1 First channel (e.g. real part)
 * \param[out] dst2 Second channel (e.g. imaginary part)
 *
 */
IUMATH_DLLAPI void splitPlanes(iu::VolumeCpu_32f_C2& src, iu::VolumeCpu_32f_C1& dst1, iu::VolumeCpu_32f_C1& dst2);
IUMATH_DLLAPI void splitPlanes(iu::VolumeGpu_32f_C2& src, iu::VolumeGpu_32f_C1& dst1, iu::VolumeGpu_32f_C1& dst2);
IUMATH_DLLAPI void splitPlanes(iu::ImageCpu_32f_C2& src, iu::ImageCpu_32f_C1& dst1, iu::ImageCpu_32f_C1& dst2);
IUMATH_DLLAPI void splitPlanes(iu::ImageGpu_32f_C2& src, iu::ImageGpu_32f_C1& dst1, iu::ImageGpu_32f_C1& dst2);
IUMATH_DLLAPI void splitPlanes(iu::LinearDeviceMemory_32f_C2& src, iu::LinearDeviceMemory_32f_C1& dst1, iu::LinearDeviceMemory_32f_C1& dst2);
IUMATH_DLLAPI void splitPlanes(iu::LinearHostMemory_32f_C2& src, iu::LinearHostMemory_32f_C1& dst1, iu::LinearHostMemory_32f_C1& dst2);

IUMATH_DLLAPI void splitPlanes(iu::LinearDeviceMemory<float2, 2>& src, iu::LinearDeviceMemory<float, 2>& dst1, iu::LinearDeviceMemory<float, 2>& dst2);
IUMATH_DLLAPI void splitPlanes(iu::LinearHostMemory<float2, 2>& src, iu::LinearHostMemory<float, 2>& dst1, iu::LinearHostMemory<float, 2>& dst2);
IUMATH_DLLAPI void splitPlanes(iu::LinearDeviceMemory<float2, 3>& src, iu::LinearDeviceMemory<float, 3>& dst1, iu::LinearDeviceMemory<float, 3>& dst2);
IUMATH_DLLAPI void splitPlanes(iu::LinearHostMemory<float2, 3>& src, iu::LinearHostMemory<float, 3>& dst1, iu::LinearHostMemory<float, 3>& dst2);
IUMATH_DLLAPI void splitPlanes(iu::LinearDeviceMemory<float2, 4>& src, iu::LinearDeviceMemory<float, 4>& dst1, iu::LinearDeviceMemory<float, 4>& dst2);
IUMATH_DLLAPI void splitPlanes(iu::LinearHostMemory<float2, 4>& src, iu::LinearHostMemory<float, 4>& dst1, iu::LinearHostMemory<float, 4>& dst2);
IUMATH_DLLAPI void splitPlanes(iu::LinearDeviceMemory<float2, 5>& src, iu::LinearDeviceMemory<float, 5>& dst1, iu::LinearDeviceMemory<float, 5>& dst2);
IUMATH_DLLAPI void splitPlanes(iu::LinearHostMemory<float2, 5>& src, iu::LinearHostMemory<float, 5>& dst1, iu::LinearHostMemory<float, 5>& dst2);

IUMATH_DLLAPI void splitPlanes(iu::VolumeCpu_64f_C2& src, iu::VolumeCpu_64f_C1& dst1, iu::VolumeCpu_64f_C1& dst2);
IUMATH_DLLAPI void splitPlanes(iu::VolumeGpu_64f_C2& src, iu::VolumeGpu_64f_C1& dst1, iu::VolumeGpu_64f_C1& dst2);
IUMATH_DLLAPI void splitPlanes(iu::ImageCpu_64f_C2& src, iu::ImageCpu_64f_C1& dst1, iu::ImageCpu_64f_C1& dst2);
IUMATH_DLLAPI void splitPlanes(iu::ImageGpu_64f_C2& src, iu::ImageGpu_64f_C1& dst1, iu::ImageGpu_64f_C1& dst2);
IUMATH_DLLAPI void splitPlanes(iu::LinearDeviceMemory_64f_C2& src, iu::LinearDeviceMemory_64f_C1& dst1, iu::LinearDeviceMemory_64f_C1& dst2);
IUMATH_DLLAPI void splitPlanes(iu::LinearHostMemory_64f_C2& src, iu::LinearHostMemory_64f_C1& dst1, iu::LinearHostMemory_64f_C1& dst2);

IUMATH_DLLAPI void splitPlanes(iu::LinearDeviceMemory<double2, 2>& src, iu::LinearDeviceMemory<double, 2>& dst1, iu::LinearDeviceMemory<double, 2>& dst2);
IUMATH_DLLAPI void splitPlanes(iu::LinearHostMemory<double2, 2>& src, iu::LinearHostMemory<double, 2>& dst1, iu::LinearHostMemory<double, 2>& dst2);
IUMATH_DLLAPI void splitPlanes(iu::LinearDeviceMemory<double2, 3>& src, iu::LinearDeviceMemory<double, 3>& dst1, iu::LinearDeviceMemory<double, 3>& dst2);
IUMATH_DLLAPI void splitPlanes(iu::LinearHostMemory<double2, 3>& src, iu::LinearHostMemory<double, 3>& dst1, iu::LinearHostMemory<double, 3>& dst2);
IUMATH_DLLAPI void splitPlanes(iu::LinearDeviceMemory<double2, 4>& src, iu::LinearDeviceMemory<double, 4>& dst1, iu::LinearDeviceMemory<double, 4>& dst2);
IUMATH_DLLAPI void splitPlanes(iu::LinearHostMemory<double2, 4>& src, iu::LinearHostMemory<double, 4>& dst1, iu::LinearHostMemory<double, 4>& dst2);
IUMATH_DLLAPI void splitPlanes(iu::LinearDeviceMemory<double2, 5>& src, iu::LinearDeviceMemory<double, 5>& dst1, iu::LinearDeviceMemory<double, 5>& dst2);
IUMATH_DLLAPI void splitPlanes(iu::LinearHostMemory<double2, 5>& src, iu::LinearHostMemory<double, 5>& dst1, iu::LinearHostMemory<double, 5>& dst2);


/** Split planes of a three channel image (e.g. rgb image)
 * \param[in] src  Combined image (e.g. rgb image)
 * \param[out] dst1 First channel (e.g. r channel)
 * \param[out] dst2 Second channel (e.g. g channel)
 * \param[out] dst3 Third channel (e.g. b channel)
 *
 */
IUMATH_DLLAPI void splitPlanes(iu::VolumeCpu_32f_C3& src, iu::VolumeCpu_32f_C1& dst1, iu::VolumeCpu_32f_C1& dst2, iu::VolumeCpu_32f_C1& dst3);
IUMATH_DLLAPI void splitPlanes(iu::VolumeGpu_32f_C3& src, iu::VolumeGpu_32f_C1& dst1, iu::VolumeGpu_32f_C1& dst2, iu::VolumeGpu_32f_C1& dst3);
IUMATH_DLLAPI void splitPlanes(iu::ImageCpu_32f_C3& src, iu::ImageCpu_32f_C1& dst1, iu::ImageCpu_32f_C1& dst2, iu::ImageCpu_32f_C1& dst3);
IUMATH_DLLAPI void splitPlanes(iu::ImageGpu_32f_C3& src, iu::ImageGpu_32f_C1& dst1, iu::ImageGpu_32f_C1& dst2, iu::ImageGpu_32f_C1& dst3);
IUMATH_DLLAPI void splitPlanes(iu::LinearHostMemory_32f_C3& src, iu::LinearHostMemory_32f_C1& dst1, iu::LinearHostMemory_32f_C1& dst2, iu::LinearHostMemory_32f_C1& dst3);
IUMATH_DLLAPI void splitPlanes(iu::LinearDeviceMemory_32f_C3& src, iu::LinearDeviceMemory_32f_C1& dst1, iu::LinearDeviceMemory_32f_C1& dst2, iu::LinearDeviceMemory_32f_C1& dst3);

IUMATH_DLLAPI void splitPlanes(iu::LinearHostMemory<float3, 2>& src, iu::LinearHostMemory<float, 2>& dst1, iu::LinearHostMemory<float, 2>& dst2, iu::LinearHostMemory<float, 2>& dst3);
IUMATH_DLLAPI void splitPlanes(iu::LinearDeviceMemory<float3, 2>& src, iu::LinearDeviceMemory<float, 2>& dst1, iu::LinearDeviceMemory<float, 2>& dst2, iu::LinearDeviceMemory<float, 2>& dst3);
IUMATH_DLLAPI void splitPlanes(iu::LinearHostMemory<float3, 3>& src, iu::LinearHostMemory<float, 3>& dst1, iu::LinearHostMemory<float, 3>& dst2, iu::LinearHostMemory<float, 3>& dst3);
IUMATH_DLLAPI void splitPlanes(iu::LinearDeviceMemory<float3, 3>& src, iu::LinearDeviceMemory<float, 3>& dst1, iu::LinearDeviceMemory<float, 3>& dst2, iu::LinearDeviceMemory<float, 3>& dst3);
IUMATH_DLLAPI void splitPlanes(iu::LinearHostMemory<float3, 4>& src, iu::LinearHostMemory<float, 4>& dst1, iu::LinearHostMemory<float, 4>& dst2, iu::LinearHostMemory<float, 4>& dst3);
IUMATH_DLLAPI void splitPlanes(iu::LinearDeviceMemory<float3, 4>& src, iu::LinearDeviceMemory<float, 4>& dst1, iu::LinearDeviceMemory<float, 4>& dst2, iu::LinearDeviceMemory<float, 4>& dst3);
IUMATH_DLLAPI void splitPlanes(iu::LinearHostMemory<float3, 5>& src, iu::LinearHostMemory<float, 5>& dst1, iu::LinearHostMemory<float, 5>& dst2, iu::LinearHostMemory<float, 5>& dst3);
IUMATH_DLLAPI void splitPlanes(iu::LinearDeviceMemory<float3, 5>& src, iu::LinearDeviceMemory<float, 5>& dst1, iu::LinearDeviceMemory<float, 5>& dst2, iu::LinearDeviceMemory<float, 5>& dst3);

IUMATH_DLLAPI void splitPlanes(iu::VolumeCpu_64f_C3& src, iu::VolumeCpu_64f_C1& dst1, iu::VolumeCpu_64f_C1& dst2, iu::VolumeCpu_64f_C1& dst3);
IUMATH_DLLAPI void splitPlanes(iu::VolumeGpu_64f_C3& src, iu::VolumeGpu_64f_C1& dst1, iu::VolumeGpu_64f_C1& dst2, iu::VolumeGpu_64f_C1& dst3);
IUMATH_DLLAPI void splitPlanes(iu::ImageCpu_64f_C3& src, iu::ImageCpu_64f_C1& dst1, iu::ImageCpu_64f_C1& dst2, iu::ImageCpu_64f_C1& dst3);
IUMATH_DLLAPI void splitPlanes(iu::ImageGpu_64f_C3& src, iu::ImageGpu_64f_C1& dst1, iu::ImageGpu_64f_C1& dst2, iu::ImageGpu_64f_C1& dst3);
IUMATH_DLLAPI void splitPlanes(iu::LinearHostMemory_64f_C3& src, iu::LinearHostMemory_64f_C1& dst1, iu::LinearHostMemory_64f_C1& dst2, iu::LinearHostMemory_64f_C1& dst3);
IUMATH_DLLAPI void splitPlanes(iu::LinearDeviceMemory_64f_C3& src, iu::LinearDeviceMemory_64f_C1& dst1, iu::LinearDeviceMemory_64f_C1& dst2, iu::LinearDeviceMemory_64f_C1& dst3);

IUMATH_DLLAPI void splitPlanes(iu::LinearHostMemory<double3, 2>& src, iu::LinearHostMemory<double, 2>& dst1, iu::LinearHostMemory<double, 2>& dst2, iu::LinearHostMemory<double, 2>& dst3);
IUMATH_DLLAPI void splitPlanes(iu::LinearDeviceMemory<double3, 2>& src, iu::LinearDeviceMemory<double, 2>& dst1, iu::LinearDeviceMemory<double, 2>& dst2, iu::LinearDeviceMemory<double, 2>& dst3);
IUMATH_DLLAPI void splitPlanes(iu::LinearHostMemory<double3, 3>& src, iu::LinearHostMemory<double, 3>& dst1, iu::LinearHostMemory<double, 3>& dst2, iu::LinearHostMemory<double, 3>& dst3);
IUMATH_DLLAPI void splitPlanes(iu::LinearDeviceMemory<double3, 3>& src, iu::LinearDeviceMemory<double, 3>& dst1, iu::LinearDeviceMemory<double, 3>& dst2, iu::LinearDeviceMemory<double, 3>& dst3);
IUMATH_DLLAPI void splitPlanes(iu::LinearHostMemory<double3, 4>& src, iu::LinearHostMemory<double, 4>& dst1, iu::LinearHostMemory<double, 4>& dst2, iu::LinearHostMemory<double, 4>& dst3);
IUMATH_DLLAPI void splitPlanes(iu::LinearDeviceMemory<double3, 4>& src, iu::LinearDeviceMemory<double, 4>& dst1, iu::LinearDeviceMemory<double, 4>& dst2, iu::LinearDeviceMemory<double, 4>& dst3);
IUMATH_DLLAPI void splitPlanes(iu::LinearHostMemory<double3, 5>& src, iu::LinearHostMemory<double, 5>& dst1, iu::LinearHostMemory<double, 5>& dst2, iu::LinearHostMemory<double, 5>& dst3);
IUMATH_DLLAPI void splitPlanes(iu::LinearDeviceMemory<double3, 5>& src, iu::LinearDeviceMemory<double, 5>& dst1, iu::LinearDeviceMemory<double, 5>& dst2, iu::LinearDeviceMemory<double, 5>& dst3);


/** Combine planes to a two channel image (e.g. complex image)
 * \param[in] src1 First channel (e.g. real part)
 * \param[in] src2 Second channel (e.g. imaginary part)
 * \param[out] dst Combined image (e.g. complex image)
 *
 */
IUMATH_DLLAPI void combinePlanes(iu::VolumeCpu_32f_C1& src1, iu::VolumeCpu_32f_C1& src2, iu::VolumeCpu_32f_C2& dst);
IUMATH_DLLAPI void combinePlanes(iu::VolumeGpu_32f_C1& src1, iu::VolumeGpu_32f_C1& src2, iu::VolumeGpu_32f_C2& dst);
IUMATH_DLLAPI void combinePlanes(iu::ImageCpu_32f_C1& src1, iu::ImageCpu_32f_C1& src2, iu::ImageCpu_32f_C2& dst);
IUMATH_DLLAPI void combinePlanes(iu::ImageGpu_32f_C1& src1, iu::ImageGpu_32f_C1& src2, iu::ImageGpu_32f_C2& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearHostMemory_32f_C1& src1, iu::LinearHostMemory_32f_C1& src2, iu::LinearHostMemory_32f_C2& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearDeviceMemory_32f_C1& src1, iu::LinearDeviceMemory_32f_C1& src2, iu::LinearDeviceMemory_32f_C2& dst);

IUMATH_DLLAPI void combinePlanes(iu::LinearHostMemory<float, 2>& src1, iu::LinearHostMemory<float, 2>& src2, iu::LinearHostMemory<float2, 2>& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearDeviceMemory<float, 2>& src1, iu::LinearDeviceMemory<float, 2>& src2, iu::LinearDeviceMemory<float2, 2>& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearHostMemory<float, 3>& src1, iu::LinearHostMemory<float, 3>& src2, iu::LinearHostMemory<float2, 3>& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearDeviceMemory<float, 3>& src1, iu::LinearDeviceMemory<float, 3>& src2, iu::LinearDeviceMemory<float2, 3>& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearHostMemory<float, 4>& src1, iu::LinearHostMemory<float, 4>& src2, iu::LinearHostMemory<float2, 4>& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearDeviceMemory<float, 4>& src1, iu::LinearDeviceMemory<float, 4>& src2, iu::LinearDeviceMemory<float2, 4>& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearHostMemory<float, 5>& src1, iu::LinearHostMemory<float, 5>& src2, iu::LinearHostMemory<float2, 5>& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearDeviceMemory<float, 5>& src1, iu::LinearDeviceMemory<float, 5>& src2, iu::LinearDeviceMemory<float2, 5>& dst);

IUMATH_DLLAPI void combinePlanes(iu::VolumeCpu_64f_C1& src1, iu::VolumeCpu_64f_C1& src2, iu::VolumeCpu_64f_C2& dst);
IUMATH_DLLAPI void combinePlanes(iu::VolumeGpu_64f_C1& src1, iu::VolumeGpu_64f_C1& src2, iu::VolumeGpu_64f_C2& dst);
IUMATH_DLLAPI void combinePlanes(iu::ImageCpu_64f_C1& src1, iu::ImageCpu_64f_C1& src2, iu::ImageCpu_64f_C2& dst);
IUMATH_DLLAPI void combinePlanes(iu::ImageGpu_64f_C1& src1, iu::ImageGpu_64f_C1& src2, iu::ImageGpu_64f_C2& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearHostMemory_64f_C1& src1, iu::LinearHostMemory_64f_C1& src2, iu::LinearHostMemory_64f_C2& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearDeviceMemory_64f_C1& src1, iu::LinearDeviceMemory_64f_C1& src2, iu::LinearDeviceMemory_64f_C2& dst);

IUMATH_DLLAPI void combinePlanes(iu::LinearHostMemory<double, 2>& src1, iu::LinearHostMemory<double, 2>& src2, iu::LinearHostMemory<double2, 2>& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearDeviceMemory<double, 2>& src1, iu::LinearDeviceMemory<double, 2>& src2, iu::LinearDeviceMemory<double2, 2>& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearHostMemory<double, 3>& src1, iu::LinearHostMemory<double, 3>& src2, iu::LinearHostMemory<double2, 3>& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearDeviceMemory<double, 3>& src1, iu::LinearDeviceMemory<double, 3>& src2, iu::LinearDeviceMemory<double2, 3>& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearHostMemory<double, 4>& src1, iu::LinearHostMemory<double, 4>& src2, iu::LinearHostMemory<double2, 4>& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearDeviceMemory<double, 4>& src1, iu::LinearDeviceMemory<double, 4>& src2, iu::LinearDeviceMemory<double2, 4>& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearHostMemory<double, 5>& src1, iu::LinearHostMemory<double, 5>& src2, iu::LinearHostMemory<double2, 5>& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearDeviceMemory<double, 5>& src1, iu::LinearDeviceMemory<double, 5>& src2, iu::LinearDeviceMemory<double2, 5>& dst);


/** Combine planes to a three channel image (e.g. rgb image)
 * \param[in] src1 First channel (e.g. r channel)
 * \param[in] src2 Second channel (e.g. g channel)
 * \param[in] src3 Third channel (e.g. b channel)
 * \param[out] dst Combined image (e.g. rgb image)
 *
 */
IUMATH_DLLAPI void combinePlanes(iu::VolumeCpu_32f_C1& src1, iu::VolumeCpu_32f_C1& src2, iu::VolumeCpu_32f_C1& src3, iu::VolumeCpu_32f_C3& dst);
IUMATH_DLLAPI void combinePlanes(iu::VolumeGpu_32f_C1& src1, iu::VolumeGpu_32f_C1& src2, iu::VolumeGpu_32f_C1& src3, iu::VolumeGpu_32f_C3& dst);
IUMATH_DLLAPI void combinePlanes(iu::ImageCpu_32f_C1& src1, iu::ImageCpu_32f_C1& src2, iu::ImageCpu_32f_C1& src3, iu::ImageCpu_32f_C3& dst);
IUMATH_DLLAPI void combinePlanes(iu::ImageGpu_32f_C1& src1, iu::ImageGpu_32f_C1& src2, iu::ImageGpu_32f_C1& src3, iu::ImageGpu_32f_C3& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearHostMemory_32f_C1& src1, iu::LinearHostMemory_32f_C1& src2, iu::LinearHostMemory_32f_C1& src3, iu::LinearHostMemory_32f_C3& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearDeviceMemory_32f_C1& src1, iu::LinearDeviceMemory_32f_C1& src2, iu::LinearDeviceMemory_32f_C1& src3, iu::LinearDeviceMemory_32f_C3& dst);

IUMATH_DLLAPI void combinePlanes(iu::LinearHostMemory<float, 2>& src1, iu::LinearHostMemory<float, 2>& src2, iu::LinearHostMemory<float, 2>& src3, iu::LinearHostMemory<float3, 2>& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearDeviceMemory<float, 2>& src1, iu::LinearDeviceMemory<float, 2>& src2, iu::LinearDeviceMemory<float, 2>& src3, iu::LinearDeviceMemory<float3, 2>& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearHostMemory<float, 3>& src1, iu::LinearHostMemory<float, 3>& src2, iu::LinearHostMemory<float, 3>& src3, iu::LinearHostMemory<float3, 3>& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearDeviceMemory<float, 3>& src1, iu::LinearDeviceMemory<float, 3>& src2, iu::LinearDeviceMemory<float, 3>& src3, iu::LinearDeviceMemory<float3, 3>& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearHostMemory<float, 4>& src1, iu::LinearHostMemory<float, 4>& src2, iu::LinearHostMemory<float, 4>& src3, iu::LinearHostMemory<float3, 4>& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearDeviceMemory<float, 4>& src1, iu::LinearDeviceMemory<float, 4>& src2, iu::LinearDeviceMemory<float, 4>& src3, iu::LinearDeviceMemory<float3, 4>& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearHostMemory<float, 5>& src1, iu::LinearHostMemory<float, 5>& src2, iu::LinearHostMemory<float, 5>& src3, iu::LinearHostMemory<float3, 5>& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearDeviceMemory<float, 5>& src1, iu::LinearDeviceMemory<float, 5>& src2, iu::LinearDeviceMemory<float, 5>& src3, iu::LinearDeviceMemory<float3, 5>& dst);

IUMATH_DLLAPI void combinePlanes(iu::VolumeCpu_64f_C1& src1, iu::VolumeCpu_64f_C1& src2, iu::VolumeCpu_64f_C1& src3, iu::VolumeCpu_64f_C3& dst);
IUMATH_DLLAPI void combinePlanes(iu::VolumeGpu_64f_C1& src1, iu::VolumeGpu_64f_C1& src2, iu::VolumeGpu_64f_C1& src3, iu::VolumeGpu_64f_C3& dst);
IUMATH_DLLAPI void combinePlanes(iu::ImageCpu_64f_C1& src1, iu::ImageCpu_64f_C1& src2, iu::ImageCpu_64f_C1& src3, iu::ImageCpu_64f_C3& dst);
IUMATH_DLLAPI void combinePlanes(iu::ImageGpu_64f_C1& src1, iu::ImageGpu_64f_C1& src2, iu::ImageGpu_64f_C1& src3, iu::ImageGpu_64f_C3& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearHostMemory_64f_C1& src1, iu::LinearHostMemory_64f_C1& src2, iu::LinearHostMemory_64f_C1& src3, iu::LinearHostMemory_64f_C3& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearDeviceMemory_64f_C1& src1, iu::LinearDeviceMemory_64f_C1& src2, iu::LinearDeviceMemory_64f_C1& src3, iu::LinearDeviceMemory_64f_C3& dst);

IUMATH_DLLAPI void combinePlanes(iu::LinearHostMemory<double, 2>& src1, iu::LinearHostMemory<double, 2>& src2, iu::LinearHostMemory<double, 2>& src3, iu::LinearHostMemory<double3, 2>& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearDeviceMemory<double, 2>& src1, iu::LinearDeviceMemory<double, 2>& src2, iu::LinearDeviceMemory<double, 2>& src3, iu::LinearDeviceMemory<double3, 2>& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearHostMemory<double, 3>& src1, iu::LinearHostMemory<double, 3>& src2, iu::LinearHostMemory<double, 3>& src3, iu::LinearHostMemory<double3, 3>& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearDeviceMemory<double, 3>& src1, iu::LinearDeviceMemory<double, 3>& src2, iu::LinearDeviceMemory<double, 3>& src3, iu::LinearDeviceMemory<double3, 3>& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearHostMemory<double, 4>& src1, iu::LinearHostMemory<double, 4>& src2, iu::LinearHostMemory<double, 4>& src3, iu::LinearHostMemory<double3, 4>& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearDeviceMemory<double, 4>& src1, iu::LinearDeviceMemory<double, 4>& src2, iu::LinearDeviceMemory<double, 4>& src3, iu::LinearDeviceMemory<double3, 4>& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearHostMemory<double, 5>& src1, iu::LinearHostMemory<double, 5>& src2, iu::LinearHostMemory<double, 5>& src3, iu::LinearHostMemory<double3, 5>& dst);
IUMATH_DLLAPI void combinePlanes(iu::LinearDeviceMemory<double, 5>& src1, iu::LinearDeviceMemory<double, 5>& src2, iu::LinearDeviceMemory<double, 5>& src3, iu::LinearDeviceMemory<double3, 5>& dst);


/** \} */ // end of MathArithmetics

/** \defgroup MathStatistics Statistics
 \ingroup Math
 \brief Image statistics
 * \{
 */
//---------------------------------------------------------------------------------------------------
// STATISTICS
/** Return minimum and maximum value of an array
 * \param[in] src Source array
 * \param[out] minVal Minimum of src
 * \param[out] maxVal Maximum of src
 */
IUMATH_DLLAPI void minMax(iu::ImageGpu_32f_C1& src, float& minVal, float& maxVal);
IUMATH_DLLAPI void minMax(iu::VolumeGpu_32f_C1& src, float& minVal, float& maxVal);

IUMATH_DLLAPI void minMax(iu::ImageCpu_32f_C1& src, float& minVal, float& maxVal);
IUMATH_DLLAPI void minMax(iu::VolumeCpu_32f_C1& src, float& minVal, float& maxVal);

IUMATH_DLLAPI void minMax(iu::ImageGpu_64f_C1& src, double& minVal, double& maxVal);
IUMATH_DLLAPI void minMax(iu::VolumeGpu_64f_C1& src, double& minVal, double& maxVal);

IUMATH_DLLAPI void minMax(iu::ImageCpu_64f_C1& src, double& minVal, double& maxVal);
IUMATH_DLLAPI void minMax(iu::VolumeCpu_64f_C1& src, double& minVal, double& maxVal);

/** Return minimum and maximum value of an array as well as their positions
 * \param[in] src Source array
 * \param[out] minVal Minimum of src
 * \param[out] maxVal Maximum of src
 * \param[out] minIdx Location of minimum of src
 * \param[out] maxIdx Location of maximum of src
 */
IUMATH_DLLAPI void minMax(iu::LinearDeviceMemory_64f_C1& src, double& minVal, double& maxVal, unsigned int& minIdx, unsigned int& maxIdx);
IUMATH_DLLAPI void minMax(iu::LinearHostMemory_64f_C1& src, double& minVal, double& maxVal, unsigned int& minIdx, unsigned int& maxIdx);

IUMATH_DLLAPI void minMax(iu::LinearDeviceMemory_32f_C1& src, float& minVal, float& maxVal, unsigned int& minIdx, unsigned int& maxIdx);
IUMATH_DLLAPI void minMax(iu::LinearHostMemory_32f_C1& src, float& minVal, float& maxVal, unsigned int& minIdx, unsigned int& maxIdx);
IUMATH_DLLAPI void minMax(iu::LinearDeviceMemory<float, 2>& src, float& minVal, float& maxVal, unsigned int& minIdx, unsigned int& maxIdx);
IUMATH_DLLAPI void minMax(iu::LinearHostMemory<float, 2>& src, float& minVal, float& maxVal, unsigned int& minIdx, unsigned int& maxIdx);
IUMATH_DLLAPI void minMax(iu::LinearDeviceMemory<float, 3>& src, float& minVal, float& maxVal, unsigned int& minIdx, unsigned int& maxIdx);
IUMATH_DLLAPI void minMax(iu::LinearHostMemory<float, 3>& src, float& minVal, float& maxVal, unsigned int& minIdx, unsigned int& maxIdx);
IUMATH_DLLAPI void minMax(iu::LinearDeviceMemory<float, 4>& src, float& minVal, float& maxVal, unsigned int& minIdx, unsigned int& maxIdx);
IUMATH_DLLAPI void minMax(iu::LinearHostMemory<float, 4>& src, float& minVal, float& maxVal, unsigned int& minIdx, unsigned int& maxIdx);
IUMATH_DLLAPI void minMax(iu::LinearDeviceMemory<float, 5>& src, float& minVal, float& maxVal, unsigned int& minIdx, unsigned int& maxIdx);
IUMATH_DLLAPI void minMax(iu::LinearHostMemory<float, 5>& src, float& minVal, float& maxVal, unsigned int& minIdx, unsigned int& maxIdx);

IUMATH_DLLAPI void minMax(iu::LinearDeviceMemory_32f_C1& src, double& minVal, double& maxVal, unsigned int& minIdx, unsigned int& maxIdx);
IUMATH_DLLAPI void minMax(iu::LinearHostMemory_32f_C1& src, double& minVal, double& maxVal, unsigned int& minIdx, unsigned int& maxIdx);
IUMATH_DLLAPI void minMax(iu::LinearDeviceMemory<double, 2>& src, double& minVal, double& maxVal, unsigned int& minIdx, unsigned int& maxIdx);
IUMATH_DLLAPI void minMax(iu::LinearHostMemory<double, 2>& src, double& minVal, double& maxVal, unsigned int& minIdx, unsigned int& maxIdx);
IUMATH_DLLAPI void minMax(iu::LinearDeviceMemory<double, 3>& src, double& minVal, double& maxVal, unsigned int& minIdx, unsigned int& maxIdx);
IUMATH_DLLAPI void minMax(iu::LinearHostMemory<double, 3>& src, double& minVal, double& maxVal, unsigned int& minIdx, unsigned int& maxIdx);
IUMATH_DLLAPI void minMax(iu::LinearDeviceMemory<double, 4>& src, double& minVal, double& maxVal, unsigned int& minIdx, unsigned int& maxIdx);
IUMATH_DLLAPI void minMax(iu::LinearHostMemory<double, 4>& src, double& minVal, double& maxVal, unsigned int& minIdx, unsigned int& maxIdx);
IUMATH_DLLAPI void minMax(iu::LinearDeviceMemory<double, 5>& src, double& minVal, double& maxVal, unsigned int& minIdx, unsigned int& maxIdx);
IUMATH_DLLAPI void minMax(iu::LinearHostMemory<double, 5>& src, double& minVal, double& maxVal, unsigned int& minIdx, unsigned int& maxIdx);

/** Calculate the sum of an array
 * \param[in] src Source array
 * \param[out] sum Resulting sum
 */
IUMATH_DLLAPI void summation(iu::ImageGpu_32f_C1& src, float& sum);
IUMATH_DLLAPI void summation(iu::VolumeGpu_32f_C1& src, float& sum);
IUMATH_DLLAPI void summation(iu::LinearDeviceMemory_32f_C1& src, float& sum);
IUMATH_DLLAPI void summation(iu::LinearDeviceMemory<float, 2>& src, float& sum);
IUMATH_DLLAPI void summation(iu::LinearDeviceMemory<float, 3>& src, float& sum);
IUMATH_DLLAPI void summation(iu::LinearDeviceMemory<float, 4>& src, float& sum);
IUMATH_DLLAPI void summation(iu::LinearDeviceMemory<float, 5>& src, float& sum);

IUMATH_DLLAPI void summation(iu::ImageCpu_32f_C1& src, float& sum);
IUMATH_DLLAPI void summation(iu::VolumeCpu_32f_C1& src, float& sum);
IUMATH_DLLAPI void summation(iu::LinearHostMemory_32f_C1& src, float& sum);
IUMATH_DLLAPI void summation(iu::LinearHostMemory<float, 2>& src, float& sum);
IUMATH_DLLAPI void summation(iu::LinearHostMemory<float, 3>& src, float& sum);
IUMATH_DLLAPI void summation(iu::LinearHostMemory<float, 4>& src, float& sum);
IUMATH_DLLAPI void summation(iu::LinearHostMemory<float, 5>& src, float& sum);

IUMATH_DLLAPI void summation(iu::ImageGpu_64f_C1& src, double& sum);
IUMATH_DLLAPI void summation(iu::VolumeGpu_64f_C1& src, double& sum);
IUMATH_DLLAPI void summation(iu::LinearDeviceMemory_64f_C1& src, double& sum);
IUMATH_DLLAPI void summation(iu::LinearDeviceMemory<double, 2>& src, double& sum);
IUMATH_DLLAPI void summation(iu::LinearDeviceMemory<double, 3>& src, double& sum);
IUMATH_DLLAPI void summation(iu::LinearDeviceMemory<double, 4>& src, double& sum);
IUMATH_DLLAPI void summation(iu::LinearDeviceMemory<double, 5>& src, double& sum);

IUMATH_DLLAPI void summation(iu::ImageCpu_64f_C1& src, double& sum);
IUMATH_DLLAPI void summation(iu::VolumeCpu_64f_C1& src, double& sum);
IUMATH_DLLAPI void summation(iu::LinearHostMemory_64f_C1& src, double& sum);
IUMATH_DLLAPI void summation(iu::LinearHostMemory<double, 2>& src, double& sum);
IUMATH_DLLAPI void summation(iu::LinearHostMemory<double, 3>& src, double& sum);
IUMATH_DLLAPI void summation(iu::LinearHostMemory<double, 4>& src, double& sum);
IUMATH_DLLAPI void summation(iu::LinearHostMemory<double, 5>& src, double& sum);


/** Calculate the L1-Norm \f$ \sum\limits_{i=1}^N \vert x_i - y_i \vert \f$
 *  where \f$ N \f$ is the total number of pixels.
 * \param[in] src Source array \f$ x \f$
 * \param[in] ref Reference array \f$ y \f$
 * \param[out] norm Resulting norm
 */
IUMATH_DLLAPI void normDiffL1(iu::ImageGpu_32f_C1& src, iu::ImageGpu_32f_C1& ref, float& norm);
IUMATH_DLLAPI void normDiffL1(iu::VolumeGpu_32f_C1& src, iu::VolumeGpu_32f_C1& ref, float& norm);

IUMATH_DLLAPI void normDiffL1(iu::ImageCpu_32f_C1& src, iu::ImageCpu_32f_C1& ref, float& norm);
IUMATH_DLLAPI void normDiffL1(iu::VolumeCpu_32f_C1& src, iu::VolumeCpu_32f_C1& ref, float& norm);

IUMATH_DLLAPI void normDiffL1(iu::ImageGpu_64f_C1& src, iu::ImageGpu_64f_C1& ref, double& norm);
IUMATH_DLLAPI void normDiffL1(iu::VolumeGpu_64f_C1& src, iu::VolumeGpu_64f_C1& ref, double& norm);

IUMATH_DLLAPI void normDiffL1(iu::ImageCpu_64f_C1& src, iu::ImageCpu_64f_C1& ref, double& norm);
IUMATH_DLLAPI void normDiffL1(iu::VolumeCpu_64f_C1& src, iu::VolumeCpu_64f_C1& ref, double& norm);

/** Calculate the L1-Norm \f$ \sum\limits_{i=1}^N \vert x_i - y \vert \f$
 *  where \f$ N \f$ is the total number of pixels.
 * \param[in] src Source array \f$ x \f$
 * \param[in] ref Reference value \f$ y \f$
 * \param[out] norm Resulting norm
 */
IUMATH_DLLAPI void normDiffL1(iu::ImageGpu_32f_C1& src, float& ref, float& norm);
IUMATH_DLLAPI void normDiffL1(iu::VolumeGpu_32f_C1& src, float& ref, float& norm);

IUMATH_DLLAPI void normDiffL1(iu::ImageCpu_32f_C1& src, float& ref, float& norm);
IUMATH_DLLAPI void normDiffL1(iu::VolumeCpu_32f_C1& src, float& ref, float& norm);

IUMATH_DLLAPI void normDiffL1(iu::ImageGpu_64f_C1& src, double& ref, double& norm);
IUMATH_DLLAPI void normDiffL1(iu::VolumeGpu_64f_C1& src, double& ref, double& norm);

IUMATH_DLLAPI void normDiffL1(iu::ImageCpu_64f_C1& src, double& ref, double& norm);
IUMATH_DLLAPI void normDiffL1(iu::VolumeCpu_64f_C1& src, double& ref, double& norm);


/** Calculate the L2-Norm \f$ \sqrt{\sum\limits_{i=1}^N ( x_i - y_i )^2}\f$
 *  where \f$ N \f$ is the total number of pixels.
 * \param[in] src Source array \f$ x \f$
 * \param[in] ref Reference array \f$ y \f$
 * \param[out] norm Resulting norm
 */
IUMATH_DLLAPI void normDiffL2(iu::ImageGpu_32f_C1& src, iu::ImageGpu_32f_C1& ref, float& norm);
IUMATH_DLLAPI void normDiffL2(iu::VolumeGpu_32f_C1& src, iu::VolumeGpu_32f_C1& ref, float& norm);

IUMATH_DLLAPI void normDiffL2(iu::ImageCpu_32f_C1& src, iu::ImageCpu_32f_C1& ref, float& norm);
IUMATH_DLLAPI void normDiffL2(iu::VolumeCpu_32f_C1& src, iu::VolumeCpu_32f_C1& ref, float& norm);

IUMATH_DLLAPI void normDiffL2(iu::ImageGpu_64f_C1& src, iu::ImageGpu_64f_C1& ref, double& norm);
IUMATH_DLLAPI void normDiffL2(iu::VolumeGpu_64f_C1& src, iu::VolumeGpu_64f_C1& ref, double& norm);

IUMATH_DLLAPI void normDiffL2(iu::ImageCpu_64f_C1& src, iu::ImageCpu_64f_C1& ref, double& norm);
IUMATH_DLLAPI void normDiffL2(iu::VolumeCpu_64f_C1& src, iu::VolumeCpu_64f_C1& ref, double& norm);

/** Calculate the L2-Norm \f$ \sqrt{\sum\limits_{i=1}^N ( x_i - y )^2}\f$
 *  where \f$ N \f$ is the total number of pixels.
 * \param[in] src Source array \f$ x \f$
 * \param[in] ref Reference value \f$ y \f$
 * \param[out] norm Resulting norm
 */
IUMATH_DLLAPI void normDiffL2(iu::ImageGpu_32f_C1& src, float& ref, float& norm);
IUMATH_DLLAPI void normDiffL2(iu::VolumeGpu_32f_C1& src, float& ref, float& norm);

IUMATH_DLLAPI void normDiffL2(iu::ImageCpu_32f_C1& src, float& ref, float& norm);
IUMATH_DLLAPI void normDiffL2(iu::VolumeCpu_32f_C1& src, float& ref, float& norm);

IUMATH_DLLAPI void normDiffL2(iu::ImageGpu_64f_C1& src, double& ref, double& norm);
IUMATH_DLLAPI void normDiffL2(iu::VolumeGpu_64f_C1& src, double& ref, double& norm);

IUMATH_DLLAPI void normDiffL2(iu::ImageCpu_64f_C1& src, double& ref, double& norm);
IUMATH_DLLAPI void normDiffL2(iu::VolumeCpu_64f_C1& src, double& ref, double& norm);

/** Calculate the mean-squared error (MSE) \f$ \frac{\sum\limits_{i=1}^N ( x_i - y_i )^2}{N}\f$
 *  where \f$ N \f$ is the total number of pixels.
 * \param[in] src Source array \f$ x \f$
 * \param[in] ref Reference array \f$ y \f$
 * \param[out] mse mean-squared error
 */
IUMATH_DLLAPI void mse(iu::ImageGpu_32f_C1& src, iu::ImageGpu_32f_C1& ref, float& mse);
IUMATH_DLLAPI void mse(iu::VolumeGpu_32f_C1& src, iu::VolumeGpu_32f_C1& ref, float& mse);

IUMATH_DLLAPI void mse(iu::ImageCpu_32f_C1& src, iu::ImageCpu_32f_C1& ref, float& mse);
IUMATH_DLLAPI void mse(iu::VolumeCpu_32f_C1& src, iu::VolumeCpu_32f_C1& ref, float& mse);

IUMATH_DLLAPI void mse(iu::ImageGpu_64f_C1& src, iu::ImageGpu_64f_C1& ref, double& mse);
IUMATH_DLLAPI void mse(iu::VolumeGpu_64f_C1& src, iu::VolumeGpu_64f_C1& ref, double& mse);

IUMATH_DLLAPI void mse(iu::ImageCpu_64f_C1& src, iu::ImageCpu_64f_C1& ref, double& mse);
IUMATH_DLLAPI void mse(iu::VolumeCpu_64f_C1& src, iu::VolumeCpu_64f_C1& ref, double& mse);

/** \} */ // end of MathStatistics

//---------------------------------------------------------------------------------------------------
/// Complex math
namespace complex {

/** \defgroup MathComplex Complex
 \ingroup Math
 \brief Complex math operations
 * \{
 */

/** Compute the absolute image of a complex (two channel) image
 * \param[in] complex_img Complex source image
 * \param[out] abs_img Absolute image
 *
 */
IUMATH_DLLAPI void abs(iu::VolumeCpu_32f_C2& complex, iu::VolumeCpu_32f_C1& real);
IUMATH_DLLAPI void abs(iu::VolumeGpu_32f_C2& complex, iu::VolumeGpu_32f_C1& real);
IUMATH_DLLAPI void abs(iu::ImageCpu_32f_C2& complex, iu::ImageCpu_32f_C1& real);
IUMATH_DLLAPI void abs(iu::ImageGpu_32f_C2& complex, iu::ImageGpu_32f_C1& real);
IUMATH_DLLAPI void abs(iu::LinearHostMemory_32f_C2& complex, iu::LinearHostMemory_32f_C1& real);
IUMATH_DLLAPI void abs(iu::LinearDeviceMemory_32f_C2& complex, iu::LinearDeviceMemory_32f_C1& real);
IUMATH_DLLAPI void abs(iu::LinearHostMemory<float2, 2>& complex, iu::LinearHostMemory<float, 2>& real);

IUMATH_DLLAPI void abs(iu::LinearDeviceMemory<float2, 2>& complex, iu::LinearDeviceMemory<float, 2>& real);
IUMATH_DLLAPI void abs(iu::LinearHostMemory<float2, 3>& complex, iu::LinearHostMemory<float, 3>& real);
IUMATH_DLLAPI void abs(iu::LinearDeviceMemory<float2, 3>& complex, iu::LinearDeviceMemory<float, 3>& real);
IUMATH_DLLAPI void abs(iu::LinearHostMemory<float2, 4>& complex, iu::LinearHostMemory<float, 4>& real);
IUMATH_DLLAPI void abs(iu::LinearDeviceMemory<float2, 4>& complex, iu::LinearDeviceMemory<float, 4>& real);
IUMATH_DLLAPI void abs(iu::LinearHostMemory<float2, 5>& complex, iu::LinearHostMemory<float, 5>& real);
IUMATH_DLLAPI void abs(iu::LinearDeviceMemory<float2, 5>& complex, iu::LinearDeviceMemory<float, 5>& real);

IUMATH_DLLAPI void abs(iu::VolumeCpu_64f_C2& complex, iu::VolumeCpu_64f_C1& real);
IUMATH_DLLAPI void abs(iu::VolumeGpu_64f_C2& complex, iu::VolumeGpu_64f_C1& real);
IUMATH_DLLAPI void abs(iu::ImageCpu_64f_C2& complex, iu::ImageCpu_64f_C1& real);
IUMATH_DLLAPI void abs(iu::ImageGpu_64f_C2& complex, iu::ImageGpu_64f_C1& real);
IUMATH_DLLAPI void abs(iu::LinearHostMemory_64f_C2& complex, iu::LinearHostMemory_64f_C1& real);
IUMATH_DLLAPI void abs(iu::LinearDeviceMemory_64f_C2& complex, iu::LinearDeviceMemory_64f_C1& real);

IUMATH_DLLAPI void abs(iu::LinearHostMemory<double2, 2>& complex, iu::LinearHostMemory<double, 2>& real);
IUMATH_DLLAPI void abs(iu::LinearDeviceMemory<double2, 2>& complex, iu::LinearDeviceMemory<double, 2>& real);
IUMATH_DLLAPI void abs(iu::LinearHostMemory<double2, 3>& complex, iu::LinearHostMemory<double, 3>& real);
IUMATH_DLLAPI void abs(iu::LinearDeviceMemory<double2, 3>& complex, iu::LinearDeviceMemory<double, 3>& real);
IUMATH_DLLAPI void abs(iu::LinearHostMemory<double2, 4>& complex, iu::LinearHostMemory<double, 4>& real);
IUMATH_DLLAPI void abs(iu::LinearDeviceMemory<double2, 4>& complex, iu::LinearDeviceMemory<double, 4>& real);
IUMATH_DLLAPI void abs(iu::LinearHostMemory<double2, 5>& complex, iu::LinearHostMemory<double, 5>& real);
IUMATH_DLLAPI void abs(iu::LinearDeviceMemory<double2, 5>& complex, iu::LinearDeviceMemory<double, 5>& real);

/** Compute the real image of a complex (two channel) image
 * \param[in] complex_img Complex source image
 * \param[out] real_img Real image
 *
 */
IUMATH_DLLAPI void real(iu::VolumeCpu_32f_C2& complex, iu::VolumeCpu_32f_C1& real);
IUMATH_DLLAPI void real(iu::VolumeGpu_32f_C2& complex, iu::VolumeGpu_32f_C1& real);
IUMATH_DLLAPI void real(iu::ImageCpu_32f_C2& complex, iu::ImageCpu_32f_C1& real);
IUMATH_DLLAPI void real(iu::ImageGpu_32f_C2& complex, iu::ImageGpu_32f_C1& real);
IUMATH_DLLAPI void real(iu::LinearHostMemory_32f_C2& complex, iu::LinearHostMemory_32f_C1& real);
IUMATH_DLLAPI void real(iu::LinearDeviceMemory_32f_C2& complex, iu::LinearDeviceMemory_32f_C1& real);

IUMATH_DLLAPI void real(iu::LinearHostMemory<float2, 2>& complex, iu::LinearHostMemory<float, 2>& real);
IUMATH_DLLAPI void real(iu::LinearDeviceMemory<float2, 2>& complex, iu::LinearDeviceMemory<float, 2>& real);
IUMATH_DLLAPI void real(iu::LinearHostMemory<float2, 3>& complex, iu::LinearHostMemory<float, 3>& real);
IUMATH_DLLAPI void real(iu::LinearDeviceMemory<float2, 3>& complex, iu::LinearDeviceMemory<float, 3>& real);
IUMATH_DLLAPI void real(iu::LinearHostMemory<float2, 4>& complex, iu::LinearHostMemory<float, 4>& real);
IUMATH_DLLAPI void real(iu::LinearDeviceMemory<float2, 4>& complex, iu::LinearDeviceMemory<float, 4>& real);
IUMATH_DLLAPI void real(iu::LinearHostMemory<float2, 5>& complex, iu::LinearHostMemory<float, 5>& real);
IUMATH_DLLAPI void real(iu::LinearDeviceMemory<float2, 5>& complex, iu::LinearDeviceMemory<float, 5>& real);

IUMATH_DLLAPI void real(iu::VolumeCpu_64f_C2& complex, iu::VolumeCpu_64f_C1& real);
IUMATH_DLLAPI void real(iu::VolumeGpu_64f_C2& complex, iu::VolumeGpu_64f_C1& real);
IUMATH_DLLAPI void real(iu::ImageCpu_64f_C2& complex, iu::ImageCpu_64f_C1& real);
IUMATH_DLLAPI void real(iu::ImageGpu_64f_C2& complex, iu::ImageGpu_64f_C1& real);
IUMATH_DLLAPI void real(iu::LinearHostMemory_64f_C2& complex, iu::LinearHostMemory_64f_C1& real);
IUMATH_DLLAPI void real(iu::LinearDeviceMemory_64f_C2& complex, iu::LinearDeviceMemory_64f_C1& real);

IUMATH_DLLAPI void real(iu::LinearHostMemory<double2, 2>& complex, iu::LinearHostMemory<double, 2>& real);
IUMATH_DLLAPI void real(iu::LinearDeviceMemory<double2, 2>& complex, iu::LinearDeviceMemory<double, 2>& real);
IUMATH_DLLAPI void real(iu::LinearHostMemory<double2, 3>& complex, iu::LinearHostMemory<double, 3>& real);
IUMATH_DLLAPI void real(iu::LinearDeviceMemory<double2, 3>& complex, iu::LinearDeviceMemory<double, 3>& real);
IUMATH_DLLAPI void real(iu::LinearHostMemory<double2, 4>& complex, iu::LinearHostMemory<double, 4>& real);
IUMATH_DLLAPI void real(iu::LinearDeviceMemory<double2, 4>& complex, iu::LinearDeviceMemory<double, 4>& real);
IUMATH_DLLAPI void real(iu::LinearHostMemory<double2, 5>& complex, iu::LinearHostMemory<double, 5>& real);
IUMATH_DLLAPI void real(iu::LinearDeviceMemory<double2, 5>& complex, iu::LinearDeviceMemory<double, 5>& real);

/** Compute the imaginary image of a complex (two channel) image
 * \param[in] complex_img Complex source image
 * \param[out] imag_img Imaginary image
 *
 */
IUMATH_DLLAPI void imag(iu::VolumeCpu_32f_C2& complex, iu::VolumeCpu_32f_C1& real);
IUMATH_DLLAPI void imag(iu::VolumeGpu_32f_C2& complex, iu::VolumeGpu_32f_C1& real);
IUMATH_DLLAPI void imag(iu::ImageCpu_32f_C2& complex, iu::ImageCpu_32f_C1& real);
IUMATH_DLLAPI void imag(iu::ImageGpu_32f_C2& complex, iu::ImageGpu_32f_C1& real);
IUMATH_DLLAPI void imag(iu::LinearHostMemory_32f_C2& complex, iu::LinearHostMemory_32f_C1& real);
IUMATH_DLLAPI void imag(iu::LinearDeviceMemory_32f_C2& complex, iu::LinearDeviceMemory_32f_C1& real);

IUMATH_DLLAPI void imag(iu::LinearHostMemory<float2, 2>& complex, iu::LinearHostMemory<float, 2>& real);
IUMATH_DLLAPI void imag(iu::LinearDeviceMemory<float2, 2>& complex, iu::LinearDeviceMemory<float, 2>& real);
IUMATH_DLLAPI void imag(iu::LinearHostMemory<float2, 3>& complex, iu::LinearHostMemory<float, 3>& real);
IUMATH_DLLAPI void imag(iu::LinearDeviceMemory<float2, 3>& complex, iu::LinearDeviceMemory<float, 3>& real);
IUMATH_DLLAPI void imag(iu::LinearHostMemory<float2, 4>& complex, iu::LinearHostMemory<float, 4>& real);
IUMATH_DLLAPI void imag(iu::LinearDeviceMemory<float2, 4>& complex, iu::LinearDeviceMemory<float, 4>& real);
IUMATH_DLLAPI void imag(iu::LinearHostMemory<float2, 5>& complex, iu::LinearHostMemory<float, 5>& real);
IUMATH_DLLAPI void imag(iu::LinearDeviceMemory<float2, 5>& complex, iu::LinearDeviceMemory<float, 5>& real);

IUMATH_DLLAPI void imag(iu::VolumeCpu_64f_C2& complex, iu::VolumeCpu_64f_C1& real);
IUMATH_DLLAPI void imag(iu::VolumeGpu_64f_C2& complex, iu::VolumeGpu_64f_C1& real);
IUMATH_DLLAPI void imag(iu::ImageCpu_64f_C2& complex, iu::ImageCpu_64f_C1& real);
IUMATH_DLLAPI void imag(iu::ImageGpu_64f_C2& complex, iu::ImageGpu_64f_C1& real);
IUMATH_DLLAPI void imag(iu::LinearHostMemory_64f_C2& complex, iu::LinearHostMemory_64f_C1& real);
IUMATH_DLLAPI void imag(iu::LinearDeviceMemory_64f_C2& complex, iu::LinearDeviceMemory_64f_C1& real);

IUMATH_DLLAPI void imag(iu::LinearHostMemory<double2, 2>& complex, iu::LinearHostMemory<double, 2>& real);
IUMATH_DLLAPI void imag(iu::LinearDeviceMemory<double2, 2>& complex, iu::LinearDeviceMemory<double, 2>& real);
IUMATH_DLLAPI void imag(iu::LinearHostMemory<double2, 3>& complex, iu::LinearHostMemory<double, 3>& real);
IUMATH_DLLAPI void imag(iu::LinearDeviceMemory<double2, 3>& complex, iu::LinearDeviceMemory<double, 3>& real);
IUMATH_DLLAPI void imag(iu::LinearHostMemory<double2, 4>& complex, iu::LinearHostMemory<double, 4>& real);
IUMATH_DLLAPI void imag(iu::LinearDeviceMemory<double2, 4>& complex, iu::LinearDeviceMemory<double, 4>& real);
IUMATH_DLLAPI void imag(iu::LinearHostMemory<double2, 5>& complex, iu::LinearHostMemory<double, 5>& real);
IUMATH_DLLAPI void imag(iu::LinearDeviceMemory<double2, 5>& complex, iu::LinearDeviceMemory<double, 5>& real);

/** Compute the phase of a complex (two channel) image
 * \param[in] complex_img Complex source image
 * \param[out] phase_img Phase image
 *
 */
IUMATH_DLLAPI void phase(iu::VolumeCpu_32f_C2& complex, iu::VolumeCpu_32f_C1& real);
IUMATH_DLLAPI void phase(iu::VolumeGpu_32f_C2& complex, iu::VolumeGpu_32f_C1& real);
IUMATH_DLLAPI void phase(iu::ImageCpu_32f_C2& complex, iu::ImageCpu_32f_C1& real);
IUMATH_DLLAPI void phase(iu::ImageGpu_32f_C2& complex, iu::ImageGpu_32f_C1& real);
IUMATH_DLLAPI void phase(iu::LinearHostMemory_32f_C2& complex, iu::LinearHostMemory_32f_C1& real);
IUMATH_DLLAPI void phase(iu::LinearDeviceMemory_32f_C2& complex, iu::LinearDeviceMemory_32f_C1& real);

IUMATH_DLLAPI void phase(iu::LinearHostMemory<float2, 2>& complex, iu::LinearHostMemory<float, 2>& real);
IUMATH_DLLAPI void phase(iu::LinearDeviceMemory<float2, 2>& complex, iu::LinearDeviceMemory<float, 2>& real);
IUMATH_DLLAPI void phase(iu::LinearHostMemory<float2, 3>& complex, iu::LinearHostMemory<float, 3>& real);
IUMATH_DLLAPI void phase(iu::LinearDeviceMemory<float2, 3>& complex, iu::LinearDeviceMemory<float, 3>& real);
IUMATH_DLLAPI void phase(iu::LinearHostMemory<float2, 4>& complex, iu::LinearHostMemory<float, 4>& real);
IUMATH_DLLAPI void phase(iu::LinearDeviceMemory<float2, 4>& complex, iu::LinearDeviceMemory<float, 4>& real);
IUMATH_DLLAPI void phase(iu::LinearHostMemory<float2, 5>& complex, iu::LinearHostMemory<float, 5>& real);
IUMATH_DLLAPI void phase(iu::LinearDeviceMemory<float2, 5>& complex, iu::LinearDeviceMemory<float, 5>& real);

IUMATH_DLLAPI void phase(iu::VolumeCpu_64f_C2& complex, iu::VolumeCpu_64f_C1& real);
IUMATH_DLLAPI void phase(iu::VolumeGpu_64f_C2& complex, iu::VolumeGpu_64f_C1& real);
IUMATH_DLLAPI void phase(iu::ImageCpu_64f_C2& complex, iu::ImageCpu_64f_C1& real);
IUMATH_DLLAPI void phase(iu::ImageGpu_64f_C2& complex, iu::ImageGpu_64f_C1& real);
IUMATH_DLLAPI void phase(iu::LinearHostMemory_64f_C2& complex, iu::LinearHostMemory_64f_C1& real);
IUMATH_DLLAPI void phase(iu::LinearDeviceMemory_64f_C2& complex, iu::LinearDeviceMemory_64f_C1& real);

IUMATH_DLLAPI void phase(iu::LinearHostMemory<double2, 2>& complex, iu::LinearHostMemory<double, 2>& real);
IUMATH_DLLAPI void phase(iu::LinearDeviceMemory<double2, 2>& complex, iu::LinearDeviceMemory<double, 2>& real);
IUMATH_DLLAPI void phase(iu::LinearHostMemory<double2, 3>& complex, iu::LinearHostMemory<double, 3>& real);
IUMATH_DLLAPI void phase(iu::LinearDeviceMemory<double2, 3>& complex, iu::LinearDeviceMemory<double, 3>& real);
IUMATH_DLLAPI void phase(iu::LinearHostMemory<double2, 4>& complex, iu::LinearHostMemory<double, 4>& real);
IUMATH_DLLAPI void phase(iu::LinearDeviceMemory<double2, 4>& complex, iu::LinearDeviceMemory<double, 4>& real);
IUMATH_DLLAPI void phase(iu::LinearHostMemory<double2, 5>& complex, iu::LinearHostMemory<double, 5>& real);
IUMATH_DLLAPI void phase(iu::LinearDeviceMemory<double2, 5>& complex, iu::LinearDeviceMemory<double, 5>& real);


/** Scale a complex (two channel) image with a scalar
 * \param[in] complex_src Complex source image
 * \param[in] scale Scaling factor
 * \param[out] complex_dst Complex result image
 *
 */
IUMATH_DLLAPI void scale(iu::VolumeCpu_32f_C2& complex_src, const float& scale, iu::VolumeCpu_32f_C2& complex_dst);
IUMATH_DLLAPI void scale(iu::VolumeGpu_32f_C2& complex_src, const float& scale, iu::VolumeGpu_32f_C2& complex_dst);
IUMATH_DLLAPI void scale(iu::ImageCpu_32f_C2& complex_src, const float& scale, iu::ImageCpu_32f_C2& complex_dst);
IUMATH_DLLAPI void scale(iu::ImageGpu_32f_C2& complex_src, const float& scale, iu::ImageGpu_32f_C2& complex_dst);
IUMATH_DLLAPI void scale(iu::LinearHostMemory_32f_C2& complex_src, const float& scale, iu::LinearHostMemory_32f_C2& complex_dst);
IUMATH_DLLAPI void scale(iu::LinearDeviceMemory_32f_C2& complex_src, const float& scale, iu::LinearDeviceMemory_32f_C2& complex_dst);

IUMATH_DLLAPI void scale(iu::LinearHostMemory<float2, 2>& complex_src, const float& scale, iu::LinearHostMemory<float2, 2>& complex_dst);
IUMATH_DLLAPI void scale(iu::LinearDeviceMemory<float2, 2>& complex_src, const float& scale, iu::LinearDeviceMemory<float2, 2>& complex_dst);
IUMATH_DLLAPI void scale(iu::LinearHostMemory<float2, 3>& complex_src, const float& scale, iu::LinearHostMemory<float2, 3>& complex_dst);
IUMATH_DLLAPI void scale(iu::LinearDeviceMemory<float2, 3>& complex_src, const float& scale, iu::LinearDeviceMemory<float2, 3>& complex_dst);
IUMATH_DLLAPI void scale(iu::LinearHostMemory<float2, 4>& complex_src, const float& scale, iu::LinearHostMemory<float2, 4>& complex_dst);
IUMATH_DLLAPI void scale(iu::LinearDeviceMemory<float2, 4>& complex_src, const float& scale, iu::LinearDeviceMemory<float2, 4>& complex_dst);
IUMATH_DLLAPI void scale(iu::LinearHostMemory<float2, 5>& complex_src, const float& scale, iu::LinearHostMemory<float2, 5>& complex_dst);
IUMATH_DLLAPI void scale(iu::LinearDeviceMemory<float2, 5>& complex_src, const float& scale, iu::LinearDeviceMemory<float2, 5>& complex_dst);

IUMATH_DLLAPI void scale(iu::VolumeCpu_64f_C2& complex_src, const double& scale, iu::VolumeCpu_64f_C2& complex_dst);
IUMATH_DLLAPI void scale(iu::VolumeGpu_64f_C2& complex_src, const double& scale, iu::VolumeGpu_64f_C2& complex_dst);
IUMATH_DLLAPI void scale(iu::ImageCpu_64f_C2& complex_src, const double& scale, iu::ImageCpu_64f_C2& complex_dst);
IUMATH_DLLAPI void scale(iu::ImageGpu_64f_C2& complex_src, const double& scale, iu::ImageGpu_64f_C2& complex_dst);
IUMATH_DLLAPI void scale(iu::LinearHostMemory_64f_C2& complex_src, const double& scale, iu::LinearHostMemory_64f_C2& complex_dst);
IUMATH_DLLAPI void scale(iu::LinearDeviceMemory_64f_C2& complex_src, const double& scale, iu::LinearDeviceMemory_64f_C2& complex_dst);

IUMATH_DLLAPI void scale(iu::LinearHostMemory<double2, 2>& complex_src, const double& scale, iu::LinearHostMemory<double2, 2>& complex_dst);
IUMATH_DLLAPI void scale(iu::LinearDeviceMemory<double2, 2>& complex_src, const double& scale, iu::LinearDeviceMemory<double2, 2>& complex_dst);
IUMATH_DLLAPI void scale(iu::LinearHostMemory<double2, 3>& complex_src, const double& scale, iu::LinearHostMemory<double2, 3>& complex_dst);
IUMATH_DLLAPI void scale(iu::LinearDeviceMemory<double2, 3>& complex_src, const double& scale, iu::LinearDeviceMemory<double2, 3>& complex_dst);
IUMATH_DLLAPI void scale(iu::LinearHostMemory<double2, 4>& complex_src, const double& scale, iu::LinearHostMemory<double2, 4>& complex_dst);
IUMATH_DLLAPI void scale(iu::LinearDeviceMemory<double2, 4>& complex_src, const double& scale, iu::LinearDeviceMemory<double2, 4>& complex_dst);
IUMATH_DLLAPI void scale(iu::LinearHostMemory<double2, 5>& complex_src, const double& scale, iu::LinearHostMemory<double2, 5>& complex_dst);
IUMATH_DLLAPI void scale(iu::LinearDeviceMemory<double2, 5>& complex_src, const double& scale, iu::LinearDeviceMemory<double2, 5>& complex_dst);


/** Multiply a complex (two channel) image with a real image
 * \param[in] complex_src First complex source image
 * \param[in] real Real source image
 * \param[out] complex_dst Complex result image
 *
 */
IUMATH_DLLAPI void multiply(iu::VolumeCpu_32f_C2& complex_src, iu::VolumeCpu_32f_C1& real, iu::VolumeCpu_32f_C2& complex_dst);
IUMATH_DLLAPI void multiply(iu::VolumeGpu_32f_C2& complex_src, iu::VolumeGpu_32f_C1& real, iu::VolumeGpu_32f_C2& complex_dst);
IUMATH_DLLAPI void multiply(iu::ImageCpu_32f_C2& complex_src, iu::ImageCpu_32f_C1& real, iu::ImageCpu_32f_C2& complex_dst);
IUMATH_DLLAPI void multiply(iu::ImageGpu_32f_C2& complex_src, iu::ImageGpu_32f_C1& real, iu::ImageGpu_32f_C2& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearHostMemory_32f_C2& complex_src, iu::LinearHostMemory_32f_C1& real, iu::LinearHostMemory_32f_C2& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearDeviceMemory_32f_C2& complex_src, iu::LinearDeviceMemory_32f_C1& real, iu::LinearDeviceMemory_32f_C2& complex_dst);

IUMATH_DLLAPI void multiply(iu::LinearHostMemory<float2, 2>& complex_src, iu::LinearHostMemory<float, 2>& real, iu::LinearHostMemory<float2, 2>& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearDeviceMemory<float2, 2>& complex_src, iu::LinearDeviceMemory<float, 2>& real, iu::LinearDeviceMemory<float2, 2>& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearHostMemory<float2, 3>& complex_src, iu::LinearHostMemory<float, 3>& real, iu::LinearHostMemory<float2, 3>& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearDeviceMemory<float2, 3>& complex_src, iu::LinearDeviceMemory<float, 3>& real, iu::LinearDeviceMemory<float2, 3>& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearHostMemory<float2, 4>& complex_src, iu::LinearHostMemory<float, 4>& real, iu::LinearHostMemory<float2, 4>& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearDeviceMemory<float2, 4>& complex_src, iu::LinearDeviceMemory<float, 4>& real, iu::LinearDeviceMemory<float2, 4>& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearHostMemory<float2, 5>& complex_src, iu::LinearHostMemory<float, 5>& real, iu::LinearHostMemory<float2, 5>& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearDeviceMemory<float2, 5>& complex_src, iu::LinearDeviceMemory<float, 5>& real, iu::LinearDeviceMemory<float2, 5>& complex_dst);

IUMATH_DLLAPI void multiply(iu::VolumeCpu_64f_C2& complex_src, iu::VolumeCpu_64f_C1& real, iu::VolumeCpu_64f_C2& complex_dst);
IUMATH_DLLAPI void multiply(iu::VolumeGpu_64f_C2& complex_src, iu::VolumeGpu_64f_C1& real, iu::VolumeGpu_64f_C2& complex_dst);
IUMATH_DLLAPI void multiply(iu::ImageCpu_64f_C2& complex_src, iu::ImageCpu_64f_C1& real, iu::ImageCpu_64f_C2& complex_dst);
IUMATH_DLLAPI void multiply(iu::ImageGpu_64f_C2& complex_src, iu::ImageGpu_64f_C1& real, iu::ImageGpu_64f_C2& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearHostMemory_64f_C2& complex_src, iu::LinearHostMemory_64f_C1& real, iu::LinearHostMemory_64f_C2& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearDeviceMemory_64f_C2& complex_src, iu::LinearDeviceMemory_64f_C1& real, iu::LinearDeviceMemory_64f_C2& complex_dst);

IUMATH_DLLAPI void multiply(iu::LinearHostMemory<double2, 2>& complex_src, iu::LinearHostMemory<double, 2>& real, iu::LinearHostMemory<double2, 2>& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearDeviceMemory<double2, 2>& complex_src, iu::LinearDeviceMemory<double, 2>& real, iu::LinearDeviceMemory<double2, 2>& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearHostMemory<double2, 3>& complex_src, iu::LinearHostMemory<double, 3>& real, iu::LinearHostMemory<double2, 3>& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearDeviceMemory<double2, 3>& complex_src, iu::LinearDeviceMemory<double, 3>& real, iu::LinearDeviceMemory<double2, 3>& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearHostMemory<double2, 4>& complex_src, iu::LinearHostMemory<double, 4>& real, iu::LinearHostMemory<double2, 4>& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearDeviceMemory<double2, 4>& complex_src, iu::LinearDeviceMemory<double, 4>& real, iu::LinearDeviceMemory<double2, 4>& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearHostMemory<double2, 5>& complex_src, iu::LinearHostMemory<double, 5>& real, iu::LinearHostMemory<double2, 5>& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearDeviceMemory<double2, 5>& complex_src, iu::LinearDeviceMemory<double, 5>& real, iu::LinearDeviceMemory<double2, 5>& complex_dst);

/** Multiply two complex (two channel) images
 * \param[in] complex_src1 First complex source image
 * \param[in] complex_src2 Second complex source image
 * \param[out] complex_dst Complex result image
 *
 */
IUMATH_DLLAPI void multiply(iu::VolumeCpu_32f_C2& complex_src1, iu::VolumeCpu_32f_C2& complex_src2, iu::VolumeCpu_32f_C2& complex_dst);
IUMATH_DLLAPI void multiply(iu::VolumeGpu_32f_C2& complex_src1, iu::VolumeGpu_32f_C2& complex_src2, iu::VolumeGpu_32f_C2& complex_dst);
IUMATH_DLLAPI void multiply(iu::ImageCpu_32f_C2& complex_src1, iu::ImageCpu_32f_C2& complex_src2, iu::ImageCpu_32f_C2& complex_dst);
IUMATH_DLLAPI void multiply(iu::ImageGpu_32f_C2& complex_src1, iu::ImageGpu_32f_C2& complex_src2, iu::ImageGpu_32f_C2& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearHostMemory_32f_C2& complex_src1, iu::LinearHostMemory_32f_C2& complex_src2, iu::LinearHostMemory_32f_C2& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearDeviceMemory_32f_C2& complex_src1, iu::LinearDeviceMemory_32f_C2& complex_src2, iu::LinearDeviceMemory_32f_C2& complex_dst);

IUMATH_DLLAPI void multiply(iu::LinearHostMemory<float2, 2>& complex_src, iu::LinearHostMemory<float2, 2>& complex_src2, iu::LinearHostMemory<float2, 2>& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearDeviceMemory<float2, 2>& complex_src, iu::LinearDeviceMemory<float2, 2>& complex_src2, iu::LinearDeviceMemory<float2, 2>& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearHostMemory<float2, 3>& complex_src, iu::LinearHostMemory<float2, 3>& complex_src2, iu::LinearHostMemory<float2, 3>& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearDeviceMemory<float2, 3>& complex_src, iu::LinearDeviceMemory<float2, 3>& complex_src2, iu::LinearDeviceMemory<float2, 3>& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearHostMemory<float2, 4>& complex_src, iu::LinearHostMemory<float2, 4>& complex_src2, iu::LinearHostMemory<float2, 4>& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearDeviceMemory<float2, 4>& complex_src, iu::LinearDeviceMemory<float2, 4>& complex_src2, iu::LinearDeviceMemory<float2, 4>& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearHostMemory<float2, 5>& complex_src, iu::LinearHostMemory<float2, 5>& complex_src2, iu::LinearHostMemory<float2, 5>& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearDeviceMemory<float2, 5>& complex_src, iu::LinearDeviceMemory<float2, 5>& complex_src2, iu::LinearDeviceMemory<float2, 5>& complex_dst);

IUMATH_DLLAPI void multiply(iu::VolumeCpu_64f_C2& complex_src1, iu::VolumeCpu_64f_C2& complex_src2, iu::VolumeCpu_64f_C2& complex_dst);
IUMATH_DLLAPI void multiply(iu::VolumeGpu_64f_C2& complex_src1, iu::VolumeGpu_64f_C2& complex_src2, iu::VolumeGpu_64f_C2& complex_dst);
IUMATH_DLLAPI void multiply(iu::ImageCpu_64f_C2& complex_src1, iu::ImageCpu_64f_C2& complex_src2, iu::ImageCpu_64f_C2& complex_dst);
IUMATH_DLLAPI void multiply(iu::ImageGpu_64f_C2& complex_src1, iu::ImageGpu_64f_C2& complex_src2, iu::ImageGpu_64f_C2& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearHostMemory_64f_C2& complex_src1, iu::LinearHostMemory_64f_C2& complex_src2, iu::LinearHostMemory_64f_C2& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearDeviceMemory_64f_C2& complex_src1, iu::LinearDeviceMemory_64f_C2& complex_src2, iu::LinearDeviceMemory_64f_C2& complex_dst);

IUMATH_DLLAPI void multiply(iu::LinearHostMemory<double2, 2>& complex_src, iu::LinearHostMemory<double2, 2>& complex_src2, iu::LinearHostMemory<double2, 2>& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearDeviceMemory<double2, 2>& complex_src, iu::LinearDeviceMemory<double2, 2>& complex_src2, iu::LinearDeviceMemory<double2, 2>& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearHostMemory<double2, 3>& complex_src, iu::LinearHostMemory<double2, 3>& complex_src2, iu::LinearHostMemory<double2, 3>& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearDeviceMemory<double2, 3>& complex_src, iu::LinearDeviceMemory<double2, 3>& complex_src2, iu::LinearDeviceMemory<double2, 3>& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearHostMemory<double2, 4>& complex_src, iu::LinearHostMemory<double2, 4>& complex_src2, iu::LinearHostMemory<double2, 4>& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearDeviceMemory<double2, 4>& complex_src, iu::LinearDeviceMemory<double2, 4>& complex_src2, iu::LinearDeviceMemory<double2, 4>& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearHostMemory<double2, 5>& complex_src, iu::LinearHostMemory<double2, 5>& complex_src2, iu::LinearHostMemory<double2, 5>& complex_dst);
IUMATH_DLLAPI void multiply(iu::LinearDeviceMemory<double2, 5>& complex_src, iu::LinearDeviceMemory<double2, 5>& complex_src2, iu::LinearDeviceMemory<double2, 5>& complex_dst);

/** Multiply one complex (two channel) image with the complex conjugate of a second complex image
 * \param[in] complex_src1 First complex source image
 * \param[in] complex_src2 Second complex source image
 * \param[out] complex_dst Complex result image
 *
 */
IUMATH_DLLAPI void multiplyConjugate(iu::VolumeCpu_32f_C2& complex_src1, iu::VolumeCpu_32f_C2& complex_src2, iu::VolumeCpu_32f_C2& complex_dst);
IUMATH_DLLAPI void multiplyConjugate(iu::VolumeGpu_32f_C2& complex_src1, iu::VolumeGpu_32f_C2& complex_src2, iu::VolumeGpu_32f_C2& complex_dst);
IUMATH_DLLAPI void multiplyConjugate(iu::ImageCpu_32f_C2& complex_src1, iu::ImageCpu_32f_C2& complex_src2, iu::ImageCpu_32f_C2& complex_dst);
IUMATH_DLLAPI void multiplyConjugate(iu::ImageGpu_32f_C2& complex_src1, iu::ImageGpu_32f_C2& complex_src2, iu::ImageGpu_32f_C2& complex_dst);
IUMATH_DLLAPI void multiplyConjugate(iu::LinearHostMemory_32f_C2& complex_src1, iu::LinearHostMemory_32f_C2& complex_src2, iu::LinearHostMemory_32f_C2& complex_dst);
IUMATH_DLLAPI void multiplyConjugate(iu::LinearDeviceMemory_32f_C2& complex_src1, iu::LinearDeviceMemory_32f_C2& complex_src2, iu::LinearDeviceMemory_32f_C2& complex_dst);

IUMATH_DLLAPI void multiplyConjugate(iu::LinearHostMemory<float2, 2>& complex_src, iu::LinearHostMemory<float2, 2>& complex_src2, iu::LinearHostMemory<float2, 2>& complex_dst);
IUMATH_DLLAPI void multiplyConjugate(iu::LinearDeviceMemory<float2, 2>& complex_src, iu::LinearDeviceMemory<float2, 2>& complex_src2, iu::LinearDeviceMemory<float2, 2>& complex_dst);
IUMATH_DLLAPI void multiplyConjugate(iu::LinearHostMemory<float2, 3>& complex_src, iu::LinearHostMemory<float2, 3>& complex_src2, iu::LinearHostMemory<float2, 3>& complex_dst);
IUMATH_DLLAPI void multiplyConjugate(iu::LinearDeviceMemory<float2, 3>& complex_src, iu::LinearDeviceMemory<float2, 3>& complex_src2, iu::LinearDeviceMemory<float2, 3>& complex_dst);
IUMATH_DLLAPI void multiplyConjugate(iu::LinearHostMemory<float2, 4>& complex_src, iu::LinearHostMemory<float2, 4>& complex_src2, iu::LinearHostMemory<float2, 4>& complex_dst);
IUMATH_DLLAPI void multiplyConjugate(iu::LinearDeviceMemory<float2, 4>& complex_src, iu::LinearDeviceMemory<float2, 4>& complex_src2, iu::LinearDeviceMemory<float2, 4>& complex_dst);
IUMATH_DLLAPI void multiplyConjugate(iu::LinearHostMemory<float2, 5>& complex_src, iu::LinearHostMemory<float2, 5>& complex_src2, iu::LinearHostMemory<float2, 5>& complex_dst);
IUMATH_DLLAPI void multiplyConjugate(iu::LinearDeviceMemory<float2, 5>& complex_src, iu::LinearDeviceMemory<float2, 5>& complex_src2, iu::LinearDeviceMemory<float2, 5>& complex_dst);

IUMATH_DLLAPI void multiplyConjugate(iu::VolumeCpu_64f_C2& complex_src1, iu::VolumeCpu_64f_C2& complex_src2, iu::VolumeCpu_64f_C2& complex_dst);
IUMATH_DLLAPI void multiplyConjugate(iu::VolumeGpu_64f_C2& complex_src1, iu::VolumeGpu_64f_C2& complex_src2, iu::VolumeGpu_64f_C2& complex_dst);
IUMATH_DLLAPI void multiplyConjugate(iu::ImageCpu_64f_C2& complex_src1, iu::ImageCpu_64f_C2& complex_src2, iu::ImageCpu_64f_C2& complex_dst);
IUMATH_DLLAPI void multiplyConjugate(iu::ImageGpu_64f_C2& complex_src1, iu::ImageGpu_64f_C2& complex_src2, iu::ImageGpu_64f_C2& complex_dst);
IUMATH_DLLAPI void multiplyConjugate(iu::LinearHostMemory_64f_C2& complex_src1, iu::LinearHostMemory_64f_C2& complex_src2, iu::LinearHostMemory_64f_C2& complex_dst);
IUMATH_DLLAPI void multiplyConjugate(iu::LinearDeviceMemory_64f_C2& complex_src1, iu::LinearDeviceMemory_64f_C2& complex_src2, iu::LinearDeviceMemory_64f_C2& complex_dst);

IUMATH_DLLAPI void multiplyConjugate(iu::LinearHostMemory<double2, 2>& complex_src, iu::LinearHostMemory<double2, 2>& complex_src2, iu::LinearHostMemory<double2, 2>& complex_dst);
IUMATH_DLLAPI void multiplyConjugate(iu::LinearDeviceMemory<double2, 2>& complex_src, iu::LinearDeviceMemory<double2, 2>& complex_src2, iu::LinearDeviceMemory<double2, 2>& complex_dst);
IUMATH_DLLAPI void multiplyConjugate(iu::LinearHostMemory<double2, 3>& complex_src, iu::LinearHostMemory<double2, 3>& complex_src2, iu::LinearHostMemory<double2, 3>& complex_dst);
IUMATH_DLLAPI void multiplyConjugate(iu::LinearDeviceMemory<double2, 3>& complex_src, iu::LinearDeviceMemory<double2, 3>& complex_src2, iu::LinearDeviceMemory<double2, 3>& complex_dst);
IUMATH_DLLAPI void multiplyConjugate(iu::LinearHostMemory<double2, 4>& complex_src, iu::LinearHostMemory<double2, 4>& complex_src2, iu::LinearHostMemory<double2, 4>& complex_dst);
IUMATH_DLLAPI void multiplyConjugate(iu::LinearDeviceMemory<double2, 4>& complex_src, iu::LinearDeviceMemory<double2, 4>& complex_src2, iu::LinearDeviceMemory<double2, 4>& complex_dst);
IUMATH_DLLAPI void multiplyConjugate(iu::LinearHostMemory<double2, 5>& complex_src, iu::LinearHostMemory<double2, 5>& complex_src2, iu::LinearHostMemory<double2, 5>& complex_dst);
IUMATH_DLLAPI void multiplyConjugate(iu::LinearDeviceMemory<double2, 5>& complex_src, iu::LinearDeviceMemory<double2, 5>& complex_src2, iu::LinearDeviceMemory<double2, 5>& complex_dst);

/** \} */ // end of MathComplex

} // namespace complex

//---------------------------------------------------------------------------------------------------
/// FFT
namespace fft {

/** \defgroup MathFFT FFT
 \ingroup Math
 \brief Fourier transforms
 * \{
 */

/** Compute the fftshift2
 * \param[in] src source
 * \param[out] dst destination
 */
IUMATH_DLLAPI void fftshift2(const iu::ImageGpu_32f_C1& src, iu::ImageGpu_32f_C1& dst);
IUMATH_DLLAPI void fftshift2(const iu::ImageGpu_32f_C2& src, iu::ImageGpu_32f_C2& dst);
IUMATH_DLLAPI void fftshift2(const iu::VolumeGpu_32f_C1& src, iu::VolumeGpu_32f_C1& dst);
IUMATH_DLLAPI void fftshift2(const iu::VolumeGpu_32f_C2& src, iu::VolumeGpu_32f_C2& dst);
IUMATH_DLLAPI void fftshift2(const iu::LinearDeviceMemory<float, 2>& src, iu::LinearDeviceMemory<float, 2>& dst);
IUMATH_DLLAPI void fftshift2(const iu::LinearDeviceMemory<float2, 2>& src, iu::LinearDeviceMemory<float2, 2>& dst);
IUMATH_DLLAPI void fftshift2(const iu::LinearDeviceMemory<float, 3>& src, iu::LinearDeviceMemory<float, 3>& dst);
IUMATH_DLLAPI void fftshift2(const iu::LinearDeviceMemory<float2, 3>& src, iu::LinearDeviceMemory<float2, 3>& dst);
IUMATH_DLLAPI void fftshift2(const iu::LinearDeviceMemory<float, 4>& src, iu::LinearDeviceMemory<float, 4>& dst);
IUMATH_DLLAPI void fftshift2(const iu::LinearDeviceMemory<float2, 4>& src, iu::LinearDeviceMemory<float2, 4>& dst);

IUMATH_DLLAPI void fftshift2(const iu::ImageGpu_64f_C1& src, iu::ImageGpu_64f_C1& dst);
IUMATH_DLLAPI void fftshift2(const iu::ImageGpu_64f_C2& src, iu::ImageGpu_64f_C2& dst);
IUMATH_DLLAPI void fftshift2(const iu::VolumeGpu_64f_C1& src, iu::VolumeGpu_64f_C1& dst);
IUMATH_DLLAPI void fftshift2(const iu::VolumeGpu_64f_C2& src, iu::VolumeGpu_64f_C2& dst);
IUMATH_DLLAPI void fftshift2(const iu::LinearDeviceMemory<double, 2>& src, iu::LinearDeviceMemory<double, 2>& dst);
IUMATH_DLLAPI void fftshift2(const iu::LinearDeviceMemory<double2, 2>& src, iu::LinearDeviceMemory<double2, 2>& dst);
IUMATH_DLLAPI void fftshift2(const iu::LinearDeviceMemory<double, 3>& src, iu::LinearDeviceMemory<double, 3>& dst);
IUMATH_DLLAPI void fftshift2(const iu::LinearDeviceMemory<double2, 3>& src, iu::LinearDeviceMemory<double2, 3>& dst);
IUMATH_DLLAPI void fftshift2(const iu::LinearDeviceMemory<double, 4>& src, iu::LinearDeviceMemory<double, 4>& dst);
IUMATH_DLLAPI void fftshift2(const iu::LinearDeviceMemory<double2, 4>& src, iu::LinearDeviceMemory<double2, 4>& dst);

/** Compute the ifftshift2
 * \param[in] src source
 * \param[out] dst destination
 */
IUMATH_DLLAPI void ifftshift2(const iu::ImageGpu_32f_C1& src, iu::ImageGpu_32f_C1& dst);
IUMATH_DLLAPI void ifftshift2(const iu::ImageGpu_32f_C2& src, iu::ImageGpu_32f_C2& dst);
IUMATH_DLLAPI void ifftshift2(const iu::VolumeGpu_32f_C1& src, iu::VolumeGpu_32f_C1& dst);
IUMATH_DLLAPI void ifftshift2(const iu::VolumeGpu_32f_C2& src, iu::VolumeGpu_32f_C2& dst);
IUMATH_DLLAPI void ifftshift2(const iu::LinearDeviceMemory<float, 2>& src, iu::LinearDeviceMemory<float, 2>& dst);
IUMATH_DLLAPI void ifftshift2(const iu::LinearDeviceMemory<float2, 2>& src, iu::LinearDeviceMemory<float2, 2>& dst);
IUMATH_DLLAPI void ifftshift2(const iu::LinearDeviceMemory<float, 3>& src, iu::LinearDeviceMemory<float, 3>& dst);
IUMATH_DLLAPI void ifftshift2(const iu::LinearDeviceMemory<float2, 3>& src, iu::LinearDeviceMemory<float2, 3>& dst);
IUMATH_DLLAPI void ifftshift2(const iu::LinearDeviceMemory<float, 4>& src, iu::LinearDeviceMemory<float, 4>& dst);
IUMATH_DLLAPI void ifftshift2(const iu::LinearDeviceMemory<float2, 4>& src, iu::LinearDeviceMemory<float2, 4>& dst);

IUMATH_DLLAPI void ifftshift2(const iu::ImageGpu_64f_C1& src, iu::ImageGpu_64f_C1& dst);
IUMATH_DLLAPI void ifftshift2(const iu::ImageGpu_64f_C2& src, iu::ImageGpu_64f_C2& dst);
IUMATH_DLLAPI void ifftshift2(const iu::VolumeGpu_64f_C1& src, iu::VolumeGpu_64f_C1& dst);
IUMATH_DLLAPI void ifftshift2(const iu::VolumeGpu_64f_C2& src, iu::VolumeGpu_64f_C2& dst);
IUMATH_DLLAPI void ifftshift2(const iu::LinearDeviceMemory<double, 2>& src, iu::LinearDeviceMemory<double, 2>& dst);
IUMATH_DLLAPI void ifftshift2(const iu::LinearDeviceMemory<double2, 2>& src, iu::LinearDeviceMemory<double2, 2>& dst);
IUMATH_DLLAPI void ifftshift2(const iu::LinearDeviceMemory<double, 3>& src, iu::LinearDeviceMemory<double, 3>& dst);
IUMATH_DLLAPI void ifftshift2(const iu::LinearDeviceMemory<double2, 3>& src, iu::LinearDeviceMemory<double2, 3>& dst);
IUMATH_DLLAPI void ifftshift2(const iu::LinearDeviceMemory<double, 4>& src, iu::LinearDeviceMemory<double, 4>& dst);
IUMATH_DLLAPI void ifftshift2(const iu::LinearDeviceMemory<double2, 4>& src, iu::LinearDeviceMemory<double2, 4>& dst);

/** Compute the (batched) fft2
 * \param[in] src source
 * \param[out] dst destination
 */
IUMATH_DLLAPI void fft2(const iu::ImageGpu_32f_C2& src, iu::ImageGpu_32f_C2& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void fft2(const iu::VolumeGpu_32f_C2& src, iu::VolumeGpu_32f_C2& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void fft2(const iu::ImageGpu_32f_C1& src, iu::ImageGpu_32f_C2& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void fft2(const iu::VolumeGpu_32f_C1& src, iu::VolumeGpu_32f_C2& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void fft2(const iu::LinearDeviceMemory<float2, 2>& src, iu::LinearDeviceMemory<float2, 2>& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void fft2(const iu::LinearDeviceMemory<float2, 3>& src, iu::LinearDeviceMemory<float2, 3>& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void fft2(const iu::LinearDeviceMemory<float2, 4>& src, iu::LinearDeviceMemory<float2, 4>& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void fft2(const iu::LinearDeviceMemory<float, 2>& src, iu::LinearDeviceMemory<float2, 2>& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void fft2(const iu::LinearDeviceMemory<float, 3>& src, iu::LinearDeviceMemory<float2, 3>& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void fft2(const iu::LinearDeviceMemory<float, 4>& src, iu::LinearDeviceMemory<float2, 4>& dst, bool scale_sqrt=false);

IUMATH_DLLAPI void fft2(const iu::ImageGpu_64f_C2& src, iu::ImageGpu_64f_C2& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void fft2(const iu::VolumeGpu_64f_C2& src, iu::VolumeGpu_64f_C2& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void fft2(const iu::ImageGpu_64f_C1& src, iu::ImageGpu_64f_C2& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void fft2(const iu::VolumeGpu_64f_C1& src, iu::VolumeGpu_64f_C2& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void fft2(const iu::LinearDeviceMemory<double2, 2>& src, iu::LinearDeviceMemory<double2, 2>& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void fft2(const iu::LinearDeviceMemory<double2, 3>& src, iu::LinearDeviceMemory<double2, 3>& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void fft2(const iu::LinearDeviceMemory<double2, 4>& src, iu::LinearDeviceMemory<double2, 4>& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void fft2(const iu::LinearDeviceMemory<double, 2>& src, iu::LinearDeviceMemory<double2, 2>& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void fft2(const iu::LinearDeviceMemory<double, 3>& src, iu::LinearDeviceMemory<double2, 3>& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void fft2(const iu::LinearDeviceMemory<double, 4>& src, iu::LinearDeviceMemory<double2, 4>& dst, bool scale_sqrt=false);

/** Compute the (batched) ifft2
 * \param[in] src source
 * \param[out] dst destination
 */
IUMATH_DLLAPI void ifft2(const iu::ImageGpu_32f_C2& src, iu::ImageGpu_32f_C2& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void ifft2(const iu::VolumeGpu_32f_C2& src, iu::VolumeGpu_32f_C2& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void ifft2(const iu::ImageGpu_32f_C2& src, iu::ImageGpu_32f_C1& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void ifft2(const iu::VolumeGpu_32f_C2& src, iu::VolumeGpu_32f_C1& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void ifft2(const iu::LinearDeviceMemory<float2, 2>& src, iu::LinearDeviceMemory<float2, 2>& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void ifft2(const iu::LinearDeviceMemory<float2, 3>& src, iu::LinearDeviceMemory<float2, 3>& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void ifft2(const iu::LinearDeviceMemory<float2, 4>& src, iu::LinearDeviceMemory<float2, 4>& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void ifft2(const iu::LinearDeviceMemory<float2, 2>& src, iu::LinearDeviceMemory<float, 2>& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void ifft2(const iu::LinearDeviceMemory<float2, 3>& src, iu::LinearDeviceMemory<float, 3>& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void ifft2(const iu::LinearDeviceMemory<float2, 4>& src, iu::LinearDeviceMemory<float, 4>& dst, bool scale_sqrt=false);

IUMATH_DLLAPI void ifft2(const iu::ImageGpu_64f_C2& src, iu::ImageGpu_64f_C2& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void ifft2(const iu::VolumeGpu_64f_C2& src, iu::VolumeGpu_64f_C2& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void ifft2(const iu::ImageGpu_64f_C2& src, iu::ImageGpu_64f_C1& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void ifft2(const iu::VolumeGpu_64f_C2& src, iu::VolumeGpu_64f_C1& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void ifft2(const iu::LinearDeviceMemory<double2, 2>& src, iu::LinearDeviceMemory<double2, 2>& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void ifft2(const iu::LinearDeviceMemory<double2, 3>& src, iu::LinearDeviceMemory<double2, 3>& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void ifft2(const iu::LinearDeviceMemory<double2, 4>& src, iu::LinearDeviceMemory<double2, 4>& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void ifft2(const iu::LinearDeviceMemory<double2, 2>& src, iu::LinearDeviceMemory<double, 2>& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void ifft2(const iu::LinearDeviceMemory<double2, 3>& src, iu::LinearDeviceMemory<double, 3>& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void ifft2(const iu::LinearDeviceMemory<double2, 4>& src, iu::LinearDeviceMemory<double, 4>& dst, bool scale_sqrt=false);

/** Compute the (batched) centered fft2
 * \param[in] src source
 * \param[out] dst destination
 */
IUMATH_DLLAPI void fft2c(const iu::ImageGpu_32f_C2& src, iu::ImageGpu_32f_C2& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void fft2c(const iu::VolumeGpu_32f_C2& src, iu::VolumeGpu_32f_C2& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void fft2c(const iu::LinearDeviceMemory<float2, 2>& src, iu::LinearDeviceMemory<float2, 2>& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void fft2c(const iu::LinearDeviceMemory<float2, 3>& src, iu::LinearDeviceMemory<float2, 3>& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void fft2c(const iu::LinearDeviceMemory<float2, 4>& src, iu::LinearDeviceMemory<float2, 4>& dst, bool scale_sqrt=false);

IUMATH_DLLAPI void fft2c(const iu::ImageGpu_64f_C2& src, iu::ImageGpu_64f_C2& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void fft2c(const iu::VolumeGpu_64f_C2& src, iu::VolumeGpu_64f_C2& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void fft2c(const iu::LinearDeviceMemory<double2, 2>& src, iu::LinearDeviceMemory<double2, 2>& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void fft2c(const iu::LinearDeviceMemory<double2, 3>& src, iu::LinearDeviceMemory<double2, 3>& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void fft2c(const iu::LinearDeviceMemory<double2, 4>& src, iu::LinearDeviceMemory<double2, 4>& dst, bool scale_sqrt=false);

/** Compute the (batched) centered ifft2
 * \param[in] src source
 * \param[out] dst destination
 */
IUMATH_DLLAPI void ifft2c(const iu::ImageGpu_32f_C2& src, iu::ImageGpu_32f_C2& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void ifft2c(const iu::VolumeGpu_32f_C2& src, iu::VolumeGpu_32f_C2& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void ifft2c(const iu::LinearDeviceMemory<float2, 2>& src, iu::LinearDeviceMemory<float2, 2>& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void ifft2c(const iu::LinearDeviceMemory<float2, 3>& src, iu::LinearDeviceMemory<float2, 3>& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void ifft2c(const iu::LinearDeviceMemory<float2, 4>& src, iu::LinearDeviceMemory<float2, 4>& dst, bool scale_sqrt=false);

IUMATH_DLLAPI void ifft2c(const iu::ImageGpu_64f_C2& src, iu::ImageGpu_64f_C2& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void ifft2c(const iu::VolumeGpu_64f_C2& src, iu::VolumeGpu_64f_C2& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void ifft2c(const iu::LinearDeviceMemory<double2, 2>& src, iu::LinearDeviceMemory<double2, 2>& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void ifft2c(const iu::LinearDeviceMemory<double2, 3>& src, iu::LinearDeviceMemory<double2, 3>& dst, bool scale_sqrt=false);
IUMATH_DLLAPI void ifft2c(const iu::LinearDeviceMemory<double2, 4>& src, iu::LinearDeviceMemory<double2, 4>& dst, bool scale_sqrt=false);

/** \} */ // end of MathFFT

} // namespace fft

} // namespace math
} // namespace iu

