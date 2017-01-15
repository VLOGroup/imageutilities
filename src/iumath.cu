
#include "iumath.h"
#include "iucore.h"

#include "iumath/arithmetics.cuh"
#include "iumath/statistics.cuh"
#include "iumath/complex.cuh"
#include "iumath/fft.cuh"

#include "iuhelpermath.h"
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

void addC(iu::LinearDeviceMemory<float, 2>& src, const float& val, iu::LinearDeviceMemory<float, 2>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory<float2, 2>& src, const float2& val, iu::LinearDeviceMemory<float2, 2>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory<float3, 2>& src, const float3& val, iu::LinearDeviceMemory<float3, 2>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory<float4, 2>& src, const float4& val, iu::LinearDeviceMemory<float4, 2>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory<float, 3>& src, const float& val, iu::LinearDeviceMemory<float, 3>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory<float2, 3>& src, const float2& val, iu::LinearDeviceMemory<float2, 3>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory<float3, 3>& src, const float3& val, iu::LinearDeviceMemory<float3, 3>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory<float4, 3>& src, const float4& val, iu::LinearDeviceMemory<float4, 3>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory<float, 4>& src, const float& val, iu::LinearDeviceMemory<float, 4>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory<float2, 4>& src, const float2& val, iu::LinearDeviceMemory<float2, 4>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory<float3, 4>& src, const float3& val, iu::LinearDeviceMemory<float3, 4>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory<float4, 4>& src, const float4& val, iu::LinearDeviceMemory<float4, 4>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory<float, 5>& src, const float& val, iu::LinearDeviceMemory<float, 5>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory<float2, 5>& src, const float2& val, iu::LinearDeviceMemory<float2, 5>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory<float3, 5>& src, const float3& val, iu::LinearDeviceMemory<float3, 5>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory<float4, 5>& src, const float4& val, iu::LinearDeviceMemory<float4, 5>& dst) {iuprivate::math::addC(src,val,dst);}

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

void addC(iu::LinearHostMemory<float, 2>& src, const float& val, iu::LinearHostMemory<float, 2>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory<float2, 2>& src, const float2& val, iu::LinearHostMemory<float2, 2>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory<float3, 2>& src, const float3& val, iu::LinearHostMemory<float3, 2>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory<float4, 2>& src, const float4& val, iu::LinearHostMemory<float4, 2>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory<float, 3>& src, const float& val, iu::LinearHostMemory<float, 3>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory<float2, 3>& src, const float2& val, iu::LinearHostMemory<float2, 3>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory<float3, 3>& src, const float3& val, iu::LinearHostMemory<float3, 3>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory<float4, 3>& src, const float4& val, iu::LinearHostMemory<float4, 3>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory<float, 4>& src, const float& val, iu::LinearHostMemory<float, 4>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory<float2, 4>& src, const float2& val, iu::LinearHostMemory<float2, 4>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory<float3, 4>& src, const float3& val, iu::LinearHostMemory<float3, 4>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory<float4, 4>& src, const float4& val, iu::LinearHostMemory<float4, 4>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory<float, 5>& src, const float& val, iu::LinearHostMemory<float, 5>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory<float2, 5>& src, const float2& val, iu::LinearHostMemory<float2, 5>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory<float3, 5>& src, const float3& val, iu::LinearHostMemory<float3, 5>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory<float4, 5>& src, const float4& val, iu::LinearHostMemory<float4, 5>& dst) {iuprivate::math::addC(src,val,dst);}

void addC(iu::LinearHostMemory_32s_C1& src, const int& val, iu::LinearHostMemory_32s_C1& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory_32u_C1& src, const unsigned int& val, iu::LinearHostMemory_32u_C1& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory_16u_C1& src, const unsigned short& val, iu::LinearHostMemory_16u_C1& dst) {iuprivate::math::addC(src,val,dst);}

void addC(iu::LinearHostMemory_8u_C1& src, const unsigned char& val, iu::LinearHostMemory_8u_C1& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory_8u_C2& src, const uchar2& val, iu::LinearHostMemory_8u_C2& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory_8u_C3& src, const uchar3& val, iu::LinearHostMemory_8u_C3& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory_8u_C4& src, const uchar4& val, iu::LinearHostMemory_8u_C4& dst) {iuprivate::math::addC(src,val,dst);}

void addC(iu::ImageGpu_64f_C1& src, const double& val, iu::ImageGpu_64f_C1& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::ImageGpu_64f_C2& src, const double2& val, iu::ImageGpu_64f_C2& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::ImageGpu_64f_C3& src, const double3& val, iu::ImageGpu_64f_C3& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::ImageGpu_64f_C4& src, const double4& val, iu::ImageGpu_64f_C4& dst) {iuprivate::math::addC(src,val,dst);}

void addC(iu::VolumeGpu_64f_C1& src, const double& val, iu::VolumeGpu_64f_C1& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::VolumeGpu_64f_C2& src, const double2& val, iu::VolumeGpu_64f_C2& dst) {iuprivate::math::addC(src,val,dst);}

void addC(iu::LinearDeviceMemory_64f_C1& src, const double& val, iu::LinearDeviceMemory_64f_C1& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory_64f_C2& src, const double2& val, iu::LinearDeviceMemory_64f_C2& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory_64f_C3& src, const double3& val, iu::LinearDeviceMemory_64f_C3& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory_64f_C4& src, const double4& val, iu::LinearDeviceMemory_64f_C4& dst) {iuprivate::math::addC(src,val,dst);}

void addC(iu::LinearDeviceMemory<double, 2>& src, const double& val, iu::LinearDeviceMemory<double, 2>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory<double2, 2>& src, const double2& val, iu::LinearDeviceMemory<double2, 2>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory<double3, 2>& src, const double3& val, iu::LinearDeviceMemory<double3, 2>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory<double4, 2>& src, const double4& val, iu::LinearDeviceMemory<double4, 2>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory<double, 3>& src, const double& val, iu::LinearDeviceMemory<double, 3>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory<double2, 3>& src, const double2& val, iu::LinearDeviceMemory<double2, 3>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory<double3, 3>& src, const double3& val, iu::LinearDeviceMemory<double3, 3>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory<double4, 3>& src, const double4& val, iu::LinearDeviceMemory<double4, 3>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory<double, 4>& src, const double& val, iu::LinearDeviceMemory<double, 4>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory<double2, 4>& src, const double2& val, iu::LinearDeviceMemory<double2, 4>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory<double3, 4>& src, const double3& val, iu::LinearDeviceMemory<double3, 4>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory<double4, 4>& src, const double4& val, iu::LinearDeviceMemory<double4, 4>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory<double, 5>& src, const double& val, iu::LinearDeviceMemory<double, 5>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory<double2, 5>& src, const double2& val, iu::LinearDeviceMemory<double2, 5>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory<double3, 5>& src, const double3& val, iu::LinearDeviceMemory<double3, 5>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearDeviceMemory<double4, 5>& src, const double4& val, iu::LinearDeviceMemory<double4, 5>& dst) {iuprivate::math::addC(src,val,dst);}

void addC(iu::ImageCpu_64f_C1& src, const double& val, iu::ImageCpu_64f_C1& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::ImageCpu_64f_C2& src, const double2& val, iu::ImageCpu_64f_C2& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::ImageCpu_64f_C3& src, const double3& val, iu::ImageCpu_64f_C3& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::ImageCpu_64f_C4& src, const double4& val, iu::ImageCpu_64f_C4& dst) {iuprivate::math::addC(src,val,dst);}

void addC(iu::VolumeCpu_64f_C1& src, const double& val, iu::VolumeCpu_64f_C1& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::VolumeCpu_64f_C2& src, const double2& val, iu::VolumeCpu_64f_C2& dst) {iuprivate::math::addC(src,val,dst);}

void addC(iu::LinearHostMemory_64f_C1& src, const double& val, iu::LinearHostMemory_64f_C1& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory_64f_C2& src, const double2& val, iu::LinearHostMemory_64f_C2& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory_64f_C3& src, const double3& val, iu::LinearHostMemory_64f_C3& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory_64f_C4& src, const double4& val, iu::LinearHostMemory_64f_C4& dst) {iuprivate::math::addC(src,val,dst);}

void addC(iu::LinearHostMemory<double, 2>& src, const double& val, iu::LinearHostMemory<double, 2>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory<double2, 2>& src, const double2& val, iu::LinearHostMemory<double2, 2>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory<double3, 2>& src, const double3& val, iu::LinearHostMemory<double3, 2>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory<double4, 2>& src, const double4& val, iu::LinearHostMemory<double4, 2>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory<double, 3>& src, const double& val, iu::LinearHostMemory<double, 3>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory<double2, 3>& src, const double2& val, iu::LinearHostMemory<double2, 3>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory<double3, 3>& src, const double3& val, iu::LinearHostMemory<double3, 3>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory<double4, 3>& src, const double4& val, iu::LinearHostMemory<double4, 3>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory<double, 4>& src, const double& val, iu::LinearHostMemory<double, 4>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory<double2, 4>& src, const double2& val, iu::LinearHostMemory<double2, 4>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory<double3, 4>& src, const double3& val, iu::LinearHostMemory<double3, 4>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory<double4, 4>& src, const double4& val, iu::LinearHostMemory<double4, 4>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory<double, 5>& src, const double& val, iu::LinearHostMemory<double, 5>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory<double2, 5>& src, const double2& val, iu::LinearHostMemory<double2, 5>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory<double3, 5>& src, const double3& val, iu::LinearHostMemory<double3, 5>& dst) {iuprivate::math::addC(src,val,dst);}
void addC(iu::LinearHostMemory<double4, 5>& src, const double4& val, iu::LinearHostMemory<double4, 5>& dst) {iuprivate::math::addC(src,val,dst);}

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
void mulC(iu::VolumeGpu_32f_C2& src, const float2& val, iu::VolumeGpu_32f_C2& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::VolumeGpu_32f_C3& src, const float3& val, iu::VolumeGpu_32f_C3& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::VolumeGpu_32f_C4& src, const float4& val, iu::VolumeGpu_32f_C4& dst) {iuprivate::math::mulC(src,val,dst);}

void mulC(iu::LinearDeviceMemory_32f_C1& src, const float& val, iu::LinearDeviceMemory_32f_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory_32f_C2& src, const float2& val, iu::LinearDeviceMemory_32f_C2& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory_32f_C3& src, const float3& val, iu::LinearDeviceMemory_32f_C3& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory_32f_C4& src, const float4& val, iu::LinearDeviceMemory_32f_C4& dst) {iuprivate::math::mulC(src,val,dst);}

void mulC(iu::LinearDeviceMemory<float, 2>& src, const float& val, iu::LinearDeviceMemory<float, 2>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory<float2, 2>& src, const float2& val, iu::LinearDeviceMemory<float2, 2>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory<float3, 2>& src, const float3& val, iu::LinearDeviceMemory<float3, 2>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory<float4, 2>& src, const float4& val, iu::LinearDeviceMemory<float4, 2>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory<float, 3>& src, const float& val, iu::LinearDeviceMemory<float, 3>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory<float2, 3>& src, const float2& val, iu::LinearDeviceMemory<float2, 3>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory<float3, 3>& src, const float3& val, iu::LinearDeviceMemory<float3, 3>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory<float4, 3>& src, const float4& val, iu::LinearDeviceMemory<float4, 3>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory<float, 4>& src, const float& val, iu::LinearDeviceMemory<float, 4>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory<float2, 4>& src, const float2& val, iu::LinearDeviceMemory<float2, 4>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory<float3, 4>& src, const float3& val, iu::LinearDeviceMemory<float3, 4>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory<float4, 4>& src, const float4& val, iu::LinearDeviceMemory<float4, 4>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory<float, 5>& src, const float& val, iu::LinearDeviceMemory<float, 5>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory<float2, 5>& src, const float2& val, iu::LinearDeviceMemory<float2, 5>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory<float3, 5>& src, const float3& val, iu::LinearDeviceMemory<float3, 5>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory<float4, 5>& src, const float4& val, iu::LinearDeviceMemory<float4, 5>& dst) {iuprivate::math::mulC(src,val,dst);}

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
void mulC(iu::VolumeCpu_32f_C2& src, const float2& val, iu::VolumeCpu_32f_C2& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::VolumeCpu_32f_C3& src, const float3& val, iu::VolumeCpu_32f_C3& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::VolumeCpu_32f_C4& src, const float4& val, iu::VolumeCpu_32f_C4& dst) {iuprivate::math::mulC(src,val,dst);}

void mulC(iu::LinearHostMemory_32f_C1& src, const float& val, iu::LinearHostMemory_32f_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory_32f_C2& src, const float2& val, iu::LinearHostMemory_32f_C2& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory_32f_C3& src, const float3& val, iu::LinearHostMemory_32f_C3& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory_32f_C4& src, const float4& val, iu::LinearHostMemory_32f_C4& dst) {iuprivate::math::mulC(src,val,dst);}

void mulC(iu::LinearHostMemory<float, 2>& src, const float& val, iu::LinearHostMemory<float, 2>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory<float2, 2>& src, const float2& val, iu::LinearHostMemory<float2, 2>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory<float3, 2>& src, const float3& val, iu::LinearHostMemory<float3, 2>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory<float4, 2>& src, const float4& val, iu::LinearHostMemory<float4, 2>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory<float, 3>& src, const float& val, iu::LinearHostMemory<float, 3>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory<float2, 3>& src, const float2& val, iu::LinearHostMemory<float2, 3>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory<float3, 3>& src, const float3& val, iu::LinearHostMemory<float3, 3>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory<float4, 3>& src, const float4& val, iu::LinearHostMemory<float4, 3>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory<float, 4>& src, const float& val, iu::LinearHostMemory<float, 4>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory<float2, 4>& src, const float2& val, iu::LinearHostMemory<float2, 4>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory<float3, 4>& src, const float3& val, iu::LinearHostMemory<float3, 4>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory<float4, 4>& src, const float4& val, iu::LinearHostMemory<float4, 4>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory<float, 5>& src, const float& val, iu::LinearHostMemory<float, 5>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory<float2, 5>& src, const float2& val, iu::LinearHostMemory<float2, 5>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory<float3, 5>& src, const float3& val, iu::LinearHostMemory<float3, 5>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory<float4, 5>& src, const float4& val, iu::LinearHostMemory<float4, 5>& dst) {iuprivate::math::mulC(src,val,dst);}

void mulC(iu::LinearHostMemory_32s_C1& src, const int& val, iu::LinearHostMemory_32s_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory_32u_C1& src, const unsigned int& val, iu::LinearHostMemory_32u_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory_16u_C1& src, const unsigned short& val, iu::LinearHostMemory_16u_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory_8u_C1& src, const unsigned char& val, iu::LinearHostMemory_8u_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory_8u_C2& src, const uchar2& val, iu::LinearHostMemory_8u_C2& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory_8u_C3& src, const uchar3& val, iu::LinearHostMemory_8u_C3& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory_8u_C4& src, const uchar4& val, iu::LinearHostMemory_8u_C4& dst) {iuprivate::math::mulC(src,val,dst);}

void mulC(iu::ImageGpu_64f_C1& src, const double& val, iu::ImageGpu_64f_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::ImageGpu_64f_C2& src, const double2& val, iu::ImageGpu_64f_C2& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::ImageGpu_64f_C3& src, const double3& val, iu::ImageGpu_64f_C3& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::ImageGpu_64f_C4& src, const double4& val, iu::ImageGpu_64f_C4& dst) {iuprivate::math::mulC(src,val,dst);}

void mulC(iu::VolumeGpu_64f_C1& src, const double& val, iu::VolumeGpu_64f_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::VolumeGpu_64f_C2& src, const double2& val, iu::VolumeGpu_64f_C2& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::VolumeGpu_64f_C3& src, const double3& val, iu::VolumeGpu_64f_C3& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::VolumeGpu_64f_C4& src, const double4& val, iu::VolumeGpu_64f_C4& dst) {iuprivate::math::mulC(src,val,dst);}

void mulC(iu::LinearDeviceMemory_64f_C1& src, const double& val, iu::LinearDeviceMemory_64f_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory_64f_C2& src, const double2& val, iu::LinearDeviceMemory_64f_C2& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory_64f_C3& src, const double3& val, iu::LinearDeviceMemory_64f_C3& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory_64f_C4& src, const double4& val, iu::LinearDeviceMemory_64f_C4& dst) {iuprivate::math::mulC(src,val,dst);}

void mulC(iu::LinearDeviceMemory<double, 2>& src, const double& val, iu::LinearDeviceMemory<double, 2>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory<double2, 2>& src, const double2& val, iu::LinearDeviceMemory<double2, 2>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory<double3, 2>& src, const double3& val, iu::LinearDeviceMemory<double3, 2>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory<double4, 2>& src, const double4& val, iu::LinearDeviceMemory<double4, 2>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory<double, 3>& src, const double& val, iu::LinearDeviceMemory<double, 3>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory<double2, 3>& src, const double2& val, iu::LinearDeviceMemory<double2, 3>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory<double3, 3>& src, const double3& val, iu::LinearDeviceMemory<double3, 3>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory<double4, 3>& src, const double4& val, iu::LinearDeviceMemory<double4, 3>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory<double, 4>& src, const double& val, iu::LinearDeviceMemory<double, 4>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory<double2, 4>& src, const double2& val, iu::LinearDeviceMemory<double2, 4>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory<double3, 4>& src, const double3& val, iu::LinearDeviceMemory<double3, 4>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory<double4, 4>& src, const double4& val, iu::LinearDeviceMemory<double4, 4>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory<double, 5>& src, const double& val, iu::LinearDeviceMemory<double, 5>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory<double2, 5>& src, const double2& val, iu::LinearDeviceMemory<double2, 5>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory<double3, 5>& src, const double3& val, iu::LinearDeviceMemory<double3, 5>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearDeviceMemory<double4, 5>& src, const double4& val, iu::LinearDeviceMemory<double4, 5>& dst) {iuprivate::math::mulC(src,val,dst);}

void mulC(iu::ImageCpu_64f_C1& src, const double& val, iu::ImageCpu_64f_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::ImageCpu_64f_C2& src, const double2& val, iu::ImageCpu_64f_C2& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::ImageCpu_64f_C3& src, const double3& val, iu::ImageCpu_64f_C3& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::ImageCpu_64f_C4& src, const double4& val, iu::ImageCpu_64f_C4& dst) {iuprivate::math::mulC(src,val,dst);}

void mulC(iu::VolumeCpu_64f_C1& src, const double& val, iu::VolumeCpu_64f_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::VolumeCpu_64f_C2& src, const double2& val, iu::VolumeCpu_64f_C2& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::VolumeCpu_64f_C3& src, const double3& val, iu::VolumeCpu_64f_C3& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::VolumeCpu_64f_C4& src, const double4& val, iu::VolumeCpu_64f_C4& dst) {iuprivate::math::mulC(src,val,dst);}

void mulC(iu::LinearHostMemory_64f_C1& src, const double& val, iu::LinearHostMemory_64f_C1& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory_64f_C2& src, const double2& val, iu::LinearHostMemory_64f_C2& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory_64f_C3& src, const double3& val, iu::LinearHostMemory_64f_C3& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory_64f_C4& src, const double4& val, iu::LinearHostMemory_64f_C4& dst) {iuprivate::math::mulC(src,val,dst);}

void mulC(iu::LinearHostMemory<double, 2>& src, const double& val, iu::LinearHostMemory<double, 2>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory<double2, 2>& src, const double2& val, iu::LinearHostMemory<double2, 2>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory<double3, 2>& src, const double3& val, iu::LinearHostMemory<double3, 2>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory<double4, 2>& src, const double4& val, iu::LinearHostMemory<double4, 2>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory<double, 3>& src, const double& val, iu::LinearHostMemory<double, 3>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory<double2, 3>& src, const double2& val, iu::LinearHostMemory<double2, 3>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory<double3, 3>& src, const double3& val, iu::LinearHostMemory<double3, 3>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory<double4, 3>& src, const double4& val, iu::LinearHostMemory<double4, 3>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory<double, 4>& src, const double& val, iu::LinearHostMemory<double, 4>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory<double2, 4>& src, const double2& val, iu::LinearHostMemory<double2, 4>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory<double3, 4>& src, const double3& val, iu::LinearHostMemory<double3, 4>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory<double4, 4>& src, const double4& val, iu::LinearHostMemory<double4, 4>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory<double, 5>& src, const double& val, iu::LinearHostMemory<double, 5>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory<double2, 5>& src, const double2& val, iu::LinearHostMemory<double2, 5>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory<double3, 5>& src, const double3& val, iu::LinearHostMemory<double3, 5>& dst) {iuprivate::math::mulC(src,val,dst);}
void mulC(iu::LinearHostMemory<double4, 5>& src, const double4& val, iu::LinearHostMemory<double4, 5>& dst) {iuprivate::math::mulC(src,val,dst);}

void abs(iu::LinearDeviceMemory<float, 1>&src, iu::LinearDeviceMemory<float,1>&dst) {iuprivate::math::abs(src, dst);}

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

void addWeighted(iu::LinearDeviceMemory<float, 2>& src1, const float& weight1,
                 iu::LinearDeviceMemory<float, 2>& src2, const float& weight2,iu::LinearDeviceMemory<float, 2>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory<float2, 2>& src1, const float2& weight1,
                 iu::LinearDeviceMemory<float2, 2>& src2, const float2& weight2,iu::LinearDeviceMemory<float2, 2>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory<float3, 2>& src1, const float3& weight1,
                 iu::LinearDeviceMemory<float3, 2>& src2, const float3& weight2,iu::LinearDeviceMemory<float3, 2>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory<float4, 2>& src1, const float4& weight1,
                 iu::LinearDeviceMemory<float4, 2>& src2, const float4& weight2,iu::LinearDeviceMemory<float4, 2>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory<float, 3>& src1, const float& weight1,
                 iu::LinearDeviceMemory<float, 3>& src2, const float& weight2,iu::LinearDeviceMemory<float, 3>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory<float2, 3>& src1, const float2& weight1,
                 iu::LinearDeviceMemory<float2, 3>& src2, const float2& weight2,iu::LinearDeviceMemory<float2, 3>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory<float3, 3>& src1, const float3& weight1,
                 iu::LinearDeviceMemory<float3, 3>& src2, const float3& weight2,iu::LinearDeviceMemory<float3, 3>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory<float4, 3>& src1, const float4& weight1,
                 iu::LinearDeviceMemory<float4, 3>& src2, const float4& weight2,iu::LinearDeviceMemory<float4, 3>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory<float, 4>& src1, const float& weight1,
                 iu::LinearDeviceMemory<float, 4>& src2, const float& weight2,iu::LinearDeviceMemory<float, 4>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory<float2, 4>& src1, const float2& weight1,
                 iu::LinearDeviceMemory<float2, 4>& src2, const float2& weight2,iu::LinearDeviceMemory<float2, 4>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory<float3, 4>& src1, const float3& weight1,
                 iu::LinearDeviceMemory<float3, 4>& src2, const float3& weight2,iu::LinearDeviceMemory<float3, 4>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory<float4, 4>& src1, const float4& weight1,
                 iu::LinearDeviceMemory<float4, 4>& src2, const float4& weight2,iu::LinearDeviceMemory<float4, 4>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory<float, 5>& src1, const float& weight1,
                 iu::LinearDeviceMemory<float, 5>& src2, const float& weight2,iu::LinearDeviceMemory<float, 5>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory<float2, 5>& src1, const float2& weight1,
                 iu::LinearDeviceMemory<float2, 5>& src2, const float2& weight2,iu::LinearDeviceMemory<float2, 5>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory<float3, 5>& src1, const float3& weight1,
                 iu::LinearDeviceMemory<float3, 5>& src2, const float3& weight2,iu::LinearDeviceMemory<float3, 5>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory<float4, 5>& src1, const float4& weight1,
                 iu::LinearDeviceMemory<float4, 5>& src2, const float4& weight2,iu::LinearDeviceMemory<float4, 5>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}

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

void addWeighted(iu::LinearHostMemory<float, 2>& src1, const float& weight1,
                 iu::LinearHostMemory<float, 2>& src2, const float& weight2,iu::LinearHostMemory<float, 2>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory<float2, 2>& src1, const float2& weight1,
                 iu::LinearHostMemory<float2, 2>& src2, const float2& weight2,iu::LinearHostMemory<float2, 2>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory<float3, 2>& src1, const float3& weight1,
                 iu::LinearHostMemory<float3, 2>& src2, const float3& weight2,iu::LinearHostMemory<float3, 2>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory<float4, 2>& src1, const float4& weight1,
                 iu::LinearHostMemory<float4, 2>& src2, const float4& weight2,iu::LinearHostMemory<float4, 2>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory<float, 3>& src1, const float& weight1,
                 iu::LinearHostMemory<float, 3>& src2, const float& weight2,iu::LinearHostMemory<float, 3>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory<float2, 3>& src1, const float2& weight1,
                 iu::LinearHostMemory<float2, 3>& src2, const float2& weight2,iu::LinearHostMemory<float2, 3>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory<float3, 3>& src1, const float3& weight1,
                 iu::LinearHostMemory<float3, 3>& src2, const float3& weight2,iu::LinearHostMemory<float3, 3>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory<float4, 3>& src1, const float4& weight1,
                 iu::LinearHostMemory<float4, 3>& src2, const float4& weight2,iu::LinearHostMemory<float4, 3>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory<float, 4>& src1, const float& weight1,
                 iu::LinearHostMemory<float, 4>& src2, const float& weight2,iu::LinearHostMemory<float, 4>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory<float2, 4>& src1, const float2& weight1,
                 iu::LinearHostMemory<float2, 4>& src2, const float2& weight2,iu::LinearHostMemory<float2, 4>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory<float3, 4>& src1, const float3& weight1,
                 iu::LinearHostMemory<float3, 4>& src2, const float3& weight2,iu::LinearHostMemory<float3, 4>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory<float4, 4>& src1, const float4& weight1,
                 iu::LinearHostMemory<float4, 4>& src2, const float4& weight2,iu::LinearHostMemory<float4, 4>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory<float, 5>& src1, const float& weight1,
                 iu::LinearHostMemory<float, 5>& src2, const float& weight2,iu::LinearHostMemory<float, 5>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory<float2, 5>& src1, const float2& weight1,
                 iu::LinearHostMemory<float2, 5>& src2, const float2& weight2,iu::LinearHostMemory<float2, 5>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory<float3, 5>& src1, const float3& weight1,
                 iu::LinearHostMemory<float3, 5>& src2, const float3& weight2,iu::LinearHostMemory<float3, 5>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory<float4, 5>& src1, const float4& weight1,
                 iu::LinearHostMemory<float4, 5>& src2, const float4& weight2,iu::LinearHostMemory<float4, 5>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}

void addWeighted(iu::ImageGpu_64f_C1& src1, const double& weight1,
                 iu::ImageGpu_64f_C1& src2, const double& weight2,iu::ImageGpu_64f_C1& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::ImageGpu_64f_C2& src1, const double2& weight1,
                 iu::ImageGpu_64f_C2& src2, const double2& weight2,iu::ImageGpu_64f_C2& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::ImageGpu_64f_C3& src1, const double3& weight1,
                 iu::ImageGpu_64f_C3& src2, const double3& weight2,iu::ImageGpu_64f_C3& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::ImageGpu_64f_C4& src1, const double4& weight1,
                 iu::ImageGpu_64f_C4& src2, const double4& weight2,iu::ImageGpu_64f_C4& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}

void addWeighted(iu::VolumeGpu_64f_C1& src1, const double& weight1,
                 iu::VolumeGpu_64f_C1& src2, const double& weight2,iu::VolumeGpu_64f_C1& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}

void addWeighted(iu::LinearDeviceMemory_64f_C1& src1, const double& weight1,
                 iu::LinearDeviceMemory_64f_C1& src2, const double& weight2,iu::LinearDeviceMemory_64f_C1& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory_64f_C2& src1, const double2& weight1,
                 iu::LinearDeviceMemory_64f_C2& src2, const double2& weight2,iu::LinearDeviceMemory_64f_C2& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory_64f_C3& src1, const double3& weight1,
                 iu::LinearDeviceMemory_64f_C3& src2, const double3& weight2,iu::LinearDeviceMemory_64f_C3& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory_64f_C4& src1, const double4& weight1,
                 iu::LinearDeviceMemory_64f_C4& src2, const double4& weight2,iu::LinearDeviceMemory_64f_C4& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}

void addWeighted(iu::LinearDeviceMemory<double, 2>& src1, const double& weight1,
                 iu::LinearDeviceMemory<double, 2>& src2, const double& weight2,iu::LinearDeviceMemory<double, 2>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory<double2, 2>& src1, const double2& weight1,
                 iu::LinearDeviceMemory<double2, 2>& src2, const double2& weight2,iu::LinearDeviceMemory<double2, 2>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory<double3, 2>& src1, const double3& weight1,
                 iu::LinearDeviceMemory<double3, 2>& src2, const double3& weight2,iu::LinearDeviceMemory<double3, 2>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory<double4, 2>& src1, const double4& weight1,
                 iu::LinearDeviceMemory<double4, 2>& src2, const double4& weight2,iu::LinearDeviceMemory<double4, 2>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory<double, 3>& src1, const double& weight1,
                 iu::LinearDeviceMemory<double, 3>& src2, const double& weight2,iu::LinearDeviceMemory<double, 3>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory<double2, 3>& src1, const double2& weight1,
                 iu::LinearDeviceMemory<double2, 3>& src2, const double2& weight2,iu::LinearDeviceMemory<double2, 3>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory<double3, 3>& src1, const double3& weight1,
                 iu::LinearDeviceMemory<double3, 3>& src2, const double3& weight2,iu::LinearDeviceMemory<double3, 3>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory<double4, 3>& src1, const double4& weight1,
                 iu::LinearDeviceMemory<double4, 3>& src2, const double4& weight2,iu::LinearDeviceMemory<double4, 3>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory<double, 4>& src1, const double& weight1,
                 iu::LinearDeviceMemory<double, 4>& src2, const double& weight2,iu::LinearDeviceMemory<double, 4>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory<double2, 4>& src1, const double2& weight1,
                 iu::LinearDeviceMemory<double2, 4>& src2, const double2& weight2,iu::LinearDeviceMemory<double2, 4>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory<double3, 4>& src1, const double3& weight1,
                 iu::LinearDeviceMemory<double3, 4>& src2, const double3& weight2,iu::LinearDeviceMemory<double3, 4>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory<double4, 4>& src1, const double4& weight1,
                 iu::LinearDeviceMemory<double4, 4>& src2, const double4& weight2,iu::LinearDeviceMemory<double4, 4>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory<double, 5>& src1, const double& weight1,
                 iu::LinearDeviceMemory<double, 5>& src2, const double& weight2,iu::LinearDeviceMemory<double, 5>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory<double2, 5>& src1, const double2& weight1,
                 iu::LinearDeviceMemory<double2, 5>& src2, const double2& weight2,iu::LinearDeviceMemory<double2, 5>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory<double3, 5>& src1, const double3& weight1,
                 iu::LinearDeviceMemory<double3, 5>& src2, const double3& weight2,iu::LinearDeviceMemory<double3, 5>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearDeviceMemory<double4, 5>& src1, const double4& weight1,
                 iu::LinearDeviceMemory<double4, 5>& src2, const double4& weight2,iu::LinearDeviceMemory<double4, 5>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}

void addWeighted(iu::ImageCpu_64f_C1& src1, const double& weight1,
                 iu::ImageCpu_64f_C1& src2, const double& weight2,iu::ImageCpu_64f_C1& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::ImageCpu_64f_C2& src1, const double2& weight1,
                 iu::ImageCpu_64f_C2& src2, const double2& weight2,iu::ImageCpu_64f_C2& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::ImageCpu_64f_C3& src1, const double3& weight1,
                 iu::ImageCpu_64f_C3& src2, const double3& weight2,iu::ImageCpu_64f_C3& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::ImageCpu_64f_C4& src1, const double4& weight1,
                 iu::ImageCpu_64f_C4& src2, const double4& weight2,iu::ImageCpu_64f_C4& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}

void addWeighted(iu::VolumeCpu_64f_C1& src1, const double& weight1,
                 iu::VolumeCpu_64f_C1& src2, const double& weight2,iu::VolumeCpu_64f_C1& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}

void addWeighted(iu::LinearHostMemory_64f_C1& src1, const double& weight1,
                 iu::LinearHostMemory_64f_C1& src2, const double& weight2,iu::LinearHostMemory_64f_C1& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory_64f_C2& src1, const double2& weight1,
                 iu::LinearHostMemory_64f_C2& src2, const double2& weight2,iu::LinearHostMemory_64f_C2& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory_64f_C3& src1, const double3& weight1,
                 iu::LinearHostMemory_64f_C3& src2, const double3& weight2,iu::LinearHostMemory_64f_C3& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory_64f_C4& src1, const double4& weight1,
                 iu::LinearHostMemory_64f_C4& src2, const double4& weight2,iu::LinearHostMemory_64f_C4& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}

void addWeighted(iu::LinearHostMemory<double, 2>& src1, const double& weight1,
                 iu::LinearHostMemory<double, 2>& src2, const double& weight2,iu::LinearHostMemory<double, 2>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory<double2, 2>& src1, const double2& weight1,
                 iu::LinearHostMemory<double2, 2>& src2, const double2& weight2,iu::LinearHostMemory<double2, 2>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory<double3, 2>& src1, const double3& weight1,
                 iu::LinearHostMemory<double3, 2>& src2, const double3& weight2,iu::LinearHostMemory<double3, 2>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory<double4, 2>& src1, const double4& weight1,
                 iu::LinearHostMemory<double4, 2>& src2, const double4& weight2,iu::LinearHostMemory<double4, 2>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory<double, 3>& src1, const double& weight1,
                 iu::LinearHostMemory<double, 3>& src2, const double& weight2,iu::LinearHostMemory<double, 3>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory<double2, 3>& src1, const double2& weight1,
                 iu::LinearHostMemory<double2, 3>& src2, const double2& weight2,iu::LinearHostMemory<double2, 3>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory<double3, 3>& src1, const double3& weight1,
                 iu::LinearHostMemory<double3, 3>& src2, const double3& weight2,iu::LinearHostMemory<double3, 3>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory<double4, 3>& src1, const double4& weight1,
                 iu::LinearHostMemory<double4, 3>& src2, const double4& weight2,iu::LinearHostMemory<double4, 3>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory<double, 4>& src1, const double& weight1,
                 iu::LinearHostMemory<double, 4>& src2, const double& weight2,iu::LinearHostMemory<double, 4>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory<double2, 4>& src1, const double2& weight1,
                 iu::LinearHostMemory<double2, 4>& src2, const double2& weight2,iu::LinearHostMemory<double2, 4>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory<double3, 4>& src1, const double3& weight1,
                 iu::LinearHostMemory<double3, 4>& src2, const double3& weight2,iu::LinearHostMemory<double3, 4>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory<double4, 4>& src1, const double4& weight1,
                 iu::LinearHostMemory<double4, 4>& src2, const double4& weight2,iu::LinearHostMemory<double4, 4>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory<double, 5>& src1, const double& weight1,
                 iu::LinearHostMemory<double, 5>& src2, const double& weight2,iu::LinearHostMemory<double, 5>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory<double2, 5>& src1, const double2& weight1,
                 iu::LinearHostMemory<double2, 5>& src2, const double2& weight2,iu::LinearHostMemory<double2, 5>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory<double3, 5>& src1, const double3& weight1,
                 iu::LinearHostMemory<double3, 5>& src2, const double3& weight2,iu::LinearHostMemory<double3, 5>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}
void addWeighted(iu::LinearHostMemory<double4, 5>& src1, const double4& weight1,
                 iu::LinearHostMemory<double4, 5>& src2, const double4& weight2,iu::LinearHostMemory<double4, 5>& dst) {iuprivate::math::addWeighted(src1,weight1,src2,weight2,dst);}


// pointwise multiply
void mul(iu::ImageGpu_32f_C1& src1, iu::ImageGpu_32f_C1& src2, iu::ImageGpu_32f_C1& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::ImageGpu_32f_C2& src1, iu::ImageGpu_32f_C2& src2, iu::ImageGpu_32f_C2& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::ImageGpu_32f_C3& src1, iu::ImageGpu_32f_C3& src2, iu::ImageGpu_32f_C3& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::ImageGpu_32f_C4& src1, iu::ImageGpu_32f_C4& src2, iu::ImageGpu_32f_C4& dst) {iuprivate::math::mul(src1,src2,dst);}

void mul(iu::ImageGpu_8u_C1& src1, iu::ImageGpu_8u_C1& src2, iu::ImageGpu_8u_C1& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::ImageGpu_8u_C4& src1, iu::ImageGpu_8u_C4& src2, iu::ImageGpu_8u_C4& dst) {iuprivate::math::mul(src1,src2,dst);}

void mul(iu::VolumeGpu_32f_C1& src1, iu::VolumeGpu_32f_C1& src2, iu::VolumeGpu_32f_C1& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::VolumeGpu_32f_C2& src1, iu::VolumeGpu_32f_C2& src2, iu::VolumeGpu_32f_C2& dst) {iuprivate::math::mul(src1,src2,dst);}

void mul(iu::LinearDeviceMemory_32f_C1& src1, iu::LinearDeviceMemory_32f_C1& src2, iu::LinearDeviceMemory_32f_C1& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory_32f_C2& src1, iu::LinearDeviceMemory_32f_C2& src2, iu::LinearDeviceMemory_32f_C2& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory_32f_C3& src1, iu::LinearDeviceMemory_32f_C3& src2, iu::LinearDeviceMemory_32f_C3& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory_32f_C4& src1, iu::LinearDeviceMemory_32f_C4& src2, iu::LinearDeviceMemory_32f_C4& dst) {iuprivate::math::mul(src1,src2,dst);}

void mul(iu::LinearDeviceMemory<float, 2>& src1, iu::LinearDeviceMemory<float, 2>& src2, iu::LinearDeviceMemory<float, 2>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory<float2, 2>& src1, iu::LinearDeviceMemory<float2, 2>& src2, iu::LinearDeviceMemory<float2, 2>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory<float3, 2>& src1, iu::LinearDeviceMemory<float3, 2>& src2, iu::LinearDeviceMemory<float3, 2>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory<float4, 2>& src1, iu::LinearDeviceMemory<float4, 2>& src2, iu::LinearDeviceMemory<float4, 2>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory<float, 3>& src1, iu::LinearDeviceMemory<float, 3>& src2, iu::LinearDeviceMemory<float, 3>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory<float2, 3>& src1, iu::LinearDeviceMemory<float2, 3>& src2, iu::LinearDeviceMemory<float2, 3>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory<float3, 3>& src1, iu::LinearDeviceMemory<float3, 3>& src2, iu::LinearDeviceMemory<float3, 3>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory<float4, 3>& src1, iu::LinearDeviceMemory<float4, 3>& src2, iu::LinearDeviceMemory<float4, 3>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory<float, 4>& src1, iu::LinearDeviceMemory<float, 4>& src2, iu::LinearDeviceMemory<float, 4>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory<float2, 4>& src1, iu::LinearDeviceMemory<float2, 4>& src2, iu::LinearDeviceMemory<float2, 4>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory<float3, 4>& src1, iu::LinearDeviceMemory<float3, 4>& src2, iu::LinearDeviceMemory<float3, 4>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory<float4, 4>& src1, iu::LinearDeviceMemory<float4, 4>& src2, iu::LinearDeviceMemory<float4, 4>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory<float, 5>& src1, iu::LinearDeviceMemory<float, 5>& src2, iu::LinearDeviceMemory<float, 5>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory<float2, 5>& src1, iu::LinearDeviceMemory<float2, 5>& src2, iu::LinearDeviceMemory<float2, 5>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory<float3, 5>& src1, iu::LinearDeviceMemory<float3, 5>& src2, iu::LinearDeviceMemory<float3, 5>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory<float4, 5>& src1, iu::LinearDeviceMemory<float4, 5>& src2, iu::LinearDeviceMemory<float4, 5>& dst) {iuprivate::math::mul(src1,src2,dst);}

void mul(iu::LinearDeviceMemory_8u_C1& src1, iu::LinearDeviceMemory_8u_C1& src2, iu::LinearDeviceMemory_8u_C1& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory_8u_C4& src1, iu::LinearDeviceMemory_8u_C4& src2, iu::LinearDeviceMemory_8u_C4& dst) {iuprivate::math::mul(src1,src2,dst);}

void mul(iu::ImageCpu_32f_C1& src1, iu::ImageCpu_32f_C1& src2, iu::ImageCpu_32f_C1& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::ImageCpu_32f_C2& src1, iu::ImageCpu_32f_C2& src2, iu::ImageCpu_32f_C2& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::ImageCpu_32f_C3& src1, iu::ImageCpu_32f_C3& src2, iu::ImageCpu_32f_C3& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::ImageCpu_32f_C4& src1, iu::ImageCpu_32f_C4& src2, iu::ImageCpu_32f_C4& dst) {iuprivate::math::mul(src1,src2,dst);}

void mul(iu::ImageCpu_8u_C1& src1, iu::ImageCpu_8u_C1& src2, iu::ImageCpu_8u_C1& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::ImageCpu_8u_C4& src1, iu::ImageCpu_8u_C4& src2, iu::ImageCpu_8u_C4& dst) {iuprivate::math::mul(src1,src2,dst);}

void mul(iu::VolumeCpu_32f_C1& src1, iu::VolumeCpu_32f_C1& src2, iu::VolumeCpu_32f_C1& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::VolumeCpu_32f_C2& src1, iu::VolumeCpu_32f_C2& src2, iu::VolumeCpu_32f_C2& dst) {iuprivate::math::mul(src1,src2,dst);}

void mul(iu::LinearHostMemory_32f_C1& src1, iu::LinearHostMemory_32f_C1& src2, iu::LinearHostMemory_32f_C1& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory_32f_C2& src1, iu::LinearHostMemory_32f_C2& src2, iu::LinearHostMemory_32f_C2& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory_32f_C3& src1, iu::LinearHostMemory_32f_C3& src2, iu::LinearHostMemory_32f_C3& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory_32f_C4& src1, iu::LinearHostMemory_32f_C4& src2, iu::LinearHostMemory_32f_C4& dst) {iuprivate::math::mul(src1,src2,dst);}

void mul(iu::LinearHostMemory<float, 2>& src1, iu::LinearHostMemory<float, 2>& src2, iu::LinearHostMemory<float, 2>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory<float2, 2>& src1, iu::LinearHostMemory<float2, 2>& src2, iu::LinearHostMemory<float2, 2>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory<float3, 2>& src1, iu::LinearHostMemory<float3, 2>& src2, iu::LinearHostMemory<float3, 2>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory<float4, 2>& src1, iu::LinearHostMemory<float4, 2>& src2, iu::LinearHostMemory<float4, 2>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory<float, 3>& src1, iu::LinearHostMemory<float, 3>& src2, iu::LinearHostMemory<float, 3>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory<float2, 3>& src1, iu::LinearHostMemory<float2, 3>& src2, iu::LinearHostMemory<float2, 3>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory<float3, 3>& src1, iu::LinearHostMemory<float3, 3>& src2, iu::LinearHostMemory<float3, 3>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory<float4, 3>& src1, iu::LinearHostMemory<float4, 3>& src2, iu::LinearHostMemory<float4, 3>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory<float, 4>& src1, iu::LinearHostMemory<float, 4>& src2, iu::LinearHostMemory<float, 4>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory<float2, 4>& src1, iu::LinearHostMemory<float2, 4>& src2, iu::LinearHostMemory<float2, 4>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory<float3, 4>& src1, iu::LinearHostMemory<float3, 4>& src2, iu::LinearHostMemory<float3, 4>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory<float4, 4>& src1, iu::LinearHostMemory<float4, 4>& src2, iu::LinearHostMemory<float4, 4>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory<float, 5>& src1, iu::LinearHostMemory<float, 5>& src2, iu::LinearHostMemory<float, 5>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory<float2, 5>& src1, iu::LinearHostMemory<float2, 5>& src2, iu::LinearHostMemory<float2, 5>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory<float3, 5>& src1, iu::LinearHostMemory<float3, 5>& src2, iu::LinearHostMemory<float3, 5>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory<float4, 5>& src1, iu::LinearHostMemory<float4, 5>& src2, iu::LinearHostMemory<float4, 5>& dst) {iuprivate::math::mul(src1,src2,dst);}

void mul(iu::LinearHostMemory_8u_C1& src1, iu::LinearHostMemory_8u_C1& src2, iu::LinearHostMemory_8u_C1& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory_8u_C4& src1, iu::LinearHostMemory_8u_C4& src2, iu::LinearHostMemory_8u_C4& dst) {iuprivate::math::mul(src1,src2,dst);}

void mul(iu::ImageGpu_64f_C1& src1, iu::ImageGpu_64f_C1& src2, iu::ImageGpu_64f_C1& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::ImageGpu_64f_C2& src1, iu::ImageGpu_64f_C2& src2, iu::ImageGpu_64f_C2& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::ImageGpu_64f_C3& src1, iu::ImageGpu_64f_C3& src2, iu::ImageGpu_64f_C3& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::ImageGpu_64f_C4& src1, iu::ImageGpu_64f_C4& src2, iu::ImageGpu_64f_C4& dst) {iuprivate::math::mul(src1,src2,dst);}

void mul(iu::VolumeGpu_64f_C1& src1, iu::VolumeGpu_64f_C1& src2, iu::VolumeGpu_64f_C1& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::VolumeGpu_64f_C2& src1, iu::VolumeGpu_64f_C2& src2, iu::VolumeGpu_64f_C2& dst) {iuprivate::math::mul(src1,src2,dst);}

void mul(iu::LinearDeviceMemory_64f_C1& src1, iu::LinearDeviceMemory_64f_C1& src2, iu::LinearDeviceMemory_64f_C1& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory_64f_C2& src1, iu::LinearDeviceMemory_64f_C2& src2, iu::LinearDeviceMemory_64f_C2& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory_64f_C3& src1, iu::LinearDeviceMemory_64f_C3& src2, iu::LinearDeviceMemory_64f_C3& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory_64f_C4& src1, iu::LinearDeviceMemory_64f_C4& src2, iu::LinearDeviceMemory_64f_C4& dst) {iuprivate::math::mul(src1,src2,dst);}

void mul(iu::LinearDeviceMemory<double, 2>& src1, iu::LinearDeviceMemory<double, 2>& src2, iu::LinearDeviceMemory<double, 2>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory<double2, 2>& src1, iu::LinearDeviceMemory<double2, 2>& src2, iu::LinearDeviceMemory<double2, 2>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory<double3, 2>& src1, iu::LinearDeviceMemory<double3, 2>& src2, iu::LinearDeviceMemory<double3, 2>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory<double4, 2>& src1, iu::LinearDeviceMemory<double4, 2>& src2, iu::LinearDeviceMemory<double4, 2>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory<double, 3>& src1, iu::LinearDeviceMemory<double, 3>& src2, iu::LinearDeviceMemory<double, 3>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory<double2, 3>& src1, iu::LinearDeviceMemory<double2, 3>& src2, iu::LinearDeviceMemory<double2, 3>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory<double3, 3>& src1, iu::LinearDeviceMemory<double3, 3>& src2, iu::LinearDeviceMemory<double3, 3>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory<double4, 3>& src1, iu::LinearDeviceMemory<double4, 3>& src2, iu::LinearDeviceMemory<double4, 3>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory<double, 4>& src1, iu::LinearDeviceMemory<double, 4>& src2, iu::LinearDeviceMemory<double, 4>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory<double2, 4>& src1, iu::LinearDeviceMemory<double2, 4>& src2, iu::LinearDeviceMemory<double2, 4>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory<double3, 4>& src1, iu::LinearDeviceMemory<double3, 4>& src2, iu::LinearDeviceMemory<double3, 4>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory<double4, 4>& src1, iu::LinearDeviceMemory<double4, 4>& src2, iu::LinearDeviceMemory<double4, 4>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory<double, 5>& src1, iu::LinearDeviceMemory<double, 5>& src2, iu::LinearDeviceMemory<double, 5>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory<double2, 5>& src1, iu::LinearDeviceMemory<double2, 5>& src2, iu::LinearDeviceMemory<double2, 5>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory<double3, 5>& src1, iu::LinearDeviceMemory<double3, 5>& src2, iu::LinearDeviceMemory<double3, 5>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearDeviceMemory<double4, 5>& src1, iu::LinearDeviceMemory<double4, 5>& src2, iu::LinearDeviceMemory<double4, 5>& dst) {iuprivate::math::mul(src1,src2,dst);}

void mul(iu::ImageCpu_64f_C1& src1, iu::ImageCpu_64f_C1& src2, iu::ImageCpu_64f_C1& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::ImageCpu_64f_C2& src1, iu::ImageCpu_64f_C2& src2, iu::ImageCpu_64f_C2& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::ImageCpu_64f_C3& src1, iu::ImageCpu_64f_C3& src2, iu::ImageCpu_64f_C3& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::ImageCpu_64f_C4& src1, iu::ImageCpu_64f_C4& src2, iu::ImageCpu_64f_C4& dst) {iuprivate::math::mul(src1,src2,dst);}

void mul(iu::VolumeCpu_64f_C1& src1, iu::VolumeCpu_64f_C1& src2, iu::VolumeCpu_64f_C1& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::VolumeCpu_64f_C2& src1, iu::VolumeCpu_64f_C2& src2, iu::VolumeCpu_64f_C2& dst) {iuprivate::math::mul(src1,src2,dst);}

void mul(iu::LinearHostMemory_64f_C1& src1, iu::LinearHostMemory_64f_C1& src2, iu::LinearHostMemory_64f_C1& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory_64f_C2& src1, iu::LinearHostMemory_64f_C2& src2, iu::LinearHostMemory_64f_C2& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory_64f_C3& src1, iu::LinearHostMemory_64f_C3& src2, iu::LinearHostMemory_64f_C3& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory_64f_C4& src1, iu::LinearHostMemory_64f_C4& src2, iu::LinearHostMemory_64f_C4& dst) {iuprivate::math::mul(src1,src2,dst);}

void mul(iu::LinearHostMemory<double, 2>& src1, iu::LinearHostMemory<double, 2>& src2, iu::LinearHostMemory<double, 2>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory<double2, 2>& src1, iu::LinearHostMemory<double2, 2>& src2, iu::LinearHostMemory<double2, 2>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory<double3, 2>& src1, iu::LinearHostMemory<double3, 2>& src2, iu::LinearHostMemory<double3, 2>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory<double4, 2>& src1, iu::LinearHostMemory<double4, 2>& src2, iu::LinearHostMemory<double4, 2>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory<double, 3>& src1, iu::LinearHostMemory<double, 3>& src2, iu::LinearHostMemory<double, 3>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory<double2, 3>& src1, iu::LinearHostMemory<double2, 3>& src2, iu::LinearHostMemory<double2, 3>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory<double3, 3>& src1, iu::LinearHostMemory<double3, 3>& src2, iu::LinearHostMemory<double3, 3>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory<double4, 3>& src1, iu::LinearHostMemory<double4, 3>& src2, iu::LinearHostMemory<double4, 3>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory<double, 4>& src1, iu::LinearHostMemory<double, 4>& src2, iu::LinearHostMemory<double, 4>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory<double2, 4>& src1, iu::LinearHostMemory<double2, 4>& src2, iu::LinearHostMemory<double2, 4>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory<double3, 4>& src1, iu::LinearHostMemory<double3, 4>& src2, iu::LinearHostMemory<double3, 4>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory<double4, 4>& src1, iu::LinearHostMemory<double4, 4>& src2, iu::LinearHostMemory<double4, 4>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory<double, 5>& src1, iu::LinearHostMemory<double, 5>& src2, iu::LinearHostMemory<double, 5>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory<double2, 5>& src1, iu::LinearHostMemory<double2, 5>& src2, iu::LinearHostMemory<double2, 5>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory<double3, 5>& src1, iu::LinearHostMemory<double3, 5>& src2, iu::LinearHostMemory<double3, 5>& dst) {iuprivate::math::mul(src1,src2,dst);}
void mul(iu::LinearHostMemory<double4, 5>& src1, iu::LinearHostMemory<double4, 5>& src2, iu::LinearHostMemory<double4, 5>& dst) {iuprivate::math::mul(src1,src2,dst);}

// set value
void fill(iu::ImageGpu_32f_C1& dst, float value) {iuprivate::math::fill(dst,value);}
void fill(iu::ImageGpu_32f_C2& dst, float2 value) {iuprivate::math::fill(dst,value);}
void fill(iu::ImageGpu_32f_C4& dst, float4 value) {iuprivate::math::fill(dst,value);}
void fill(iu::ImageGpu_32u_C1& dst, unsigned int value) {iuprivate::math::fill(dst,value);}
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
void fill(iu::LinearDeviceMemory_32f_C2& dst, float2 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearDeviceMemory_32f_C3& dst, float3 value) {iuprivate::math::fill(dst,value);}

void fill(iu::LinearDeviceMemory<float, 2>& dst, float value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearDeviceMemory<float, 3>& dst, float value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearDeviceMemory<float, 4>& dst, float value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearDeviceMemory<float, 5>& dst, float value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearDeviceMemory<float2, 2>& dst, float2 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearDeviceMemory<float2, 3>& dst, float2 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearDeviceMemory<float2, 4>& dst, float2 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearDeviceMemory<float2, 5>& dst, float2 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearDeviceMemory<float3, 2>& dst, float3 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearDeviceMemory<float3, 3>& dst, float3 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearDeviceMemory<float3, 4>& dst, float3 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearDeviceMemory<float3, 5>& dst, float3 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearDeviceMemory<float4, 2>& dst, float4 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearDeviceMemory<float4, 3>& dst, float4 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearDeviceMemory<float4, 4>& dst, float4 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearDeviceMemory<float4, 5>& dst, float4 value) {iuprivate::math::fill(dst,value);}

void fill(iu::LinearHostMemory_32f_C1& dst, float value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearHostMemory_32f_C2& dst, float2 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearHostMemory_32f_C3& dst, float3 value) {iuprivate::math::fill(dst,value);}

void fill(iu::LinearHostMemory<float, 2>& dst, float value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearHostMemory<float, 3>& dst, float value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearHostMemory<float, 4>& dst, float value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearHostMemory<float, 5>& dst, float value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearHostMemory<float2, 2>& dst, float2 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearHostMemory<float2, 3>& dst, float2 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearHostMemory<float2, 4>& dst, float2 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearHostMemory<float2, 5>& dst, float2 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearHostMemory<float3, 2>& dst, float3 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearHostMemory<float3, 3>& dst, float3 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearHostMemory<float3, 4>& dst, float3 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearHostMemory<float3, 5>& dst, float3 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearHostMemory<float4, 2>& dst, float4 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearHostMemory<float4, 3>& dst, float4 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearHostMemory<float4, 4>& dst, float4 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearHostMemory<float4, 5>& dst, float4 value) {iuprivate::math::fill(dst,value);}

void fill(iu::VolumeGpu_32f_C1& dst, float value) {iuprivate::math::fill(dst,value);}
void fill(iu::VolumeGpu_32f_C2& dst, float2 value) {iuprivate::math::fill(dst,value);}
void fill(iu::VolumeGpu_32f_C3& dst, float3 value) {iuprivate::math::fill(dst,value);}

void fill(iu::VolumeCpu_32f_C1& dst, float value) {iuprivate::math::fill(dst,value);}
void fill(iu::VolumeCpu_32f_C2& dst, float2 value) {iuprivate::math::fill(dst,value);}
void fill(iu::VolumeCpu_32f_C3& dst, float3 value) {iuprivate::math::fill(dst,value);}

void fill(iu::ImageGpu_64f_C1& dst, double value) {iuprivate::math::fill(dst,value);}
void fill(iu::ImageGpu_64f_C2& dst, double2 value) {iuprivate::math::fill(dst,value);}
void fill(iu::ImageGpu_64f_C4& dst, double4 value) {iuprivate::math::fill(dst,value);}

void fill(iu::ImageCpu_64f_C1& dst, double value) {iuprivate::math::fill(dst,value);}
void fill(iu::ImageCpu_64f_C2& dst, double2 value) {iuprivate::math::fill(dst,value);}
void fill(iu::ImageCpu_64f_C4& dst, double4 value) {iuprivate::math::fill(dst,value);}

void fill(iu::LinearDeviceMemory_64f_C1& dst, double value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearDeviceMemory_64f_C2& dst, double2 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearDeviceMemory_64f_C3& dst, double3 value) {iuprivate::math::fill(dst,value);}

void fill(iu::LinearDeviceMemory<double, 2>& dst, double value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearDeviceMemory<double, 3>& dst, double value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearDeviceMemory<double, 4>& dst, double value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearDeviceMemory<double, 5>& dst, double value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearDeviceMemory<double2, 2>& dst, double2 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearDeviceMemory<double2, 3>& dst, double2 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearDeviceMemory<double2, 4>& dst, double2 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearDeviceMemory<double2, 5>& dst, double2 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearDeviceMemory<double3, 2>& dst, double3 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearDeviceMemory<double3, 3>& dst, double3 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearDeviceMemory<double3, 4>& dst, double3 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearDeviceMemory<double3, 5>& dst, double3 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearDeviceMemory<double4, 2>& dst, double4 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearDeviceMemory<double4, 3>& dst, double4 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearDeviceMemory<double4, 4>& dst, double4 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearDeviceMemory<double4, 5>& dst, double4 value) {iuprivate::math::fill(dst,value);}

void fill(iu::LinearHostMemory_64f_C1& dst, double value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearHostMemory_64f_C2& dst, double2 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearHostMemory_64f_C3& dst, double3 value) {iuprivate::math::fill(dst,value);}

void fill(iu::LinearHostMemory<double, 2>& dst, double value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearHostMemory<double, 3>& dst, double value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearHostMemory<double, 4>& dst, double value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearHostMemory<double, 5>& dst, double value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearHostMemory<double2, 2>& dst, double2 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearHostMemory<double2, 3>& dst, double2 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearHostMemory<double2, 4>& dst, double2 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearHostMemory<double2, 5>& dst, double2 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearHostMemory<double3, 2>& dst, double3 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearHostMemory<double3, 3>& dst, double3 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearHostMemory<double3, 4>& dst, double3 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearHostMemory<double3, 5>& dst, double3 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearHostMemory<double4, 2>& dst, double4 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearHostMemory<double4, 3>& dst, double4 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearHostMemory<double4, 4>& dst, double4 value) {iuprivate::math::fill(dst,value);}
void fill(iu::LinearHostMemory<double4, 5>& dst, double4 value) {iuprivate::math::fill(dst,value);}

void fill(iu::VolumeGpu_64f_C1& dst, double value) {iuprivate::math::fill(dst,value);}
void fill(iu::VolumeGpu_64f_C2& dst, double2 value) {iuprivate::math::fill(dst,value);}
void fill(iu::VolumeGpu_64f_C3& dst, double3 value) {iuprivate::math::fill(dst,value);}

void fill(iu::VolumeCpu_64f_C1& dst, double value) {iuprivate::math::fill(dst,value);}
void fill(iu::VolumeCpu_64f_C2& dst, double2 value) {iuprivate::math::fill(dst,value);}
void fill(iu::VolumeCpu_64f_C3& dst, double3 value) {iuprivate::math::fill(dst,value);}

// min-max
void minMax(iu::ImageGpu_32f_C1& src, float& minVal, float& maxVal) {iuprivate::math::minMax(src,minVal,maxVal);}
void minMax(iu::VolumeGpu_32f_C1& src, float& minVal, float& maxVal) {iuprivate::math::minMax(src,minVal,maxVal);}

void minMax(iu::ImageCpu_32f_C1& src, float& minVal, float& maxVal) {iuprivate::math::minMax(src,minVal,maxVal);}
void minMax(iu::VolumeCpu_32f_C1& src, float& minVal, float& maxVal) {iuprivate::math::minMax(src,minVal,maxVal);}

void minMax(iu::LinearDeviceMemory_32f_C1& src, float& minVal, float& maxVal, unsigned int& minIdx, unsigned int& maxIdx) {iuprivate::math::minMax(src,minVal,maxVal,minIdx,maxIdx);}
void minMax(iu::LinearHostMemory_32f_C1& src, float& minVal, float& maxVal, unsigned int& minIdx, unsigned int& maxIdx) {iuprivate::math::minMax(src,minVal,maxVal,minIdx,maxIdx);}
void minMax(iu::LinearDeviceMemory<float, 2>& src, float& minVal, float& maxVal, unsigned int& minIdx, unsigned int& maxIdx) {iuprivate::math::minMax(src,minVal,maxVal,minIdx,maxIdx);}
void minMax(iu::LinearHostMemory<float, 2>& src, float& minVal, float& maxVal, unsigned int& minIdx, unsigned int& maxIdx) {iuprivate::math::minMax(src,minVal,maxVal,minIdx,maxIdx);}
void minMax(iu::LinearDeviceMemory<float, 3>& src, float& minVal, float& maxVal, unsigned int& minIdx, unsigned int& maxIdx) {iuprivate::math::minMax(src,minVal,maxVal,minIdx,maxIdx);}
void minMax(iu::LinearHostMemory<float, 3>& src, float& minVal, float& maxVal, unsigned int& minIdx, unsigned int& maxIdx) {iuprivate::math::minMax(src,minVal,maxVal,minIdx,maxIdx);}
void minMax(iu::LinearDeviceMemory<float, 4>& src, float& minVal, float& maxVal, unsigned int& minIdx, unsigned int& maxIdx) {iuprivate::math::minMax(src,minVal,maxVal,minIdx,maxIdx);}
void minMax(iu::LinearHostMemory<float, 4>& src, float& minVal, float& maxVal, unsigned int& minIdx, unsigned int& maxIdx) {iuprivate::math::minMax(src,minVal,maxVal,minIdx,maxIdx);}
void minMax(iu::LinearDeviceMemory<float, 5>& src, float& minVal, float& maxVal, unsigned int& minIdx, unsigned int& maxIdx) {iuprivate::math::minMax(src,minVal,maxVal,minIdx,maxIdx);}
void minMax(iu::LinearHostMemory<float, 5>& src, float& minVal, float& maxVal, unsigned int& minIdx, unsigned int& maxIdx) {iuprivate::math::minMax(src,minVal,maxVal,minIdx,maxIdx);}

void minMax(iu::ImageGpu_64f_C1& src, double& minVal, double& maxVal) {iuprivate::math::minMax(src,minVal,maxVal);}
void minMax(iu::VolumeGpu_64f_C1& src, double& minVal, double& maxVal) {iuprivate::math::minMax(src,minVal,maxVal);}

void minMax(iu::ImageCpu_64f_C1& src, double& minVal, double& maxVal) {iuprivate::math::minMax(src,minVal,maxVal);}
void minMax(iu::VolumeCpu_64f_C1& src, double& minVal, double& maxVal) {iuprivate::math::minMax(src,minVal,maxVal);}

void minMax(iu::LinearDeviceMemory_64f_C1& src, double& minVal, double& maxVal, unsigned int& minIdx, unsigned int& maxIdx) {iuprivate::math::minMax(src,minVal,maxVal,minIdx,maxIdx);}
void minMax(iu::LinearHostMemory_64f_C1& src, double& minVal, double& maxVal, unsigned int& minIdx, unsigned int& maxIdx) {iuprivate::math::minMax(src,minVal,maxVal,minIdx,maxIdx);}

void minMax(iu::LinearDeviceMemory<double, 2>& src, double& minVal, double& maxVal, unsigned int& minIdx, unsigned int& maxIdx) {iuprivate::math::minMax(src,minVal,maxVal,minIdx,maxIdx);}
void minMax(iu::LinearHostMemory<double, 2>& src, double& minVal, double& maxVal, unsigned int& minIdx, unsigned int& maxIdx) {iuprivate::math::minMax(src,minVal,maxVal,minIdx,maxIdx);}
void minMax(iu::LinearDeviceMemory<double, 3>& src, double& minVal, double& maxVal, unsigned int& minIdx, unsigned int& maxIdx) {iuprivate::math::minMax(src,minVal,maxVal,minIdx,maxIdx);}
void minMax(iu::LinearHostMemory<double, 3>& src, double& minVal, double& maxVal, unsigned int& minIdx, unsigned int& maxIdx) {iuprivate::math::minMax(src,minVal,maxVal,minIdx,maxIdx);}
void minMax(iu::LinearDeviceMemory<double, 4>& src, double& minVal, double& maxVal, unsigned int& minIdx, unsigned int& maxIdx) {iuprivate::math::minMax(src,minVal,maxVal,minIdx,maxIdx);}
void minMax(iu::LinearHostMemory<double, 4>& src, double& minVal, double& maxVal, unsigned int& minIdx, unsigned int& maxIdx) {iuprivate::math::minMax(src,minVal,maxVal,minIdx,maxIdx);}
void minMax(iu::LinearDeviceMemory<double, 5>& src, double& minVal, double& maxVal, unsigned int& minIdx, unsigned int& maxIdx) {iuprivate::math::minMax(src,minVal,maxVal,minIdx,maxIdx);}
void minMax(iu::LinearHostMemory<double, 5>& src, double& minVal, double& maxVal, unsigned int& minIdx, unsigned int& maxIdx) {iuprivate::math::minMax(src,minVal,maxVal,minIdx,maxIdx);}

//sum
void summation(iu::ImageGpu_32f_C1& src, float& sum) {iuprivate::math::summation(src,0.f,sum);}
void summation(iu::VolumeGpu_32f_C1& src, float& sum) {iuprivate::math::summation(src,0.f,sum);}
void summation(iu::LinearDeviceMemory_32f_C1& src, float& sum) {iuprivate::math::summation(src,0.f,sum);}
void summation(iu::LinearDeviceMemory<float, 2>& src, float& sum) {iuprivate::math::summation(src,0.f,sum);}
void summation(iu::LinearDeviceMemory<float, 3>& src, float& sum) {iuprivate::math::summation(src,0.f,sum);}
void summation(iu::LinearDeviceMemory<float, 4>& src, float& sum) {iuprivate::math::summation(src,0.f,sum);}
void summation(iu::LinearDeviceMemory<float, 5>& src, float& sum) {iuprivate::math::summation(src,0.f,sum);}
void summation(iu::ImageGpu_32f_C2& src, float2& sum) {iuprivate::math::summation(src,iu::type_trait<float2>::make(0),sum);}
void summation(iu::VolumeGpu_32f_C2& src, float2& sum) {iuprivate::math::summation(src,iu::type_trait<float2>::make(0),sum);}
void summation(iu::LinearDeviceMemory_32f_C2& src, float2& sum) {iuprivate::math::summation(src,iu::type_trait<float2>::make(0),sum);}
void summation(iu::LinearDeviceMemory<float2, 2>& src, float2& sum) {iuprivate::math::summation(src,iu::type_trait<float2>::make(0),sum);}
void summation(iu::LinearDeviceMemory<float2, 3>& src, float2& sum) {iuprivate::math::summation(src,iu::type_trait<float2>::make(0),sum);}
void summation(iu::LinearDeviceMemory<float2, 4>& src, float2& sum) {iuprivate::math::summation(src,iu::type_trait<float2>::make(0),sum);}
void summation(iu::LinearDeviceMemory<float2, 5>& src, float2& sum) {iuprivate::math::summation(src,iu::type_trait<float2>::make(0),sum);}

void summation(iu::ImageCpu_32f_C1& src, float& sum) {iuprivate::math::summation(src,0.f,sum);}
void summation(iu::VolumeCpu_32f_C1& src, float& sum) {iuprivate::math::summation(src,0.f,sum);}
void summation(iu::LinearHostMemory_32f_C1& src, float& sum) {iuprivate::math::summation(src,0.f,sum);}
void summation(iu::LinearHostMemory<float, 2>& src, float& sum) {iuprivate::math::summation(src,0.f,sum);}
void summation(iu::LinearHostMemory<float, 3>& src, float& sum) {iuprivate::math::summation(src,0.f,sum);}
void summation(iu::LinearHostMemory<float, 4>& src, float& sum) {iuprivate::math::summation(src,0.f,sum);}
void summation(iu::LinearHostMemory<float, 5>& src, float& sum) {iuprivate::math::summation(src,0.f,sum);}
void summation(iu::ImageCpu_32f_C2& src, float2& sum) {iuprivate::math::summation(src,iu::type_trait<float2>::make(0),sum);}
void summation(iu::VolumeCpu_32f_C2& src, float2& sum) {iuprivate::math::summation(src,iu::type_trait<float2>::make(0),sum);}
void summation(iu::LinearHostMemory_32f_C2& src, float2& sum) {iuprivate::math::summation(src,iu::type_trait<float2>::make(0),sum);}
void summation(iu::LinearHostMemory<float2, 2>& src, float2& sum) {iuprivate::math::summation(src,iu::type_trait<float2>::make(0),sum);}
void summation(iu::LinearHostMemory<float2, 3>& src, float2& sum) {iuprivate::math::summation(src,iu::type_trait<float2>::make(0),sum);}
void summation(iu::LinearHostMemory<float2, 4>& src, float2& sum) {iuprivate::math::summation(src,iu::type_trait<float2>::make(0),sum);}
void summation(iu::LinearHostMemory<float2, 5>& src, float2& sum) {iuprivate::math::summation(src,iu::type_trait<float2>::make(0),sum);}

void summation(iu::ImageGpu_64f_C1& src, double& sum) {iuprivate::math::summation(src,0.0,sum);}
void summation(iu::VolumeGpu_64f_C1& src, double& sum) {iuprivate::math::summation(src,0.0,sum);}
void summation(iu::LinearDeviceMemory_64f_C1& src, double& sum) {iuprivate::math::summation(src,0.0,sum);}
void summation(iu::LinearDeviceMemory<double, 2>& src, double& sum) {iuprivate::math::summation(src,0.0,sum);}
void summation(iu::LinearDeviceMemory<double, 3>& src, double& sum) {iuprivate::math::summation(src,0.0,sum);}
void summation(iu::LinearDeviceMemory<double, 4>& src, double& sum) {iuprivate::math::summation(src,0.0,sum);}
void summation(iu::LinearDeviceMemory<double, 5>& src, double& sum) {iuprivate::math::summation(src,0.0,sum);}
void summation(iu::ImageGpu_64f_C2& src, double2& sum) {iuprivate::math::summation(src,iu::type_trait<double2>::make(0),sum);}
void summation(iu::VolumeGpu_64f_C2& src, double2& sum) {iuprivate::math::summation(src,iu::type_trait<double2>::make(0),sum);}
void summation(iu::LinearDeviceMemory_64f_C2& src, double2& sum) {iuprivate::math::summation(src,iu::type_trait<double2>::make(0),sum);}
void summation(iu::LinearDeviceMemory<double2, 2>& src, double2& sum) {iuprivate::math::summation(src,iu::type_trait<double2>::make(0),sum);}
void summation(iu::LinearDeviceMemory<double2, 3>& src, double2& sum) {iuprivate::math::summation(src,iu::type_trait<double2>::make(0),sum);}
void summation(iu::LinearDeviceMemory<double2, 4>& src, double2& sum) {iuprivate::math::summation(src,iu::type_trait<double2>::make(0),sum);}
void summation(iu::LinearDeviceMemory<double2, 5>& src, double2& sum) {iuprivate::math::summation(src,iu::type_trait<double2>::make(0),sum);}

void summation(iu::ImageCpu_64f_C1& src, double& sum) {iuprivate::math::summation(src,0.0,sum);}
void summation(iu::VolumeCpu_64f_C1& src, double& sum) {iuprivate::math::summation(src,0.0,sum);}
void summation(iu::LinearHostMemory_64f_C1& src, double& sum) {iuprivate::math::summation(src,0.0,sum);}
void summation(iu::LinearHostMemory<double, 2>& src, double& sum) {iuprivate::math::summation(src,0.0,sum);}
void summation(iu::LinearHostMemory<double, 3>& src, double& sum) {iuprivate::math::summation(src,0.0,sum);}
void summation(iu::LinearHostMemory<double, 4>& src, double& sum) {iuprivate::math::summation(src,0.0,sum);}
void summation(iu::LinearHostMemory<double, 5>& src, double& sum) {iuprivate::math::summation(src,0.0,sum);}
void summation(iu::ImageCpu_64f_C2& src, double2& sum) {iuprivate::math::summation(src,iu::type_trait<double2>::make(0),sum);}
void summation(iu::VolumeCpu_64f_C2& src, double2& sum) {iuprivate::math::summation(src,iu::type_trait<double2>::make(0),sum);}
void summation(iu::LinearHostMemory_64f_C2& src, double2& sum) {iuprivate::math::summation(src,iu::type_trait<double2>::make(0),sum);}
void summation(iu::LinearHostMemory<double2, 2>& src, double2& sum) {iuprivate::math::summation(src,iu::type_trait<double2>::make(0),sum);}
void summation(iu::LinearHostMemory<double2, 3>& src, double2& sum) {iuprivate::math::summation(src,iu::type_trait<double2>::make(0),sum);}
void summation(iu::LinearHostMemory<double2, 4>& src, double2& sum) {iuprivate::math::summation(src,iu::type_trait<double2>::make(0),sum);}
void summation(iu::LinearHostMemory<double2, 5>& src, double2& sum) {iuprivate::math::summation(src,iu::type_trait<double2>::make(0),sum);}

// Dot product
void dotProduct(iu::ImageGpu_32f_C1& src1, iu::ImageGpu_32f_C1& src2, float& result)
{
  iu::ImageGpu_32f_C1 dst(src1.size());
  iuprivate::math::mul(src1, src2, dst);
  iuprivate::math::summation(dst,0.0f,result);
}
void dotProduct(iu::VolumeGpu_32f_C1& src1, iu::VolumeGpu_32f_C1& src2, float& result)
{
  iu::VolumeGpu_32f_C1 dst(src1.size());
  iuprivate::math::mul(src1, src2, dst);
  iuprivate::math::summation(dst,0.0f,result);
}
void dotProduct(iu::LinearDeviceMemory_32f_C1& src1, iu::LinearDeviceMemory_32f_C1& src2, float& result)
{
  iu::LinearDeviceMemory_32f_C1 dst(src1.size());
  iuprivate::math::mul(src1, src2, dst);
  iuprivate::math::summation(dst,0.0f,result);
}
void dotProduct(iu::LinearDeviceMemory<float, 2>& src1, iu::LinearDeviceMemory<float, 2>& src2, float& result)
{
  iu::LinearDeviceMemory<float, 2> dst(src1.size());
  iuprivate::math::mul(src1, src2, dst);
  iuprivate::math::summation(dst,0.0f,result);
}
void dotProduct(iu::LinearDeviceMemory<float, 3>& src1, iu::LinearDeviceMemory<float, 3>& src2, float& result)
{
  iu::LinearDeviceMemory<float, 3> dst(src1.size());
  iuprivate::math::mul(src1, src2, dst);
  iuprivate::math::summation(dst,0.0f,result);
}
void dotProduct(iu::LinearDeviceMemory<float, 4>& src1, iu::LinearDeviceMemory<float, 4>& src2, float& result)
{
  iu::LinearDeviceMemory<float, 4> dst(src1.size());
  iuprivate::math::mul(src1, src2, dst);
  iuprivate::math::summation(dst,0.0f,result);
}
void dotProduct(iu::LinearDeviceMemory<float, 5>& src1, iu::LinearDeviceMemory<float, 5>& src2, float& result)
{
  iu::LinearDeviceMemory<float, 5> dst(src1.size());
  iuprivate::math::mul(src1, src2, dst);
  iuprivate::math::summation(dst,0.0f,result);
}

void dotProduct(iu::ImageCpu_32f_C1& src1, iu::ImageCpu_32f_C1& src2, float& result)
{
  iu::ImageCpu_32f_C1 dst(src1.size());
  iuprivate::math::mul(src1, src2, dst);
  iuprivate::math::summation(dst,0.0f,result);
}
void dotProduct(iu::VolumeCpu_32f_C1& src1, iu::VolumeCpu_32f_C1& src2, float& result)
{
  iu::VolumeCpu_32f_C1 dst(src1.size());
  iuprivate::math::mul(src1, src2, dst);
  iuprivate::math::summation(dst,0.0f,result);
}
void dotProduct(iu::LinearHostMemory_32f_C1& src1, iu::LinearHostMemory_32f_C1& src2, float& result)
{
  iu::LinearHostMemory_32f_C1 dst(src1.size());
  iuprivate::math::mul(src1, src2, dst);
  iuprivate::math::summation(dst,0.0f,result);
}
void dotProduct(iu::LinearHostMemory<float, 2>& src1, iu::LinearHostMemory<float, 2>& src2, float& result)
{
  iu::LinearHostMemory<float, 2> dst(src1.size());
  iuprivate::math::mul(src1, src2, dst);
  iuprivate::math::summation(dst,0.0f,result);
}
void dotProduct(iu::LinearHostMemory<float, 3>& src1, iu::LinearHostMemory<float, 3>& src2, float& result)
{
  iu::LinearHostMemory<float, 3> dst(src1.size());
  iuprivate::math::mul(src1, src2, dst);
  iuprivate::math::summation(dst,0.0f,result);
}
void dotProduct(iu::LinearHostMemory<float, 4>& src1, iu::LinearHostMemory<float, 4>& src2, float& result)
{
  iu::LinearHostMemory<float, 4> dst(src1.size());
  iuprivate::math::mul(src1, src2, dst);
  iuprivate::math::summation(dst,0.0f,result);
}
void dotProduct(iu::LinearHostMemory<float, 5>& src1, iu::LinearHostMemory<float, 5>& src2, float& result)
{
  iu::LinearHostMemory<float, 5> dst(src1.size());
  iuprivate::math::mul(src1, src2, dst);
  iuprivate::math::summation(dst,0.0f,result);
}

void dotProduct(iu::ImageGpu_64f_C1& src1, iu::ImageGpu_64f_C1& src2, double& result)
{
  iu::ImageGpu_64f_C1 dst(src1.size());
  iuprivate::math::mul(src1, src2, dst);
  iuprivate::math::summation(dst,0.0,result);
}
void dotProduct(iu::VolumeGpu_64f_C1& src1, iu::VolumeGpu_64f_C1& src2, double& result)
{
  iu::VolumeGpu_64f_C1 dst(src1.size());
  iuprivate::math::mul(src1, src2, dst);
  iuprivate::math::summation(dst,0.0,result);
}
void dotProduct(iu::LinearDeviceMemory_64f_C1& src1, iu::LinearDeviceMemory_64f_C1& src2, double& result)
{
  iu::LinearDeviceMemory_64f_C1 dst(src1.size());
  iuprivate::math::mul(src1, src2, dst);
  iuprivate::math::summation(dst,0.0,result);
}
void dotProduct(iu::LinearDeviceMemory<double, 2>& src1, iu::LinearDeviceMemory<double, 2>& src2, double& result)
{
  iu::LinearDeviceMemory<double, 2> dst(src1.size());
  iuprivate::math::mul(src1, src2, dst);
  iuprivate::math::summation(dst,0.0,result);
}
void dotProduct(iu::LinearDeviceMemory<double, 3>& src1, iu::LinearDeviceMemory<double, 3>& src2, double& result)
{
  iu::LinearDeviceMemory<double, 3> dst(src1.size());
  iuprivate::math::mul(src1, src2, dst);
  iuprivate::math::summation(dst,0.0,result);
}
void dotProduct(iu::LinearDeviceMemory<double, 4>& src1, iu::LinearDeviceMemory<double, 4>& src2, double& result)
{
  iu::LinearDeviceMemory<double, 4> dst(src1.size());
  iuprivate::math::mul(src1, src2, dst);
  iuprivate::math::summation(dst,0.0,result);
}
void dotProduct(iu::LinearDeviceMemory<double, 5>& src1, iu::LinearDeviceMemory<double, 5>& src2, double& result)
{
  iu::LinearDeviceMemory<double, 5> dst(src1.size());
  iuprivate::math::mul(src1, src2, dst);
  iuprivate::math::summation(dst,0.0,result);
}

void dotProduct(iu::ImageCpu_64f_C1& src1, iu::ImageCpu_64f_C1& src2, double& result)
{
  iu::ImageCpu_64f_C1 dst(src1.size());
  iuprivate::math::mul(src1, src2, dst);
  iuprivate::math::summation(dst,0.0,result);
}
void dotProduct(iu::VolumeCpu_64f_C1& src1, iu::VolumeCpu_64f_C1& src2, double& result)
{
  iu::VolumeCpu_64f_C1 dst(src1.size());
  iuprivate::math::mul(src1, src2, dst);
  iuprivate::math::summation(dst,0.0,result);
}
void dotProduct(iu::LinearHostMemory_64f_C1& src1, iu::LinearHostMemory_64f_C1& src2, double& result)
{
  iu::LinearHostMemory_64f_C1 dst(src1.size());
  iuprivate::math::mul(src1, src2, dst);
  iuprivate::math::summation(dst,0.0,result);
}
void dotProduct(iu::LinearHostMemory<double, 2>& src1, iu::LinearHostMemory<double, 2>& src2, double& result)
{
  iu::LinearHostMemory<double, 2> dst(src1.size());
  iuprivate::math::mul(src1, src2, dst);
  iuprivate::math::summation(dst,0.0,result);
}
void dotProduct(iu::LinearHostMemory<double, 3>& src1, iu::LinearHostMemory<double, 3>& src2, double& result)
{
  iu::LinearHostMemory<double, 3> dst(src1.size());
  iuprivate::math::mul(src1, src2, dst);
  iuprivate::math::summation(dst,0.0,result);
}
void dotProduct(iu::LinearHostMemory<double, 4>& src1, iu::LinearHostMemory<double, 4>& src2, double& result)
{
  iu::LinearHostMemory<double, 4> dst(src1.size());
  iuprivate::math::mul(src1, src2, dst);
  iuprivate::math::summation(dst,0.0,result);
}
void dotProduct(iu::LinearHostMemory<double, 5>& src1, iu::LinearHostMemory<double, 5>& src2, double& result)
{
  iu::LinearHostMemory<double, 5> dst(src1.size());
  iuprivate::math::mul(src1, src2, dst);
  iuprivate::math::summation(dst,0.0,result);
}

// L1-norm
void normDiffL1(iu::ImageGpu_32f_C1& src, iu::ImageGpu_32f_C1& ref, float& norm) {iuprivate::math::normDiffL1(src,ref,norm);}
void normDiffL1(iu::ImageGpu_32f_C1& src, float& ref, float& norm) {iuprivate::math::normDiffL1(src,ref,norm);}
void normDiffL1(iu::VolumeGpu_32f_C1& src, iu::VolumeGpu_32f_C1& ref, float& norm) {iuprivate::math::normDiffL1(src,ref,norm);}
void normDiffL1(iu::VolumeGpu_32f_C1& src, float& ref, float& norm) {iuprivate::math::normDiffL1(src,ref,norm);}

void normDiffL1(iu::ImageCpu_32f_C1& src, iu::ImageCpu_32f_C1& ref, float& norm) {iuprivate::math::normDiffL1(src,ref,norm);}
void normDiffL1(iu::ImageCpu_32f_C1& src, float& ref, float& norm) {iuprivate::math::normDiffL1(src,ref,norm);}
void normDiffL1(iu::VolumeCpu_32f_C1& src, iu::VolumeCpu_32f_C1& ref, float& norm) {iuprivate::math::normDiffL1(src,ref,norm);}
void normDiffL1(iu::VolumeCpu_32f_C1& src, float& ref, float& norm) {iuprivate::math::normDiffL1(src,ref,norm);}

void normDiffL1(iu::ImageGpu_64f_C1& src, iu::ImageGpu_64f_C1& ref, double& norm) {iuprivate::math::normDiffL1(src,ref,norm);}
void normDiffL1(iu::ImageGpu_64f_C1& src, double& ref, double& norm) {iuprivate::math::normDiffL1(src,ref,norm);}
void normDiffL1(iu::VolumeGpu_64f_C1& src, iu::VolumeGpu_64f_C1& ref, double& norm) {iuprivate::math::normDiffL1(src,ref,norm);}
void normDiffL1(iu::VolumeGpu_64f_C1& src, double& ref, double& norm) {iuprivate::math::normDiffL1(src,ref,norm);}

void normDiffL1(iu::ImageCpu_64f_C1& src, iu::ImageCpu_64f_C1& ref, double& norm) {iuprivate::math::normDiffL1(src,ref,norm);}
void normDiffL1(iu::ImageCpu_64f_C1& src, double& ref, double& norm) {iuprivate::math::normDiffL1(src,ref,norm);}
void normDiffL1(iu::VolumeCpu_64f_C1& src, iu::VolumeCpu_64f_C1& ref, double& norm) {iuprivate::math::normDiffL1(src,ref,norm);}
void normDiffL1(iu::VolumeCpu_64f_C1& src, double& ref, double& norm) {iuprivate::math::normDiffL1(src,ref,norm);}


// L2-norm
void normDiffL2(iu::ImageGpu_32f_C1& src, iu::ImageGpu_32f_C1& ref, float& norm) {iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::VolumeGpu_32f_C1& src, iu::VolumeGpu_32f_C1& ref, float& norm) {iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearDeviceMemory_32f_C1& src, iu::LinearDeviceMemory_32f_C1& ref, float& norm){iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearDeviceMemory<float, 2>& src, iu::LinearDeviceMemory<float, 2>& ref, float& norm){iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearDeviceMemory<float2, 2>& src, iu::LinearDeviceMemory<float2, 2>& ref, float& norm){iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearDeviceMemory<float, 3>& src, iu::LinearDeviceMemory<float, 3>& ref, float& norm){iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearDeviceMemory<float2, 3>& src, iu::LinearDeviceMemory<float2, 3>& ref, float& norm){iuprivate::math::normDiffL2(src,ref,norm);}

void normDiffL2(iu::ImageGpu_32f_C1& src, float& ref, float& norm) {iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::VolumeGpu_32f_C1& src, float& ref, float& norm) {iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearDeviceMemory_32f_C1& src, float& ref, float& norm){iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearDeviceMemory<float, 2>& src, float& ref, float& norm){iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearDeviceMemory<float2, 2>& src, float2& ref, float& norm){iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearDeviceMemory<float, 3>& src, float& ref, float& norm){iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearDeviceMemory<float2, 3>& src, float2& ref, float& norm){iuprivate::math::normDiffL2(src,ref,norm);}

void normDiffL2(iu::ImageCpu_32f_C1& src, iu::ImageCpu_32f_C1& ref, float& norm) {iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::VolumeCpu_32f_C1& src, iu::VolumeCpu_32f_C1& ref, float& norm) {iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearHostMemory_32f_C1& src, iu::LinearHostMemory_32f_C1& ref, float& norm){iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearHostMemory<float, 2>& src, iu::LinearHostMemory<float, 2>& ref, float& norm){iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearHostMemory<float2, 2>& src, iu::LinearHostMemory<float2, 2>& ref, float& norm){iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearHostMemory<float, 3>& src, iu::LinearHostMemory<float, 3>& ref, float& norm){iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearHostMemory<float2, 3>& src, iu::LinearHostMemory<float2, 3>& ref, float& norm){iuprivate::math::normDiffL2(src,ref,norm);}

void normDiffL2(iu::ImageCpu_32f_C1& src, float& ref, float& norm) {iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::VolumeCpu_32f_C1& src, float& ref, float& norm) {iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearHostMemory_32f_C1& src, float& ref, float& norm){iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearHostMemory<float, 2>& src, float& ref, float& norm){iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearHostMemory<float2, 2>& src, float2& ref, float& norm){iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearHostMemory<float, 3>& src, float& ref, float& norm){iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearHostMemory<float2, 3>& src, float2& ref, float& norm){iuprivate::math::normDiffL2(src,ref,norm);}

void normDiffL2(iu::ImageGpu_64f_C1& src, iu::ImageGpu_64f_C1& ref, double& norm) {iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::VolumeGpu_64f_C1& src, iu::VolumeGpu_64f_C1& ref, double& norm) {iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearDeviceMemory_64f_C1& src, iu::LinearDeviceMemory_64f_C1& ref, double& norm){iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearDeviceMemory<double, 2>& src, iu::LinearDeviceMemory<double, 2>& ref, double& norm){iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearDeviceMemory<double2, 2>& src, iu::LinearDeviceMemory<double2, 2>& ref, double& norm){iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearDeviceMemory<double, 3>& src, iu::LinearDeviceMemory<double, 3>& ref, double& norm){iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearDeviceMemory<double2, 3>& src, iu::LinearDeviceMemory<double2, 3>& ref, double& norm){iuprivate::math::normDiffL2(src,ref,norm);}

void normDiffL2(iu::ImageGpu_64f_C1& src, double& ref, double& norm) {iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::VolumeGpu_64f_C1& src, double& ref, double& norm) {iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearDeviceMemory_64f_C1& src, double& ref, double& norm){iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearDeviceMemory<double, 2>& src, double& ref, double& norm){iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearDeviceMemory<double2, 2>& src, double2& ref, double& norm){iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearDeviceMemory<double, 3>& src, double& ref, double& norm){iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearDeviceMemory<double2, 3>& src, double2& ref, double& norm){iuprivate::math::normDiffL2(src,ref,norm);}

void normDiffL2(iu::ImageCpu_64f_C1& src, iu::ImageCpu_64f_C1& ref, double& norm) {iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::VolumeCpu_64f_C1& src, iu::VolumeCpu_64f_C1& ref, double& norm) {iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearHostMemory_64f_C1& src, iu::LinearHostMemory_64f_C1& ref, double& norm){iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearHostMemory<double, 2>& src, iu::LinearHostMemory<double, 2>& ref, double& norm){iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearHostMemory<double2, 2>& src, iu::LinearHostMemory<double2, 2>& ref, double& norm){iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearHostMemory<double, 3>& src, iu::LinearHostMemory<double, 3>& ref, double& norm){iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearHostMemory<double2, 3>& src, iu::LinearHostMemory<double2, 3>& ref, double& norm){iuprivate::math::normDiffL2(src,ref,norm);}

void normDiffL2(iu::ImageCpu_64f_C1& src, double& ref, double& norm) {iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::VolumeCpu_64f_C1& src, double& ref, double& norm) {iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearHostMemory_64f_C1& src, double& ref, double& norm){iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearHostMemory<double, 2>& src, double& ref, double& norm){iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearHostMemory<double2, 2>& src, double2& ref, double& norm){iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearHostMemory<double, 3>& src, double& ref, double& norm){iuprivate::math::normDiffL2(src,ref,norm);}
void normDiffL2(iu::LinearHostMemory<double2, 3>& src, double2& ref, double& norm){iuprivate::math::normDiffL2(src,ref,norm);}
// MSE
void mse(iu::ImageGpu_32f_C1& src, iu::ImageGpu_32f_C1& ref, float& mse) {iuprivate::math::mse(src,ref,mse);}
void mse(iu::VolumeGpu_32f_C1& src, iu::VolumeGpu_32f_C1& ref, float& mse) {iuprivate::math::mse(src,ref,mse);}

void mse(iu::ImageCpu_32f_C1& src, iu::ImageCpu_32f_C1& ref, float& mse) {iuprivate::math::mse(src,ref,mse);}
void mse(iu::VolumeCpu_32f_C1& src, iu::VolumeCpu_32f_C1& ref, float& mse) {iuprivate::math::mse(src,ref,mse);}

void mse(iu::ImageGpu_64f_C1& src, iu::ImageGpu_64f_C1& ref, double& mse) {iuprivate::math::mse(src,ref,mse);}
void mse(iu::VolumeGpu_64f_C1& src, iu::VolumeGpu_64f_C1& ref, double& mse) {iuprivate::math::mse(src,ref,mse);}

void mse(iu::ImageCpu_64f_C1& src, iu::ImageCpu_64f_C1& ref, double& mse) {iuprivate::math::mse(src,ref,mse);}
void mse(iu::VolumeCpu_64f_C1& src, iu::VolumeCpu_64f_C1& ref, double& mse) {iuprivate::math::mse(src,ref,mse);}

// split planes
void splitPlanes(iu::VolumeCpu_32f_C2& src, iu::VolumeCpu_32f_C1& dst1, iu::VolumeCpu_32f_C1& dst2) {iuprivate::math::splitPlanes(src, dst1, dst2);}
void splitPlanes(iu::VolumeGpu_32f_C2& src, iu::VolumeGpu_32f_C1& dst1, iu::VolumeGpu_32f_C1& dst2) {iuprivate::math::splitPlanes(src, dst1, dst2);}
void splitPlanes(iu::ImageCpu_32f_C2& src, iu::ImageCpu_32f_C1& dst1, iu::ImageCpu_32f_C1& dst2) {iuprivate::math::splitPlanes(src, dst1, dst2);}
void splitPlanes(iu::ImageGpu_32f_C2& src, iu::ImageGpu_32f_C1& dst1, iu::ImageGpu_32f_C1& dst2) {iuprivate::math::splitPlanes(src, dst1, dst2);}
void splitPlanes(iu::LinearHostMemory_32f_C2& src, iu::LinearHostMemory_32f_C1& dst1, iu::LinearHostMemory_32f_C1& dst2) {iuprivate::math::splitPlanes(src, dst1, dst2);}
void splitPlanes(iu::LinearDeviceMemory_32f_C2& src, iu::LinearDeviceMemory_32f_C1& dst1, iu::LinearDeviceMemory_32f_C1& dst2) {iuprivate::math::splitPlanes(src, dst1, dst2);}
void splitPlanes(iu::LinearHostMemory<float2, 2>& src, iu::LinearHostMemory<float, 2>& dst1, iu::LinearHostMemory<float, 2>& dst2) {iuprivate::math::splitPlanes(src, dst1, dst2);}
void splitPlanes(iu::LinearDeviceMemory<float2, 2>& src, iu::LinearDeviceMemory<float, 2>& dst1, iu::LinearDeviceMemory<float, 2>& dst2) {iuprivate::math::splitPlanes(src, dst1, dst2);}
void splitPlanes(iu::LinearHostMemory<float2, 3>& src, iu::LinearHostMemory<float, 3>& dst1, iu::LinearHostMemory<float, 3>& dst2) {iuprivate::math::splitPlanes(src, dst1, dst2);}
void splitPlanes(iu::LinearDeviceMemory<float2, 3>& src, iu::LinearDeviceMemory<float, 3>& dst1, iu::LinearDeviceMemory<float, 3>& dst2) {iuprivate::math::splitPlanes(src, dst1, dst2);}
void splitPlanes(iu::LinearHostMemory<float2, 4>& src, iu::LinearHostMemory<float, 4>& dst1, iu::LinearHostMemory<float, 4>& dst2) {iuprivate::math::splitPlanes(src, dst1, dst2);}
void splitPlanes(iu::LinearDeviceMemory<float2, 4>& src, iu::LinearDeviceMemory<float, 4>& dst1, iu::LinearDeviceMemory<float, 4>& dst2) {iuprivate::math::splitPlanes(src, dst1, dst2);}
void splitPlanes(iu::LinearHostMemory<float2, 5>& src, iu::LinearHostMemory<float, 5>& dst1, iu::LinearHostMemory<float, 5>& dst2) {iuprivate::math::splitPlanes(src, dst1, dst2);}
void splitPlanes(iu::LinearDeviceMemory<float2, 5>& src, iu::LinearDeviceMemory<float, 5>& dst1, iu::LinearDeviceMemory<float, 5>& dst2) {iuprivate::math::splitPlanes(src, dst1, dst2);}

void splitPlanes(iu::VolumeCpu_32f_C3& src, iu::VolumeCpu_32f_C1& dst1, iu::VolumeCpu_32f_C1& dst2, iu::VolumeCpu_32f_C1& dst3){iuprivate::math::splitPlanes(src, dst1, dst2, dst3);}
void splitPlanes(iu::VolumeGpu_32f_C3& src, iu::VolumeGpu_32f_C1& dst1, iu::VolumeGpu_32f_C1& dst2, iu::VolumeGpu_32f_C1& dst3){iuprivate::math::splitPlanes(src, dst1, dst2, dst3);}
void splitPlanes(iu::ImageCpu_32f_C3& src, iu::ImageCpu_32f_C1& dst1, iu::ImageCpu_32f_C1& dst2, iu::ImageCpu_32f_C1& dst3){iuprivate::math::splitPlanes(src, dst1, dst2, dst3);}
void splitPlanes(iu::ImageGpu_32f_C3& src, iu::ImageGpu_32f_C1& dst1, iu::ImageGpu_32f_C1& dst2, iu::ImageGpu_32f_C1& dst3){iuprivate::math::splitPlanes(src, dst1, dst2, dst3);}
void splitPlanes(iu::LinearHostMemory_32f_C3& src, iu::LinearHostMemory_32f_C1& dst1, iu::LinearHostMemory_32f_C1& dst2, iu::LinearHostMemory_32f_C1& dst3){iuprivate::math::splitPlanes(src, dst1, dst2, dst3);}
void splitPlanes(iu::LinearDeviceMemory_32f_C3& src, iu::LinearDeviceMemory_32f_C1& dst1, iu::LinearDeviceMemory_32f_C1& dst2, iu::LinearDeviceMemory_32f_C1& dst3){iuprivate::math::splitPlanes(src, dst1, dst2, dst3);}
void splitPlanes(iu::LinearHostMemory<float3, 2>& src, iu::LinearHostMemory<float, 2>& dst1, iu::LinearHostMemory<float, 2>& dst2, iu::LinearHostMemory<float, 2>& dst3){iuprivate::math::splitPlanes(src, dst1, dst2, dst3);}
void splitPlanes(iu::LinearDeviceMemory<float3, 2>& src, iu::LinearDeviceMemory<float, 2>& dst1, iu::LinearDeviceMemory<float, 2>& dst2, iu::LinearDeviceMemory<float, 2>& dst3){iuprivate::math::splitPlanes(src, dst1, dst2, dst3);}
void splitPlanes(iu::LinearHostMemory<float3, 3>& src, iu::LinearHostMemory<float, 3>& dst1, iu::LinearHostMemory<float, 3>& dst2, iu::LinearHostMemory<float, 3>& dst3){iuprivate::math::splitPlanes(src, dst1, dst2, dst3);}
void splitPlanes(iu::LinearDeviceMemory<float3, 3>& src, iu::LinearDeviceMemory<float, 3>& dst1, iu::LinearDeviceMemory<float, 3>& dst2, iu::LinearDeviceMemory<float, 3>& dst3){iuprivate::math::splitPlanes(src, dst1, dst2, dst3);}
void splitPlanes(iu::LinearHostMemory<float3, 4>& src, iu::LinearHostMemory<float, 4>& dst1, iu::LinearHostMemory<float, 4>& dst2, iu::LinearHostMemory<float, 4>& dst3){iuprivate::math::splitPlanes(src, dst1, dst2, dst3);}
void splitPlanes(iu::LinearDeviceMemory<float3, 4>& src, iu::LinearDeviceMemory<float, 4>& dst1, iu::LinearDeviceMemory<float, 4>& dst2, iu::LinearDeviceMemory<float, 4>& dst3){iuprivate::math::splitPlanes(src, dst1, dst2, dst3);}
void splitPlanes(iu::LinearHostMemory<float3, 5>& src, iu::LinearHostMemory<float, 5>& dst1, iu::LinearHostMemory<float, 5>& dst2, iu::LinearHostMemory<float, 5>& dst3){iuprivate::math::splitPlanes(src, dst1, dst2, dst3);}
void splitPlanes(iu::LinearDeviceMemory<float3, 5>& src, iu::LinearDeviceMemory<float, 5>& dst1, iu::LinearDeviceMemory<float, 5>& dst2, iu::LinearDeviceMemory<float, 5>& dst3){iuprivate::math::splitPlanes(src, dst1, dst2, dst3);}

void splitPlanes(iu::VolumeCpu_64f_C2& src, iu::VolumeCpu_64f_C1& dst1, iu::VolumeCpu_64f_C1& dst2) {iuprivate::math::splitPlanes(src, dst1, dst2);}
void splitPlanes(iu::VolumeGpu_64f_C2& src, iu::VolumeGpu_64f_C1& dst1, iu::VolumeGpu_64f_C1& dst2) {iuprivate::math::splitPlanes(src, dst1, dst2);}
void splitPlanes(iu::ImageCpu_64f_C2& src, iu::ImageCpu_64f_C1& dst1, iu::ImageCpu_64f_C1& dst2) {iuprivate::math::splitPlanes(src, dst1, dst2);}
void splitPlanes(iu::ImageGpu_64f_C2& src, iu::ImageGpu_64f_C1& dst1, iu::ImageGpu_64f_C1& dst2) {iuprivate::math::splitPlanes(src, dst1, dst2);}
void splitPlanes(iu::LinearHostMemory_64f_C2& src, iu::LinearHostMemory_64f_C1& dst1, iu::LinearHostMemory_64f_C1& dst2) {iuprivate::math::splitPlanes(src, dst1, dst2);}
void splitPlanes(iu::LinearDeviceMemory_64f_C2& src, iu::LinearDeviceMemory_64f_C1& dst1, iu::LinearDeviceMemory_64f_C1& dst2) {iuprivate::math::splitPlanes(src, dst1, dst2);}
void splitPlanes(iu::LinearHostMemory<double2, 2>& src, iu::LinearHostMemory<double, 2>& dst1, iu::LinearHostMemory<double, 2>& dst2) {iuprivate::math::splitPlanes(src, dst1, dst2);}
void splitPlanes(iu::LinearDeviceMemory<double2, 2>& src, iu::LinearDeviceMemory<double, 2>& dst1, iu::LinearDeviceMemory<double, 2>& dst2) {iuprivate::math::splitPlanes(src, dst1, dst2);}
void splitPlanes(iu::LinearHostMemory<double2, 3>& src, iu::LinearHostMemory<double, 3>& dst1, iu::LinearHostMemory<double, 3>& dst2) {iuprivate::math::splitPlanes(src, dst1, dst2);}
void splitPlanes(iu::LinearDeviceMemory<double2, 3>& src, iu::LinearDeviceMemory<double, 3>& dst1, iu::LinearDeviceMemory<double, 3>& dst2) {iuprivate::math::splitPlanes(src, dst1, dst2);}
void splitPlanes(iu::LinearHostMemory<double2, 4>& src, iu::LinearHostMemory<double, 4>& dst1, iu::LinearHostMemory<double, 4>& dst2) {iuprivate::math::splitPlanes(src, dst1, dst2);}
void splitPlanes(iu::LinearDeviceMemory<double2, 4>& src, iu::LinearDeviceMemory<double, 4>& dst1, iu::LinearDeviceMemory<double, 4>& dst2) {iuprivate::math::splitPlanes(src, dst1, dst2);}
void splitPlanes(iu::LinearHostMemory<double2, 5>& src, iu::LinearHostMemory<double, 5>& dst1, iu::LinearHostMemory<double, 5>& dst2) {iuprivate::math::splitPlanes(src, dst1, dst2);}
void splitPlanes(iu::LinearDeviceMemory<double2, 5>& src, iu::LinearDeviceMemory<double, 5>& dst1, iu::LinearDeviceMemory<double, 5>& dst2) {iuprivate::math::splitPlanes(src, dst1, dst2);}

void splitPlanes(iu::VolumeCpu_64f_C3& src, iu::VolumeCpu_64f_C1& dst1, iu::VolumeCpu_64f_C1& dst2, iu::VolumeCpu_64f_C1& dst3){iuprivate::math::splitPlanes(src, dst1, dst2, dst3);}
void splitPlanes(iu::VolumeGpu_64f_C3& src, iu::VolumeGpu_64f_C1& dst1, iu::VolumeGpu_64f_C1& dst2, iu::VolumeGpu_64f_C1& dst3){iuprivate::math::splitPlanes(src, dst1, dst2, dst3);}
void splitPlanes(iu::ImageCpu_64f_C3& src, iu::ImageCpu_64f_C1& dst1, iu::ImageCpu_64f_C1& dst2, iu::ImageCpu_64f_C1& dst3){iuprivate::math::splitPlanes(src, dst1, dst2, dst3);}
void splitPlanes(iu::ImageGpu_64f_C3& src, iu::ImageGpu_64f_C1& dst1, iu::ImageGpu_64f_C1& dst2, iu::ImageGpu_64f_C1& dst3){iuprivate::math::splitPlanes(src, dst1, dst2, dst3);}
void splitPlanes(iu::LinearHostMemory_64f_C3& src, iu::LinearHostMemory_64f_C1& dst1, iu::LinearHostMemory_64f_C1& dst2, iu::LinearHostMemory_64f_C1& dst3){iuprivate::math::splitPlanes(src, dst1, dst2, dst3);}
void splitPlanes(iu::LinearDeviceMemory_64f_C3& src, iu::LinearDeviceMemory_64f_C1& dst1, iu::LinearDeviceMemory_64f_C1& dst2, iu::LinearDeviceMemory_64f_C1& dst3){iuprivate::math::splitPlanes(src, dst1, dst2, dst3);}
void splitPlanes(iu::LinearHostMemory<double3, 2>& src, iu::LinearHostMemory<double, 2>& dst1, iu::LinearHostMemory<double, 2>& dst2, iu::LinearHostMemory<double, 2>& dst3){iuprivate::math::splitPlanes(src, dst1, dst2, dst3);}
void splitPlanes(iu::LinearDeviceMemory<double3, 2>& src, iu::LinearDeviceMemory<double, 2>& dst1, iu::LinearDeviceMemory<double, 2>& dst2, iu::LinearDeviceMemory<double, 2>& dst3){iuprivate::math::splitPlanes(src, dst1, dst2, dst3);}
void splitPlanes(iu::LinearHostMemory<double3, 3>& src, iu::LinearHostMemory<double, 3>& dst1, iu::LinearHostMemory<double, 3>& dst2, iu::LinearHostMemory<double, 3>& dst3){iuprivate::math::splitPlanes(src, dst1, dst2, dst3);}
void splitPlanes(iu::LinearDeviceMemory<double3, 3>& src, iu::LinearDeviceMemory<double, 3>& dst1, iu::LinearDeviceMemory<double, 3>& dst2, iu::LinearDeviceMemory<double, 3>& dst3){iuprivate::math::splitPlanes(src, dst1, dst2, dst3);}
void splitPlanes(iu::LinearHostMemory<double3, 4>& src, iu::LinearHostMemory<double, 4>& dst1, iu::LinearHostMemory<double, 4>& dst2, iu::LinearHostMemory<double, 4>& dst3){iuprivate::math::splitPlanes(src, dst1, dst2, dst3);}
void splitPlanes(iu::LinearDeviceMemory<double3, 4>& src, iu::LinearDeviceMemory<double, 4>& dst1, iu::LinearDeviceMemory<double, 4>& dst2, iu::LinearDeviceMemory<double, 4>& dst3){iuprivate::math::splitPlanes(src, dst1, dst2, dst3);}
void splitPlanes(iu::LinearHostMemory<double3, 5>& src, iu::LinearHostMemory<double, 5>& dst1, iu::LinearHostMemory<double, 5>& dst2, iu::LinearHostMemory<double, 5>& dst3){iuprivate::math::splitPlanes(src, dst1, dst2, dst3);}
void splitPlanes(iu::LinearDeviceMemory<double3, 5>& src, iu::LinearDeviceMemory<double, 5>& dst1, iu::LinearDeviceMemory<double, 5>& dst2, iu::LinearDeviceMemory<double, 5>& dst3){iuprivate::math::splitPlanes(src, dst1, dst2, dst3);}

// combine planes
void combinePlanes(iu::VolumeCpu_32f_C1& src1, iu::VolumeCpu_32f_C1& src2, iu::VolumeCpu_32f_C2& dst) {iuprivate::math::combinePlanes(src1, src2, dst);}
void combinePlanes(iu::VolumeGpu_32f_C1& src1, iu::VolumeGpu_32f_C1& src2, iu::VolumeGpu_32f_C2& dst) {iuprivate::math::combinePlanes(src1, src2, dst);}
void combinePlanes(iu::ImageCpu_32f_C1& src1, iu::ImageCpu_32f_C1& src2, iu::ImageCpu_32f_C2& dst) {iuprivate::math::combinePlanes(src1, src2, dst);}
void combinePlanes(iu::ImageGpu_32f_C1& src1, iu::ImageGpu_32f_C1& src2, iu::ImageGpu_32f_C2& dst) {iuprivate::math::combinePlanes(src1, src2, dst);}
void combinePlanes(iu::LinearHostMemory_32f_C1& src1, iu::LinearHostMemory_32f_C1& src2, iu::LinearHostMemory_32f_C2& dst) {iuprivate::math::combinePlanes(src1, src2, dst);}
void combinePlanes(iu::LinearDeviceMemory_32f_C1& src1, iu::LinearDeviceMemory_32f_C1& src2, iu::LinearDeviceMemory_32f_C2& dst) {iuprivate::math::combinePlanes(src1, src2, dst);}
void combinePlanes(iu::LinearHostMemory<float, 2>& src1, iu::LinearHostMemory<float, 2>& src2, iu::LinearHostMemory<float2, 2>& dst) {iuprivate::math::combinePlanes(src1, src2, dst);}
void combinePlanes(iu::LinearDeviceMemory<float, 2>& src1, iu::LinearDeviceMemory<float, 2>& src2, iu::LinearDeviceMemory<float2, 2>& dst) {iuprivate::math::combinePlanes(src1, src2, dst);}
void combinePlanes(iu::LinearHostMemory<float, 3>& src1, iu::LinearHostMemory<float, 3>& src2, iu::LinearHostMemory<float2, 3>& dst) {iuprivate::math::combinePlanes(src1, src2, dst);}
void combinePlanes(iu::LinearDeviceMemory<float, 3>& src1, iu::LinearDeviceMemory<float, 3>& src2, iu::LinearDeviceMemory<float2, 3>& dst) {iuprivate::math::combinePlanes(src1, src2, dst);}
void combinePlanes(iu::LinearHostMemory<float, 4>& src1, iu::LinearHostMemory<float, 4>& src2, iu::LinearHostMemory<float2, 4>& dst) {iuprivate::math::combinePlanes(src1, src2, dst);}
void combinePlanes(iu::LinearDeviceMemory<float, 4>& src1, iu::LinearDeviceMemory<float, 4>& src2, iu::LinearDeviceMemory<float2, 4>& dst) {iuprivate::math::combinePlanes(src1, src2, dst);}
void combinePlanes(iu::LinearHostMemory<float, 5>& src1, iu::LinearHostMemory<float, 5>& src2, iu::LinearHostMemory<float2, 5>& dst) {iuprivate::math::combinePlanes(src1, src2, dst);}
void combinePlanes(iu::LinearDeviceMemory<float, 5>& src1, iu::LinearDeviceMemory<float, 5>& src2, iu::LinearDeviceMemory<float2, 5>& dst) {iuprivate::math::combinePlanes(src1, src2, dst);}

void combinePlanes(iu::VolumeCpu_32f_C1& src1, iu::VolumeCpu_32f_C1& src2, iu::VolumeCpu_32f_C1& src3, iu::VolumeCpu_32f_C3& dst){iuprivate::math::combinePlanes(src1, src2, src3, dst);}
void combinePlanes(iu::VolumeGpu_32f_C1& src1, iu::VolumeGpu_32f_C1& src2, iu::VolumeGpu_32f_C1& src3, iu::VolumeGpu_32f_C3& dst){iuprivate::math::combinePlanes(src1, src2, src3, dst);}
void combinePlanes(iu::ImageCpu_32f_C1& src1, iu::ImageCpu_32f_C1& src2, iu::ImageCpu_32f_C1& src3, iu::ImageCpu_32f_C3& dst){iuprivate::math::combinePlanes(src1, src2, src3, dst);}
void combinePlanes(iu::ImageGpu_32f_C1& src1, iu::ImageGpu_32f_C1& src2, iu::ImageGpu_32f_C1& src3, iu::ImageGpu_32f_C3& dst){iuprivate::math::combinePlanes(src1, src2, src3, dst);}
void combinePlanes(iu::LinearHostMemory_32f_C1& src1, iu::LinearHostMemory_32f_C1& src2, iu::LinearHostMemory_32f_C1& src3, iu::LinearHostMemory_32f_C3& dst){iuprivate::math::combinePlanes(src1, src2, src3, dst);}
void combinePlanes(iu::LinearDeviceMemory_32f_C1& src1, iu::LinearDeviceMemory_32f_C1& src2, iu::LinearDeviceMemory_32f_C1& src3, iu::LinearDeviceMemory_32f_C3& dst){iuprivate::math::combinePlanes(src1, src2, src3, dst);}
void combinePlanes(iu::LinearHostMemory<float, 2>& src1, iu::LinearHostMemory<float, 2>& src2, iu::LinearHostMemory<float, 2>& src3, iu::LinearHostMemory<float3, 2>& dst){iuprivate::math::combinePlanes(src1, src2, src3, dst);}
void combinePlanes(iu::LinearDeviceMemory<float, 2>& src1, iu::LinearDeviceMemory<float, 2>& src2, iu::LinearDeviceMemory<float, 2>& src3, iu::LinearDeviceMemory<float3, 2>& dst){iuprivate::math::combinePlanes(src1, src2, src3, dst);}
void combinePlanes(iu::LinearHostMemory<float, 3>& src1, iu::LinearHostMemory<float, 3>& src2, iu::LinearHostMemory<float, 3>& src3, iu::LinearHostMemory<float3, 3>& dst){iuprivate::math::combinePlanes(src1, src2, src3, dst);}
void combinePlanes(iu::LinearDeviceMemory<float, 3>& src1, iu::LinearDeviceMemory<float, 3>& src2, iu::LinearDeviceMemory<float, 3>& src3, iu::LinearDeviceMemory<float3, 3>& dst){iuprivate::math::combinePlanes(src1, src2, src3, dst);}
void combinePlanes(iu::LinearHostMemory<float, 4>& src1, iu::LinearHostMemory<float, 4>& src2, iu::LinearHostMemory<float, 4>& src3, iu::LinearHostMemory<float3, 4>& dst){iuprivate::math::combinePlanes(src1, src2, src3, dst);}
void combinePlanes(iu::LinearDeviceMemory<float, 4>& src1, iu::LinearDeviceMemory<float, 4>& src2, iu::LinearDeviceMemory<float, 4>& src3, iu::LinearDeviceMemory<float3, 4>& dst){iuprivate::math::combinePlanes(src1, src2, src3, dst);}
void combinePlanes(iu::LinearHostMemory<float, 5>& src1, iu::LinearHostMemory<float, 5>& src2, iu::LinearHostMemory<float, 5>& src3, iu::LinearHostMemory<float3, 5>& dst){iuprivate::math::combinePlanes(src1, src2, src3, dst);}
void combinePlanes(iu::LinearDeviceMemory<float, 5>& src1, iu::LinearDeviceMemory<float, 5>& src2, iu::LinearDeviceMemory<float, 5>& src3, iu::LinearDeviceMemory<float3, 5>& dst){iuprivate::math::combinePlanes(src1, src2, src3, dst);}

void combinePlanes(iu::VolumeCpu_64f_C1& src1, iu::VolumeCpu_64f_C1& src2, iu::VolumeCpu_64f_C2& dst) {iuprivate::math::combinePlanes(src1, src2, dst);}
void combinePlanes(iu::VolumeGpu_64f_C1& src1, iu::VolumeGpu_64f_C1& src2, iu::VolumeGpu_64f_C2& dst) {iuprivate::math::combinePlanes(src1, src2, dst);}
void combinePlanes(iu::ImageCpu_64f_C1& src1, iu::ImageCpu_64f_C1& src2, iu::ImageCpu_64f_C2& dst) {iuprivate::math::combinePlanes(src1, src2, dst);}
void combinePlanes(iu::ImageGpu_64f_C1& src1, iu::ImageGpu_64f_C1& src2, iu::ImageGpu_64f_C2& dst) {iuprivate::math::combinePlanes(src1, src2, dst);}
void combinePlanes(iu::LinearHostMemory_64f_C1& src1, iu::LinearHostMemory_64f_C1& src2, iu::LinearHostMemory_64f_C2& dst) {iuprivate::math::combinePlanes(src1, src2, dst);}
void combinePlanes(iu::LinearDeviceMemory_64f_C1& src1, iu::LinearDeviceMemory_64f_C1& src2, iu::LinearDeviceMemory_64f_C2& dst) {iuprivate::math::combinePlanes(src1, src2, dst);}
void combinePlanes(iu::LinearHostMemory<double, 2>& src1, iu::LinearHostMemory<double, 2>& src2, iu::LinearHostMemory<double2, 2>& dst) {iuprivate::math::combinePlanes(src1, src2, dst);}
void combinePlanes(iu::LinearDeviceMemory<double, 2>& src1, iu::LinearDeviceMemory<double, 2>& src2, iu::LinearDeviceMemory<double2, 2>& dst) {iuprivate::math::combinePlanes(src1, src2, dst);}
void combinePlanes(iu::LinearHostMemory<double, 3>& src1, iu::LinearHostMemory<double, 3>& src2, iu::LinearHostMemory<double2, 3>& dst) {iuprivate::math::combinePlanes(src1, src2, dst);}
void combinePlanes(iu::LinearDeviceMemory<double, 3>& src1, iu::LinearDeviceMemory<double, 3>& src2, iu::LinearDeviceMemory<double2, 3>& dst) {iuprivate::math::combinePlanes(src1, src2, dst);}
void combinePlanes(iu::LinearHostMemory<double, 4>& src1, iu::LinearHostMemory<double, 4>& src2, iu::LinearHostMemory<double2, 4>& dst) {iuprivate::math::combinePlanes(src1, src2, dst);}
void combinePlanes(iu::LinearDeviceMemory<double, 4>& src1, iu::LinearDeviceMemory<double, 4>& src2, iu::LinearDeviceMemory<double2, 4>& dst) {iuprivate::math::combinePlanes(src1, src2, dst);}
void combinePlanes(iu::LinearHostMemory<double, 5>& src1, iu::LinearHostMemory<double, 5>& src2, iu::LinearHostMemory<double2, 5>& dst) {iuprivate::math::combinePlanes(src1, src2, dst);}
void combinePlanes(iu::LinearDeviceMemory<double, 5>& src1, iu::LinearDeviceMemory<double, 5>& src2, iu::LinearDeviceMemory<double2, 5>& dst) {iuprivate::math::combinePlanes(src1, src2, dst);}

void combinePlanes(iu::VolumeCpu_64f_C1& src1, iu::VolumeCpu_64f_C1& src2, iu::VolumeCpu_64f_C1& src3, iu::VolumeCpu_64f_C3& dst){iuprivate::math::combinePlanes(src1, src2, src3, dst);}
void combinePlanes(iu::VolumeGpu_64f_C1& src1, iu::VolumeGpu_64f_C1& src2, iu::VolumeGpu_64f_C1& src3, iu::VolumeGpu_64f_C3& dst){iuprivate::math::combinePlanes(src1, src2, src3, dst);}
void combinePlanes(iu::ImageCpu_64f_C1& src1, iu::ImageCpu_64f_C1& src2, iu::ImageCpu_64f_C1& src3, iu::ImageCpu_64f_C3& dst){iuprivate::math::combinePlanes(src1, src2, src3, dst);}
void combinePlanes(iu::ImageGpu_64f_C1& src1, iu::ImageGpu_64f_C1& src2, iu::ImageGpu_64f_C1& src3, iu::ImageGpu_64f_C3& dst){iuprivate::math::combinePlanes(src1, src2, src3, dst);}
void combinePlanes(iu::LinearHostMemory_64f_C1& src1, iu::LinearHostMemory_64f_C1& src2, iu::LinearHostMemory_64f_C1& src3, iu::LinearHostMemory_64f_C3& dst){iuprivate::math::combinePlanes(src1, src2, src3, dst);}
void combinePlanes(iu::LinearDeviceMemory_64f_C1& src1, iu::LinearDeviceMemory_64f_C1& src2, iu::LinearDeviceMemory_64f_C1& src3, iu::LinearDeviceMemory_64f_C3& dst){iuprivate::math::combinePlanes(src1, src2, src3, dst);}
void combinePlanes(iu::LinearHostMemory<double, 2>& src1, iu::LinearHostMemory<double, 2>& src2, iu::LinearHostMemory<double, 2>& src3, iu::LinearHostMemory<double3, 2>& dst){iuprivate::math::combinePlanes(src1, src2, src3, dst);}
void combinePlanes(iu::LinearDeviceMemory<double, 2>& src1, iu::LinearDeviceMemory<double, 2>& src2, iu::LinearDeviceMemory<double, 2>& src3, iu::LinearDeviceMemory<double3, 2>& dst){iuprivate::math::combinePlanes(src1, src2, src3, dst);}
void combinePlanes(iu::LinearHostMemory<double, 3>& src1, iu::LinearHostMemory<double, 3>& src2, iu::LinearHostMemory<double, 3>& src3, iu::LinearHostMemory<double3, 3>& dst){iuprivate::math::combinePlanes(src1, src2, src3, dst);}
void combinePlanes(iu::LinearDeviceMemory<double, 3>& src1, iu::LinearDeviceMemory<double, 3>& src2, iu::LinearDeviceMemory<double, 3>& src3, iu::LinearDeviceMemory<double3, 3>& dst){iuprivate::math::combinePlanes(src1, src2, src3, dst);}
void combinePlanes(iu::LinearHostMemory<double, 4>& src1, iu::LinearHostMemory<double, 4>& src2, iu::LinearHostMemory<double, 4>& src3, iu::LinearHostMemory<double3, 4>& dst){iuprivate::math::combinePlanes(src1, src2, src3, dst);}
void combinePlanes(iu::LinearDeviceMemory<double, 4>& src1, iu::LinearDeviceMemory<double, 4>& src2, iu::LinearDeviceMemory<double, 4>& src3, iu::LinearDeviceMemory<double3, 4>& dst){iuprivate::math::combinePlanes(src1, src2, src3, dst);}
void combinePlanes(iu::LinearHostMemory<double, 5>& src1, iu::LinearHostMemory<double, 5>& src2, iu::LinearHostMemory<double, 5>& src3, iu::LinearHostMemory<double3, 5>& dst){iuprivate::math::combinePlanes(src1, src2, src3, dst);}
void combinePlanes(iu::LinearDeviceMemory<double, 5>& src1, iu::LinearDeviceMemory<double, 5>& src2, iu::LinearDeviceMemory<double, 5>& src3, iu::LinearDeviceMemory<double3, 5>& dst){iuprivate::math::combinePlanes(src1, src2, src3, dst);}

namespace complex {
// abs
void abs(iu::VolumeCpu_32f_C2& complex, iu::VolumeCpu_32f_C1& real) {iuprivate::math::complex::abs(complex, real);}
void abs(iu::VolumeGpu_32f_C2& complex, iu::VolumeGpu_32f_C1& real) {iuprivate::math::complex::abs(complex, real);}
void abs(iu::ImageCpu_32f_C2& complex, iu::ImageCpu_32f_C1& real) {iuprivate::math::complex::abs(complex, real);}
void abs(iu::ImageGpu_32f_C2& complex, iu::ImageGpu_32f_C1& real) {iuprivate::math::complex::abs(complex, real);}
void abs(iu::LinearHostMemory_32f_C2& complex, iu::LinearHostMemory_32f_C1& real) {iuprivate::math::complex::abs(complex, real);}
void abs(iu::LinearDeviceMemory_32f_C2& complex, iu::LinearDeviceMemory_32f_C1& real) {iuprivate::math::complex::abs(complex, real);}
void abs(iu::LinearHostMemory<float2, 2>& complex, iu::LinearHostMemory<float, 2>& real) {iuprivate::math::complex::abs(complex, real);}
void abs(iu::LinearDeviceMemory<float2, 2>& complex, iu::LinearDeviceMemory<float, 2>& real) {iuprivate::math::complex::abs(complex, real);}
void abs(iu::LinearHostMemory<float2, 3>& complex, iu::LinearHostMemory<float, 3>& real) {iuprivate::math::complex::abs(complex, real);}
void abs(iu::LinearDeviceMemory<float2, 3>& complex, iu::LinearDeviceMemory<float, 3>& real) {iuprivate::math::complex::abs(complex, real);}
void abs(iu::LinearHostMemory<float2, 4>& complex, iu::LinearHostMemory<float, 4>& real) {iuprivate::math::complex::abs(complex, real);}
void abs(iu::LinearDeviceMemory<float2, 4>& complex, iu::LinearDeviceMemory<float, 4>& real) {iuprivate::math::complex::abs(complex, real);}
void abs(iu::LinearHostMemory<float2, 5>& complex, iu::LinearHostMemory<float, 5>& real) {iuprivate::math::complex::abs(complex, real);}
void abs(iu::LinearDeviceMemory<float2, 5>& complex, iu::LinearDeviceMemory<float, 5>& real) {iuprivate::math::complex::abs(complex, real);}

void abs(iu::VolumeCpu_64f_C2& complex, iu::VolumeCpu_64f_C1& real) {iuprivate::math::complex::abs(complex, real);}
void abs(iu::VolumeGpu_64f_C2& complex, iu::VolumeGpu_64f_C1& real) {iuprivate::math::complex::abs(complex, real);}
void abs(iu::ImageCpu_64f_C2& complex, iu::ImageCpu_64f_C1& real) {iuprivate::math::complex::abs(complex, real);}
void abs(iu::ImageGpu_64f_C2& complex, iu::ImageGpu_64f_C1& real) {iuprivate::math::complex::abs(complex, real);}
void abs(iu::LinearHostMemory_64f_C2& complex, iu::LinearHostMemory_64f_C1& real) {iuprivate::math::complex::abs(complex, real);}
void abs(iu::LinearDeviceMemory_64f_C2& complex, iu::LinearDeviceMemory_64f_C1& real) {iuprivate::math::complex::abs(complex, real);}
void abs(iu::LinearHostMemory<double2, 2>& complex, iu::LinearHostMemory<double, 2>& real) {iuprivate::math::complex::abs(complex, real);}
void abs(iu::LinearDeviceMemory<double2, 2>& complex, iu::LinearDeviceMemory<double, 2>& real) {iuprivate::math::complex::abs(complex, real);}
void abs(iu::LinearHostMemory<double2, 3>& complex, iu::LinearHostMemory<double, 3>& real) {iuprivate::math::complex::abs(complex, real);}
void abs(iu::LinearDeviceMemory<double2, 3>& complex, iu::LinearDeviceMemory<double, 3>& real) {iuprivate::math::complex::abs(complex, real);}
void abs(iu::LinearHostMemory<double2, 4>& complex, iu::LinearHostMemory<double, 4>& real) {iuprivate::math::complex::abs(complex, real);}
void abs(iu::LinearDeviceMemory<double2, 4>& complex, iu::LinearDeviceMemory<double, 4>& real) {iuprivate::math::complex::abs(complex, real);}
void abs(iu::LinearHostMemory<double2, 5>& complex, iu::LinearHostMemory<double, 5>& real) {iuprivate::math::complex::abs(complex, real);}
void abs(iu::LinearDeviceMemory<double2, 5>& complex, iu::LinearDeviceMemory<double, 5>& real) {iuprivate::math::complex::abs(complex, real);}

// real
void real(iu::VolumeCpu_32f_C2& complex, iu::VolumeCpu_32f_C1& real) {iuprivate::math::complex::real(complex, real);}
void real(iu::VolumeGpu_32f_C2& complex, iu::VolumeGpu_32f_C1& real) {iuprivate::math::complex::real(complex, real);}
void real(iu::ImageCpu_32f_C2& complex, iu::ImageCpu_32f_C1& real) {iuprivate::math::complex::real(complex, real);}
void real(iu::ImageGpu_32f_C2& complex, iu::ImageGpu_32f_C1& real) {iuprivate::math::complex::real(complex, real);}
void real(iu::LinearHostMemory_32f_C2& complex, iu::LinearHostMemory_32f_C1& real) {iuprivate::math::complex::real(complex, real);}
void real(iu::LinearDeviceMemory_32f_C2& complex, iu::LinearDeviceMemory_32f_C1& real) {iuprivate::math::complex::real(complex, real);}
void real(iu::LinearHostMemory<float2, 2>& complex, iu::LinearHostMemory<float, 2>& real) {iuprivate::math::complex::real(complex, real);}
void real(iu::LinearDeviceMemory<float2, 2>& complex, iu::LinearDeviceMemory<float, 2>& real) {iuprivate::math::complex::real(complex, real);}
void real(iu::LinearHostMemory<float2, 3>& complex, iu::LinearHostMemory<float, 3>& real) {iuprivate::math::complex::real(complex, real);}
void real(iu::LinearDeviceMemory<float2, 3>& complex, iu::LinearDeviceMemory<float, 3>& real) {iuprivate::math::complex::real(complex, real);}
void real(iu::LinearHostMemory<float2, 4>& complex, iu::LinearHostMemory<float, 4>& real) {iuprivate::math::complex::real(complex, real);}
void real(iu::LinearDeviceMemory<float2, 4>& complex, iu::LinearDeviceMemory<float, 4>& real) {iuprivate::math::complex::real(complex, real);}
void real(iu::LinearHostMemory<float2, 5>& complex, iu::LinearHostMemory<float, 5>& real) {iuprivate::math::complex::real(complex, real);}
void real(iu::LinearDeviceMemory<float2, 5>& complex, iu::LinearDeviceMemory<float, 5>& real) {iuprivate::math::complex::real(complex, real);}

void real(iu::VolumeCpu_64f_C2& complex, iu::VolumeCpu_64f_C1& real) {iuprivate::math::complex::real(complex, real);}
void real(iu::VolumeGpu_64f_C2& complex, iu::VolumeGpu_64f_C1& real) {iuprivate::math::complex::real(complex, real);}
void real(iu::ImageCpu_64f_C2& complex, iu::ImageCpu_64f_C1& real) {iuprivate::math::complex::real(complex, real);}
void real(iu::ImageGpu_64f_C2& complex, iu::ImageGpu_64f_C1& real) {iuprivate::math::complex::real(complex, real);}
void real(iu::LinearHostMemory_64f_C2& complex, iu::LinearHostMemory_64f_C1& real) {iuprivate::math::complex::real(complex, real);}
void real(iu::LinearDeviceMemory_64f_C2& complex, iu::LinearDeviceMemory_64f_C1& real) {iuprivate::math::complex::real(complex, real);}
void real(iu::LinearHostMemory<double2, 2>& complex, iu::LinearHostMemory<double, 2>& real) {iuprivate::math::complex::real(complex, real);}
void real(iu::LinearDeviceMemory<double2, 2>& complex, iu::LinearDeviceMemory<double, 2>& real) {iuprivate::math::complex::real(complex, real);}
void real(iu::LinearHostMemory<double2, 3>& complex, iu::LinearHostMemory<double, 3>& real) {iuprivate::math::complex::real(complex, real);}
void real(iu::LinearDeviceMemory<double2, 3>& complex, iu::LinearDeviceMemory<double, 3>& real) {iuprivate::math::complex::real(complex, real);}
void real(iu::LinearHostMemory<double2, 4>& complex, iu::LinearHostMemory<double, 4>& real) {iuprivate::math::complex::real(complex, real);}
void real(iu::LinearDeviceMemory<double2, 4>& complex, iu::LinearDeviceMemory<double, 4>& real) {iuprivate::math::complex::real(complex, real);}
void real(iu::LinearHostMemory<double2, 5>& complex, iu::LinearHostMemory<double, 5>& real) {iuprivate::math::complex::real(complex, real);}
void real(iu::LinearDeviceMemory<double2, 5>& complex, iu::LinearDeviceMemory<double, 5>& real) {iuprivate::math::complex::real(complex, real);}

// imag
void imag(iu::VolumeCpu_32f_C2& complex, iu::VolumeCpu_32f_C1& real) {iuprivate::math::complex::imag(complex, real);}
void imag(iu::VolumeGpu_32f_C2& complex, iu::VolumeGpu_32f_C1& real) {iuprivate::math::complex::imag(complex, real);}
void imag(iu::ImageCpu_32f_C2& complex, iu::ImageCpu_32f_C1& real) {iuprivate::math::complex::imag(complex, real);}
void imag(iu::ImageGpu_32f_C2& complex, iu::ImageGpu_32f_C1& real) {iuprivate::math::complex::imag(complex, real);}
void imag(iu::LinearHostMemory_32f_C2& complex, iu::LinearHostMemory_32f_C1& real) {iuprivate::math::complex::imag(complex, real);}
void imag(iu::LinearDeviceMemory_32f_C2& complex, iu::LinearDeviceMemory_32f_C1& real) {iuprivate::math::complex::imag(complex, real);}
void imag(iu::LinearHostMemory<float2, 2>& complex, iu::LinearHostMemory<float, 2>& real) {iuprivate::math::complex::imag(complex, real);}
void imag(iu::LinearDeviceMemory<float2, 2>& complex, iu::LinearDeviceMemory<float, 2>& real) {iuprivate::math::complex::imag(complex, real);}
void imag(iu::LinearHostMemory<float2, 3>& complex, iu::LinearHostMemory<float, 3>& real) {iuprivate::math::complex::imag(complex, real);}
void imag(iu::LinearDeviceMemory<float2, 3>& complex, iu::LinearDeviceMemory<float, 3>& real) {iuprivate::math::complex::imag(complex, real);}
void imag(iu::LinearHostMemory<float2, 4>& complex, iu::LinearHostMemory<float, 4>& real) {iuprivate::math::complex::imag(complex, real);}
void imag(iu::LinearDeviceMemory<float2, 4>& complex, iu::LinearDeviceMemory<float, 4>& real) {iuprivate::math::complex::imag(complex, real);}
void imag(iu::LinearHostMemory<float2, 5>& complex, iu::LinearHostMemory<float, 5>& real) {iuprivate::math::complex::imag(complex, real);}
void imag(iu::LinearDeviceMemory<float2, 5>& complex, iu::LinearDeviceMemory<float, 5>& real) {iuprivate::math::complex::imag(complex, real);}

void imag(iu::VolumeCpu_64f_C2& complex, iu::VolumeCpu_64f_C1& real) {iuprivate::math::complex::imag(complex, real);}
void imag(iu::VolumeGpu_64f_C2& complex, iu::VolumeGpu_64f_C1& real) {iuprivate::math::complex::imag(complex, real);}
void imag(iu::ImageCpu_64f_C2& complex, iu::ImageCpu_64f_C1& real) {iuprivate::math::complex::imag(complex, real);}
void imag(iu::ImageGpu_64f_C2& complex, iu::ImageGpu_64f_C1& real) {iuprivate::math::complex::imag(complex, real);}
void imag(iu::LinearHostMemory_64f_C2& complex, iu::LinearHostMemory_64f_C1& real) {iuprivate::math::complex::imag(complex, real);}
void imag(iu::LinearDeviceMemory_64f_C2& complex, iu::LinearDeviceMemory_64f_C1& real) {iuprivate::math::complex::imag(complex, real);}
void imag(iu::LinearHostMemory<double2, 2>& complex, iu::LinearHostMemory<double, 2>& real) {iuprivate::math::complex::imag(complex, real);}
void imag(iu::LinearDeviceMemory<double2, 2>& complex, iu::LinearDeviceMemory<double, 2>& real) {iuprivate::math::complex::imag(complex, real);}
void imag(iu::LinearHostMemory<double2, 3>& complex, iu::LinearHostMemory<double, 3>& real) {iuprivate::math::complex::imag(complex, real);}
void imag(iu::LinearDeviceMemory<double2, 3>& complex, iu::LinearDeviceMemory<double, 3>& real) {iuprivate::math::complex::imag(complex, real);}
void imag(iu::LinearHostMemory<double2, 4>& complex, iu::LinearHostMemory<double, 4>& real) {iuprivate::math::complex::imag(complex, real);}
void imag(iu::LinearDeviceMemory<double2, 4>& complex, iu::LinearDeviceMemory<double, 4>& real) {iuprivate::math::complex::imag(complex, real);}
void imag(iu::LinearHostMemory<double2, 5>& complex, iu::LinearHostMemory<double, 5>& real) {iuprivate::math::complex::imag(complex, real);}
void imag(iu::LinearDeviceMemory<double2, 5>& complex, iu::LinearDeviceMemory<double, 5>& real) {iuprivate::math::complex::imag(complex, real);}

// phase
void phase(iu::VolumeCpu_32f_C2& complex, iu::VolumeCpu_32f_C1& real) {iuprivate::math::complex::phase(complex, real);}
void phase(iu::VolumeGpu_32f_C2& complex, iu::VolumeGpu_32f_C1& real) {iuprivate::math::complex::phase(complex, real);}
void phase(iu::ImageCpu_32f_C2& complex, iu::ImageCpu_32f_C1& real) {iuprivate::math::complex::phase(complex, real);}
void phase(iu::ImageGpu_32f_C2& complex, iu::ImageGpu_32f_C1& real) {iuprivate::math::complex::phase(complex, real);}
void phase(iu::LinearHostMemory_32f_C2& complex, iu::LinearHostMemory_32f_C1& real) {iuprivate::math::complex::phase(complex, real);}
void phase(iu::LinearDeviceMemory_32f_C2& complex, iu::LinearDeviceMemory_32f_C1& real) {iuprivate::math::complex::phase(complex, real);}
void phase(iu::LinearHostMemory<float2, 2>& complex, iu::LinearHostMemory<float, 2>& real) {iuprivate::math::complex::phase(complex, real);}
void phase(iu::LinearDeviceMemory<float2, 2>& complex, iu::LinearDeviceMemory<float, 2>& real) {iuprivate::math::complex::phase(complex, real);}
void phase(iu::LinearHostMemory<float2, 3>& complex, iu::LinearHostMemory<float, 3>& real) {iuprivate::math::complex::phase(complex, real);}
void phase(iu::LinearDeviceMemory<float2, 3>& complex, iu::LinearDeviceMemory<float, 3>& real) {iuprivate::math::complex::phase(complex, real);}
void phase(iu::LinearHostMemory<float2, 4>& complex, iu::LinearHostMemory<float, 4>& real) {iuprivate::math::complex::phase(complex, real);}
void phase(iu::LinearDeviceMemory<float2, 4>& complex, iu::LinearDeviceMemory<float, 4>& real) {iuprivate::math::complex::phase(complex, real);}
void phase(iu::LinearHostMemory<float2, 5>& complex, iu::LinearHostMemory<float, 5>& real) {iuprivate::math::complex::phase(complex, real);}
void phase(iu::LinearDeviceMemory<float2, 5>& complex, iu::LinearDeviceMemory<float, 5>& real) {iuprivate::math::complex::phase(complex, real);}

void phase(iu::VolumeCpu_64f_C2& complex, iu::VolumeCpu_64f_C1& real) {iuprivate::math::complex::phase(complex, real);}
void phase(iu::VolumeGpu_64f_C2& complex, iu::VolumeGpu_64f_C1& real) {iuprivate::math::complex::phase(complex, real);}
void phase(iu::ImageCpu_64f_C2& complex, iu::ImageCpu_64f_C1& real) {iuprivate::math::complex::phase(complex, real);}
void phase(iu::ImageGpu_64f_C2& complex, iu::ImageGpu_64f_C1& real) {iuprivate::math::complex::phase(complex, real);}
void phase(iu::LinearHostMemory_64f_C2& complex, iu::LinearHostMemory_64f_C1& real) {iuprivate::math::complex::phase(complex, real);}
void phase(iu::LinearDeviceMemory_64f_C2& complex, iu::LinearDeviceMemory_64f_C1& real) {iuprivate::math::complex::phase(complex, real);}
void phase(iu::LinearHostMemory<double2, 2>& complex, iu::LinearHostMemory<double, 2>& real) {iuprivate::math::complex::phase(complex, real);}
void phase(iu::LinearDeviceMemory<double2, 2>& complex, iu::LinearDeviceMemory<double, 2>& real) {iuprivate::math::complex::phase(complex, real);}
void phase(iu::LinearHostMemory<double2, 3>& complex, iu::LinearHostMemory<double, 3>& real) {iuprivate::math::complex::phase(complex, real);}
void phase(iu::LinearDeviceMemory<double2, 3>& complex, iu::LinearDeviceMemory<double, 3>& real) {iuprivate::math::complex::phase(complex, real);}
void phase(iu::LinearHostMemory<double2, 4>& complex, iu::LinearHostMemory<double, 4>& real) {iuprivate::math::complex::phase(complex, real);}
void phase(iu::LinearDeviceMemory<double2, 4>& complex, iu::LinearDeviceMemory<double, 4>& real) {iuprivate::math::complex::phase(complex, real);}
void phase(iu::LinearHostMemory<double2, 5>& complex, iu::LinearHostMemory<double, 5>& real) {iuprivate::math::complex::phase(complex, real);}
void phase(iu::LinearDeviceMemory<double2, 5>& complex, iu::LinearDeviceMemory<double, 5>& real) {iuprivate::math::complex::phase(complex, real);}

// scale
void scale(iu::VolumeCpu_32f_C2& complex_src, const float& scale, iu::VolumeCpu_32f_C2& complex_dst) {iuprivate::math::mulC(complex_src, make_float2(scale, scale), complex_dst);}
void scale(iu::VolumeGpu_32f_C2& complex_src, const float& scale, iu::VolumeGpu_32f_C2& complex_dst) {iuprivate::math::mulC(complex_src, make_float2(scale, scale), complex_dst);}
void scale(iu::ImageCpu_32f_C2& complex_src, const float& scale, iu::ImageCpu_32f_C2& complex_dst) {iuprivate::math::mulC(complex_src, make_float2(scale, scale), complex_dst);}
void scale(iu::ImageGpu_32f_C2& complex_src, const float& scale, iu::ImageGpu_32f_C2& complex_dst) {iuprivate::math::mulC(complex_src, make_float2(scale, scale), complex_dst);}
void scale(iu::LinearHostMemory_32f_C2& complex_src, const float& scale, iu::LinearHostMemory_32f_C2& complex_dst) {iuprivate::math::mulC(complex_src, make_float2(scale, scale), complex_dst);}
void scale(iu::LinearDeviceMemory_32f_C2& complex_src, const float& scale, iu::LinearDeviceMemory_32f_C2& complex_dst) {iuprivate::math::mulC(complex_src, make_float2(scale, scale), complex_dst);}
void scale(iu::LinearHostMemory<float2, 2>& complex_src, const float& scale, iu::LinearHostMemory<float2, 2>& complex_dst) {iuprivate::math::mulC(complex_src, make_float2(scale, scale), complex_dst);}
void scale(iu::LinearDeviceMemory<float2, 2>& complex_src, const float& scale, iu::LinearDeviceMemory<float2, 2>& complex_dst) {iuprivate::math::mulC(complex_src, make_float2(scale, scale), complex_dst);}
void scale(iu::LinearHostMemory<float2, 3>& complex_src, const float& scale, iu::LinearHostMemory<float2, 3>& complex_dst) {iuprivate::math::mulC(complex_src, make_float2(scale, scale), complex_dst);}
void scale(iu::LinearDeviceMemory<float2, 3>& complex_src, const float& scale, iu::LinearDeviceMemory<float2, 3>& complex_dst) {iuprivate::math::mulC(complex_src, make_float2(scale, scale), complex_dst);}
void scale(iu::LinearHostMemory<float2, 4>& complex_src, const float& scale, iu::LinearHostMemory<float2, 4>& complex_dst) {iuprivate::math::mulC(complex_src, make_float2(scale, scale), complex_dst);}
void scale(iu::LinearDeviceMemory<float2, 4>& complex_src, const float& scale, iu::LinearDeviceMemory<float2, 4>& complex_dst) {iuprivate::math::mulC(complex_src, make_float2(scale, scale), complex_dst);}
void scale(iu::LinearHostMemory<float2, 5>& complex_src, const float& scale, iu::LinearHostMemory<float2, 5>& complex_dst) {iuprivate::math::mulC(complex_src, make_float2(scale, scale), complex_dst);}
void scale(iu::LinearDeviceMemory<float2, 5>& complex_src, const float& scale, iu::LinearDeviceMemory<float2, 5>& complex_dst) {iuprivate::math::mulC(complex_src, make_float2(scale, scale), complex_dst);}

void scale(iu::VolumeCpu_64f_C2& complex_src, const double& scale, iu::VolumeCpu_64f_C2& complex_dst) {iuprivate::math::mulC(complex_src, make_double2(scale, scale), complex_dst);}
void scale(iu::VolumeGpu_64f_C2& complex_src, const double& scale, iu::VolumeGpu_64f_C2& complex_dst) {iuprivate::math::mulC(complex_src, make_double2(scale, scale), complex_dst);}
void scale(iu::ImageCpu_64f_C2& complex_src, const double& scale, iu::ImageCpu_64f_C2& complex_dst) {iuprivate::math::mulC(complex_src, make_double2(scale, scale), complex_dst);}
void scale(iu::ImageGpu_64f_C2& complex_src, const double& scale, iu::ImageGpu_64f_C2& complex_dst) {iuprivate::math::mulC(complex_src, make_double2(scale, scale), complex_dst);}
void scale(iu::LinearHostMemory_64f_C2& complex_src, const double& scale, iu::LinearHostMemory_64f_C2& complex_dst) {iuprivate::math::mulC(complex_src, make_double2(scale, scale), complex_dst);}
void scale(iu::LinearDeviceMemory_64f_C2& complex_src, const double& scale, iu::LinearDeviceMemory_64f_C2& complex_dst) {iuprivate::math::mulC(complex_src, make_double2(scale, scale), complex_dst);}
void scale(iu::LinearHostMemory<double2, 2>& complex_src, const double& scale, iu::LinearHostMemory<double2, 2>& complex_dst) {iuprivate::math::mulC(complex_src, make_double2(scale, scale), complex_dst);}
void scale(iu::LinearDeviceMemory<double2, 2>& complex_src, const double& scale, iu::LinearDeviceMemory<double2, 2>& complex_dst) {iuprivate::math::mulC(complex_src, make_double2(scale, scale), complex_dst);}
void scale(iu::LinearHostMemory<double2, 3>& complex_src, const double& scale, iu::LinearHostMemory<double2, 3>& complex_dst) {iuprivate::math::mulC(complex_src, make_double2(scale, scale), complex_dst);}
void scale(iu::LinearDeviceMemory<double2, 3>& complex_src, const double& scale, iu::LinearDeviceMemory<double2, 3>& complex_dst) {iuprivate::math::mulC(complex_src, make_double2(scale, scale), complex_dst);}
void scale(iu::LinearHostMemory<double2, 4>& complex_src, const double& scale, iu::LinearHostMemory<double2, 4>& complex_dst) {iuprivate::math::mulC(complex_src, make_double2(scale, scale), complex_dst);}
void scale(iu::LinearDeviceMemory<double2, 4>& complex_src, const double& scale, iu::LinearDeviceMemory<double2, 4>& complex_dst) {iuprivate::math::mulC(complex_src, make_double2(scale, scale), complex_dst);}
void scale(iu::LinearHostMemory<double2, 5>& complex_src, const double& scale, iu::LinearHostMemory<double2, 5>& complex_dst) {iuprivate::math::mulC(complex_src, make_double2(scale, scale), complex_dst);}
void scale(iu::LinearDeviceMemory<double2, 5>& complex_src, const double& scale, iu::LinearDeviceMemory<double2, 5>& complex_dst) {iuprivate::math::mulC(complex_src, make_double2(scale, scale), complex_dst);}

// multiply complex with real
void multiply(iu::VolumeCpu_32f_C2& complex_src, iu::VolumeCpu_32f_C1& real, iu::VolumeCpu_32f_C2& complex_dst)
{
  iuprivate::math::complex::multiply(complex_src, real, complex_dst);
}
void multiply(iu::VolumeGpu_32f_C2& complex_src, iu::VolumeGpu_32f_C1& real, iu::VolumeGpu_32f_C2& complex_dst)
{
  iuprivate::math::complex::multiply(complex_src, real, complex_dst);
}
void multiply(iu::ImageCpu_32f_C2& complex_src, iu::ImageCpu_32f_C1& real, iu::ImageCpu_32f_C2& complex_dst)
{
  iuprivate::math::complex::multiply(complex_src, real, complex_dst);
}
void multiply(iu::ImageGpu_32f_C2& complex_src, iu::ImageGpu_32f_C1& real, iu::ImageGpu_32f_C2& complex_dst)
{
  iuprivate::math::complex::multiply(complex_src, real, complex_dst);
}

void multiply(iu::LinearHostMemory_32f_C2& complex_src, iu::LinearHostMemory_32f_C1& real, iu::LinearHostMemory_32f_C2& complex_dst)
{
  iuprivate::math::complex::multiply(complex_src, real, complex_dst);
}
void multiply(iu::LinearDeviceMemory_32f_C2& complex_src, iu::LinearDeviceMemory_32f_C1& real, iu::LinearDeviceMemory_32f_C2& complex_dst)
{
  iuprivate::math::complex::multiply(complex_src, real, complex_dst);
}

void multiply(iu::VolumeCpu_64f_C2& complex_src, iu::VolumeCpu_64f_C1& real, iu::VolumeCpu_64f_C2& complex_dst)
{
  iuprivate::math::complex::multiply(complex_src, real, complex_dst);
}
void multiply(iu::VolumeGpu_64f_C2& complex_src, iu::VolumeGpu_64f_C1& real, iu::VolumeGpu_64f_C2& complex_dst)
{
  iuprivate::math::complex::multiply(complex_src, real, complex_dst);
}

void multiply(iu::ImageCpu_64f_C2& complex_src, iu::ImageCpu_64f_C1& real, iu::ImageCpu_64f_C2& complex_dst)
{
  iuprivate::math::complex::multiply(complex_src, real, complex_dst);
}
void multiply(iu::ImageGpu_64f_C2& complex_src, iu::ImageGpu_64f_C1& real, iu::ImageGpu_64f_C2& complex_dst)
{
  iuprivate::math::complex::multiply(complex_src, real, complex_dst);
}

void multiply(iu::LinearHostMemory_64f_C2& complex_src, iu::LinearHostMemory_64f_C1& real, iu::LinearHostMemory_64f_C2& complex_dst)
{
  iuprivate::math::complex::multiply(complex_src, real, complex_dst);
}
void multiply(iu::LinearDeviceMemory_64f_C2& complex_src, iu::LinearDeviceMemory_64f_C1& real, iu::LinearDeviceMemory_64f_C2& complex_dst)
{
  iuprivate::math::complex::multiply(complex_src, real, complex_dst);
}

void multiply(iu::LinearHostMemory<float2, 2>& complex_src, iu::LinearHostMemory<float, 2>& real, iu::LinearHostMemory<float2, 2>& complex_dst)
{
  iuprivate::math::complex::multiply(complex_src, real, complex_dst);
}
void multiply(iu::LinearDeviceMemory<float2, 2>& complex_src, iu::LinearDeviceMemory<float, 2>& real, iu::LinearDeviceMemory<float2, 2>& complex_dst)
{
  iuprivate::math::complex::multiply(complex_src, real, complex_dst);
}
void multiply(iu::LinearHostMemory<float2, 3>& complex_src, iu::LinearHostMemory<float, 3>& real, iu::LinearHostMemory<float2, 3>& complex_dst)
{
  iuprivate::math::complex::multiply(complex_src, real, complex_dst);
}
void multiply(iu::LinearDeviceMemory<float2, 3>& complex_src, iu::LinearDeviceMemory<float, 3>& real, iu::LinearDeviceMemory<float2, 3>& complex_dst)
{
  iuprivate::math::complex::multiply(complex_src, real, complex_dst);
}
void multiply(iu::LinearHostMemory<float2, 4>& complex_src, iu::LinearHostMemory<float, 4>& real, iu::LinearHostMemory<float2, 4>& complex_dst)
{
  iuprivate::math::complex::multiply(complex_src, real, complex_dst);
}
void multiply(iu::LinearDeviceMemory<float2, 4>& complex_src, iu::LinearDeviceMemory<float, 4>& real, iu::LinearDeviceMemory<float2, 4>& complex_dst)
{
  iuprivate::math::complex::multiply(complex_src, real, complex_dst);
}
void multiply(iu::LinearHostMemory<float2, 5>& complex_src, iu::LinearHostMemory<float, 5>& real, iu::LinearHostMemory<float2, 5>& complex_dst)
{
  iuprivate::math::complex::multiply(complex_src, real, complex_dst);
}
void multiply(iu::LinearDeviceMemory<float2, 5>& complex_src, iu::LinearDeviceMemory<float, 5>& real, iu::LinearDeviceMemory<float2, 5>& complex_dst)
{
  iuprivate::math::complex::multiply(complex_src, real, complex_dst);
}

void multiply(iu::LinearHostMemory<double2, 2>& complex_src, iu::LinearHostMemory<double, 2>& real, iu::LinearHostMemory<double2, 2>& complex_dst)
{
  iuprivate::math::complex::multiply(complex_src, real, complex_dst);
}
void multiply(iu::LinearDeviceMemory<double2, 2>& complex_src, iu::LinearDeviceMemory<double, 2>& real, iu::LinearDeviceMemory<double2, 2>& complex_dst)
{
  iuprivate::math::complex::multiply(complex_src, real, complex_dst);
}
void multiply(iu::LinearHostMemory<double2, 3>& complex_src, iu::LinearHostMemory<double, 3>& real, iu::LinearHostMemory<double2, 3>& complex_dst)
{
  iuprivate::math::complex::multiply(complex_src, real, complex_dst);
}
void multiply(iu::LinearDeviceMemory<double2, 3>& complex_src, iu::LinearDeviceMemory<double, 3>& real, iu::LinearDeviceMemory<double2, 3>& complex_dst)
{
  iuprivate::math::complex::multiply(complex_src, real, complex_dst);
}
void multiply(iu::LinearHostMemory<double2, 4>& complex_src, iu::LinearHostMemory<double, 4>& real, iu::LinearHostMemory<double2, 4>& complex_dst)
{
  iuprivate::math::complex::multiply(complex_src, real, complex_dst);
}
void multiply(iu::LinearDeviceMemory<double2, 4>& complex_src, iu::LinearDeviceMemory<double, 4>& real, iu::LinearDeviceMemory<double2, 4>& complex_dst)
{
  iuprivate::math::complex::multiply(complex_src, real, complex_dst);
}
void multiply(iu::LinearHostMemory<double2, 5>& complex_src, iu::LinearHostMemory<double, 5>& real, iu::LinearHostMemory<double2, 5>& complex_dst)
{
  iuprivate::math::complex::multiply(complex_src, real, complex_dst);
}
void multiply(iu::LinearDeviceMemory<double2, 5>& complex_src, iu::LinearDeviceMemory<double, 5>& real, iu::LinearDeviceMemory<double2, 5>& complex_dst)
{
  iuprivate::math::complex::multiply(complex_src, real, complex_dst);
}

// multiply complex with complex
void multiply(iu::VolumeCpu_32f_C2& complex_src1, iu::VolumeCpu_32f_C2& complex_src2, iu::VolumeCpu_32f_C2& complex_dst){iuprivate::math::complex::multiply(complex_src1,complex_src2,complex_dst);}
void multiply(iu::VolumeGpu_32f_C2& complex_src1, iu::VolumeGpu_32f_C2& complex_src2, iu::VolumeGpu_32f_C2& complex_dst){iuprivate::math::complex::multiply(complex_src1,complex_src2,complex_dst);}
void multiply(iu::ImageCpu_32f_C2& complex_src1, iu::ImageCpu_32f_C2& complex_src2, iu::ImageCpu_32f_C2& complex_dst){iuprivate::math::complex::multiply(complex_src1,complex_src2,complex_dst);}
void multiply(iu::ImageGpu_32f_C2& complex_src1, iu::ImageGpu_32f_C2& complex_src2, iu::ImageGpu_32f_C2& complex_dst){iuprivate::math::complex::multiply(complex_src1,complex_src2,complex_dst);}
void multiply(iu::LinearHostMemory_32f_C2& complex_src1, iu::LinearHostMemory_32f_C2& complex_src2, iu::LinearHostMemory_32f_C2& complex_dst){iuprivate::math::complex::multiply(complex_src1,complex_src2,complex_dst);}
void multiply(iu::LinearDeviceMemory_32f_C2& complex_src1, iu::LinearDeviceMemory_32f_C2& complex_src2, iu::LinearDeviceMemory_32f_C2& complex_dst){iuprivate::math::complex::multiply(complex_src1,complex_src2,complex_dst);}
void multiply(iu::LinearHostMemory<float2, 2>& complex_src1, iu::LinearHostMemory<float2, 2>& complex_src2, iu::LinearHostMemory<float2, 2>& complex_dst){iuprivate::math::complex::multiply(complex_src1,complex_src2,complex_dst);}
void multiply(iu::LinearDeviceMemory<float2, 2>& complex_src1, iu::LinearDeviceMemory<float2, 2>& complex_src2, iu::LinearDeviceMemory<float2, 2>& complex_dst){iuprivate::math::complex::multiply(complex_src1,complex_src2,complex_dst);}
void multiply(iu::LinearHostMemory<float2, 3>& complex_src1, iu::LinearHostMemory<float2, 3>& complex_src2, iu::LinearHostMemory<float2, 3>& complex_dst){iuprivate::math::complex::multiply(complex_src1,complex_src2,complex_dst);}
void multiply(iu::LinearDeviceMemory<float2, 3>& complex_src1, iu::LinearDeviceMemory<float2, 3>& complex_src2, iu::LinearDeviceMemory<float2, 3>& complex_dst){iuprivate::math::complex::multiply(complex_src1,complex_src2,complex_dst);}
void multiply(iu::LinearHostMemory<float2, 4>& complex_src1, iu::LinearHostMemory<float2, 4>& complex_src2, iu::LinearHostMemory<float2, 4>& complex_dst){iuprivate::math::complex::multiply(complex_src1,complex_src2,complex_dst);}
void multiply(iu::LinearDeviceMemory<float2, 4>& complex_src1, iu::LinearDeviceMemory<float2, 4>& complex_src2, iu::LinearDeviceMemory<float2, 4>& complex_dst){iuprivate::math::complex::multiply(complex_src1,complex_src2,complex_dst);}
void multiply(iu::LinearHostMemory<float2, 5>& complex_src1, iu::LinearHostMemory<float2, 5>& complex_src2, iu::LinearHostMemory<float2, 5>& complex_dst){iuprivate::math::complex::multiply(complex_src1,complex_src2,complex_dst);}
void multiply(iu::LinearDeviceMemory<float2, 5>& complex_src1, iu::LinearDeviceMemory<float2, 5>& complex_src2, iu::LinearDeviceMemory<float2, 5>& complex_dst){iuprivate::math::complex::multiply(complex_src1,complex_src2,complex_dst);}

void multiply(iu::VolumeCpu_64f_C2& complex_src1, iu::VolumeCpu_64f_C2& complex_src2, iu::VolumeCpu_64f_C2& complex_dst){iuprivate::math::complex::multiply(complex_src1,complex_src2,complex_dst);}
void multiply(iu::VolumeGpu_64f_C2& complex_src1, iu::VolumeGpu_64f_C2& complex_src2, iu::VolumeGpu_64f_C2& complex_dst){iuprivate::math::complex::multiply(complex_src1,complex_src2,complex_dst);}
void multiply(iu::ImageCpu_64f_C2& complex_src1, iu::ImageCpu_64f_C2& complex_src2, iu::ImageCpu_64f_C2& complex_dst){iuprivate::math::complex::multiply(complex_src1,complex_src2,complex_dst);}
void multiply(iu::ImageGpu_64f_C2& complex_src1, iu::ImageGpu_64f_C2& complex_src2, iu::ImageGpu_64f_C2& complex_dst){iuprivate::math::complex::multiply(complex_src1,complex_src2,complex_dst);}
void multiply(iu::LinearHostMemory_64f_C2& complex_src1, iu::LinearHostMemory_64f_C2& complex_src2, iu::LinearHostMemory_64f_C2& complex_dst){iuprivate::math::complex::multiply(complex_src1,complex_src2,complex_dst);}
void multiply(iu::LinearDeviceMemory_64f_C2& complex_src1, iu::LinearDeviceMemory_64f_C2& complex_src2, iu::LinearDeviceMemory_64f_C2& complex_dst){iuprivate::math::complex::multiply(complex_src1,complex_src2,complex_dst);}
void multiply(iu::LinearHostMemory<double2, 2>& complex_src1, iu::LinearHostMemory<double2, 2>& complex_src2, iu::LinearHostMemory<double2, 2>& complex_dst){iuprivate::math::complex::multiply(complex_src1,complex_src2,complex_dst);}
void multiply(iu::LinearDeviceMemory<double2, 2>& complex_src1, iu::LinearDeviceMemory<double2, 2>& complex_src2, iu::LinearDeviceMemory<double2, 2>& complex_dst){iuprivate::math::complex::multiply(complex_src1,complex_src2,complex_dst);}
void multiply(iu::LinearHostMemory<double2, 3>& complex_src1, iu::LinearHostMemory<double2, 3>& complex_src2, iu::LinearHostMemory<double2, 3>& complex_dst){iuprivate::math::complex::multiply(complex_src1,complex_src2,complex_dst);}
void multiply(iu::LinearDeviceMemory<double2, 3>& complex_src1, iu::LinearDeviceMemory<double2, 3>& complex_src2, iu::LinearDeviceMemory<double2, 3>& complex_dst){iuprivate::math::complex::multiply(complex_src1,complex_src2,complex_dst);}
void multiply(iu::LinearHostMemory<double2, 4>& complex_src1, iu::LinearHostMemory<double2, 4>& complex_src2, iu::LinearHostMemory<double2, 4>& complex_dst){iuprivate::math::complex::multiply(complex_src1,complex_src2,complex_dst);}
void multiply(iu::LinearDeviceMemory<double2, 4>& complex_src1, iu::LinearDeviceMemory<double2, 4>& complex_src2, iu::LinearDeviceMemory<double2, 4>& complex_dst){iuprivate::math::complex::multiply(complex_src1,complex_src2,complex_dst);}
void multiply(iu::LinearHostMemory<double2, 5>& complex_src1, iu::LinearHostMemory<double2, 5>& complex_src2, iu::LinearHostMemory<double2, 5>& complex_dst){iuprivate::math::complex::multiply(complex_src1,complex_src2,complex_dst);}
void multiply(iu::LinearDeviceMemory<double2, 5>& complex_src1, iu::LinearDeviceMemory<double2, 5>& complex_src2, iu::LinearDeviceMemory<double2, 5>& complex_dst){iuprivate::math::complex::multiply(complex_src1,complex_src2,complex_dst);}

// multiply complex with complex conjugate
void multiplyConjugate(iu::VolumeCpu_32f_C2& complex_src1, iu::VolumeCpu_32f_C2& complex_src2, iu::VolumeCpu_32f_C2& complex_dst){iuprivate::math::complex::multiplyConjugate(complex_src1,complex_src2,complex_dst);}
void multiplyConjugate(iu::VolumeGpu_32f_C2& complex_src1, iu::VolumeGpu_32f_C2& complex_src2, iu::VolumeGpu_32f_C2& complex_dst){iuprivate::math::complex::multiplyConjugate(complex_src1,complex_src2,complex_dst);}
void multiplyConjugate(iu::ImageCpu_32f_C2& complex_src1, iu::ImageCpu_32f_C2& complex_src2, iu::ImageCpu_32f_C2& complex_dst){iuprivate::math::complex::multiplyConjugate(complex_src1,complex_src2,complex_dst);}
void multiplyConjugate(iu::ImageGpu_32f_C2& complex_src1, iu::ImageGpu_32f_C2& complex_src2, iu::ImageGpu_32f_C2& complex_dst){iuprivate::math::complex::multiplyConjugate(complex_src1,complex_src2,complex_dst);}
void multiplyConjugate(iu::LinearHostMemory_32f_C2& complex_src1, iu::LinearHostMemory_32f_C2& complex_src2, iu::LinearHostMemory_32f_C2& complex_dst){iuprivate::math::complex::multiplyConjugate(complex_src1,complex_src2,complex_dst);}
void multiplyConjugate(iu::LinearDeviceMemory_32f_C2& complex_src1, iu::LinearDeviceMemory_32f_C2& complex_src2, iu::LinearDeviceMemory_32f_C2& complex_dst){iuprivate::math::complex::multiplyConjugate(complex_src1,complex_src2,complex_dst);}
void multiplyConjugate(iu::LinearHostMemory<float2, 2>& complex_src1, iu::LinearHostMemory<float2, 2>& complex_src2, iu::LinearHostMemory<float2, 2>& complex_dst){iuprivate::math::complex::multiplyConjugate(complex_src1,complex_src2,complex_dst);}
void multiplyConjugate(iu::LinearDeviceMemory<float2, 2>& complex_src1, iu::LinearDeviceMemory<float2, 2>& complex_src2, iu::LinearDeviceMemory<float2, 2>& complex_dst){iuprivate::math::complex::multiplyConjugate(complex_src1,complex_src2,complex_dst);}
void multiplyConjugate(iu::LinearHostMemory<float2, 3>& complex_src1, iu::LinearHostMemory<float2, 3>& complex_src2, iu::LinearHostMemory<float2, 3>& complex_dst){iuprivate::math::complex::multiplyConjugate(complex_src1,complex_src2,complex_dst);}
void multiplyConjugate(iu::LinearDeviceMemory<float2, 3>& complex_src1, iu::LinearDeviceMemory<float2, 3>& complex_src2, iu::LinearDeviceMemory<float2, 3>& complex_dst){iuprivate::math::complex::multiplyConjugate(complex_src1,complex_src2,complex_dst);}
void multiplyConjugate(iu::LinearHostMemory<float2, 4>& complex_src1, iu::LinearHostMemory<float2, 4>& complex_src2, iu::LinearHostMemory<float2, 4>& complex_dst){iuprivate::math::complex::multiplyConjugate(complex_src1,complex_src2,complex_dst);}
void multiplyConjugate(iu::LinearDeviceMemory<float2, 4>& complex_src1, iu::LinearDeviceMemory<float2, 4>& complex_src2, iu::LinearDeviceMemory<float2, 4>& complex_dst){iuprivate::math::complex::multiplyConjugate(complex_src1,complex_src2,complex_dst);}
void multiplyConjugate(iu::LinearHostMemory<float2, 5>& complex_src1, iu::LinearHostMemory<float2, 5>& complex_src2, iu::LinearHostMemory<float2, 5>& complex_dst){iuprivate::math::complex::multiplyConjugate(complex_src1,complex_src2,complex_dst);}
void multiplyConjugate(iu::LinearDeviceMemory<float2, 5>& complex_src1, iu::LinearDeviceMemory<float2, 5>& complex_src2, iu::LinearDeviceMemory<float2, 5>& complex_dst){iuprivate::math::complex::multiplyConjugate(complex_src1,complex_src2,complex_dst);}

void multiplyConjugate(iu::VolumeCpu_64f_C2& complex_src1, iu::VolumeCpu_64f_C2& complex_src2, iu::VolumeCpu_64f_C2& complex_dst){iuprivate::math::complex::multiplyConjugate(complex_src1,complex_src2,complex_dst);}
void multiplyConjugate(iu::VolumeGpu_64f_C2& complex_src1, iu::VolumeGpu_64f_C2& complex_src2, iu::VolumeGpu_64f_C2& complex_dst){iuprivate::math::complex::multiplyConjugate(complex_src1,complex_src2,complex_dst);}
void multiplyConjugate(iu::ImageCpu_64f_C2& complex_src1, iu::ImageCpu_64f_C2& complex_src2, iu::ImageCpu_64f_C2& complex_dst){iuprivate::math::complex::multiplyConjugate(complex_src1,complex_src2,complex_dst);}
void multiplyConjugate(iu::ImageGpu_64f_C2& complex_src1, iu::ImageGpu_64f_C2& complex_src2, iu::ImageGpu_64f_C2& complex_dst){iuprivate::math::complex::multiplyConjugate(complex_src1,complex_src2,complex_dst);}
void multiplyConjugate(iu::LinearHostMemory_64f_C2& complex_src1, iu::LinearHostMemory_64f_C2& complex_src2, iu::LinearHostMemory_64f_C2& complex_dst){iuprivate::math::complex::multiplyConjugate(complex_src1,complex_src2,complex_dst);}
void multiplyConjugate(iu::LinearDeviceMemory_64f_C2& complex_src1, iu::LinearDeviceMemory_64f_C2& complex_src2, iu::LinearDeviceMemory_64f_C2& complex_dst){iuprivate::math::complex::multiplyConjugate(complex_src1,complex_src2,complex_dst);}
void multiplyConjugate(iu::LinearHostMemory<double2, 2>& complex_src1, iu::LinearHostMemory<double2, 2>& complex_src2, iu::LinearHostMemory<double2, 2>& complex_dst){iuprivate::math::complex::multiplyConjugate(complex_src1,complex_src2,complex_dst);}
void multiplyConjugate(iu::LinearDeviceMemory<double2, 2>& complex_src1, iu::LinearDeviceMemory<double2, 2>& complex_src2, iu::LinearDeviceMemory<double2, 2>& complex_dst){iuprivate::math::complex::multiplyConjugate(complex_src1,complex_src2,complex_dst);}
void multiplyConjugate(iu::LinearHostMemory<double2, 3>& complex_src1, iu::LinearHostMemory<double2, 3>& complex_src2, iu::LinearHostMemory<double2, 3>& complex_dst){iuprivate::math::complex::multiplyConjugate(complex_src1,complex_src2,complex_dst);}
void multiplyConjugate(iu::LinearDeviceMemory<double2, 3>& complex_src1, iu::LinearDeviceMemory<double2, 3>& complex_src2, iu::LinearDeviceMemory<double2, 3>& complex_dst){iuprivate::math::complex::multiplyConjugate(complex_src1,complex_src2,complex_dst);}
void multiplyConjugate(iu::LinearHostMemory<double2, 4>& complex_src1, iu::LinearHostMemory<double2, 4>& complex_src2, iu::LinearHostMemory<double2, 4>& complex_dst){iuprivate::math::complex::multiplyConjugate(complex_src1,complex_src2,complex_dst);}
void multiplyConjugate(iu::LinearDeviceMemory<double2, 4>& complex_src1, iu::LinearDeviceMemory<double2, 4>& complex_src2, iu::LinearDeviceMemory<double2, 4>& complex_dst){iuprivate::math::complex::multiplyConjugate(complex_src1,complex_src2,complex_dst);}
void multiplyConjugate(iu::LinearHostMemory<double2, 5>& complex_src1, iu::LinearHostMemory<double2, 5>& complex_src2, iu::LinearHostMemory<double2, 5>& complex_dst){iuprivate::math::complex::multiplyConjugate(complex_src1,complex_src2,complex_dst);}
void multiplyConjugate(iu::LinearDeviceMemory<double2, 5>& complex_src1, iu::LinearDeviceMemory<double2, 5>& complex_src2, iu::LinearDeviceMemory<double2, 5>& complex_dst){iuprivate::math::complex::multiplyConjugate(complex_src1,complex_src2,complex_dst);}

void dotProduct(iu::ImageGpu_32f_C2& src1, iu::ImageGpu_32f_C2& src2, float2& result)
{
  iu::ImageGpu_32f_C2 dst(src1.size());
  iuprivate::math::complex::multiplyConjugate(src1, src2, dst);
  iuprivate::math::summation(dst, iu::type_trait<float2>::make(0), result);
}
void dotProduct(iu::VolumeGpu_32f_C2& src1, iu::VolumeGpu_32f_C2& src2, float2& result)
{
  iu::VolumeGpu_32f_C2 dst(src1.size());
  iuprivate::math::complex::multiplyConjugate(src1, src2, dst);
  iuprivate::math::summation(dst, iu::type_trait<float2>::make(0), result);
}
void dotProduct(iu::LinearDeviceMemory_32f_C2& src1, iu::LinearDeviceMemory_32f_C2& src2, float2& result)
{
  iu::LinearDeviceMemory_32f_C2 dst(src1.size());
  iuprivate::math::complex::multiplyConjugate(src1, src2, dst);
  iuprivate::math::summation(dst, iu::type_trait<float2>::make(0), result);
}
void dotProduct(iu::LinearDeviceMemory<float2, 2>& src1, iu::LinearDeviceMemory<float2, 2>& src2, float2& result)
{
  iu::LinearDeviceMemory<float2, 2> dst(src1.size());
  iuprivate::math::complex::multiplyConjugate(src1, src2, dst);
  iuprivate::math::summation(dst, iu::type_trait<float2>::make(0), result);
}
void dotProduct(iu::LinearDeviceMemory<float2, 3>& src1, iu::LinearDeviceMemory<float2, 3>& src2, float2& result)
{
  iu::LinearDeviceMemory<float2, 3> dst(src1.size());
  iuprivate::math::complex::multiplyConjugate(src1, src2, dst);
  iuprivate::math::summation(dst, iu::type_trait<float2>::make(0), result);
}
void dotProduct(iu::LinearDeviceMemory<float2, 4>& src1, iu::LinearDeviceMemory<float2, 4>& src2, float2& result)
{
  iu::LinearDeviceMemory<float2, 4> dst(src1.size());
  iuprivate::math::complex::multiplyConjugate(src1, src2, dst);
  iuprivate::math::summation(dst, iu::type_trait<float2>::make(0), result);
}
void dotProduct(iu::LinearDeviceMemory<float2, 5>& src1, iu::LinearDeviceMemory<float2, 5>& src2, float2& result)
{
  iu::LinearDeviceMemory<float2, 5> dst(src1.size());
  iuprivate::math::complex::multiplyConjugate(src1, src2, dst);
  iuprivate::math::summation(dst, iu::type_trait<float2>::make(0), result);
}

void dotProduct(iu::ImageCpu_32f_C2& src1, iu::ImageCpu_32f_C2& src2, float2& result)
{
  iu::ImageCpu_32f_C2 dst(src1.size());
  iuprivate::math::complex::multiplyConjugate(src1, src2, dst);
  iuprivate::math::summation(dst, iu::type_trait<float2>::make(0), result);
}
void dotProduct(iu::VolumeCpu_32f_C2& src1, iu::VolumeCpu_32f_C2& src2, float2& result)
{
  iu::VolumeCpu_32f_C2 dst(src1.size());
  iuprivate::math::complex::multiplyConjugate(src1, src2, dst);
  iuprivate::math::summation(dst, iu::type_trait<float2>::make(0), result);
}
void dotProduct(iu::LinearHostMemory_32f_C2& src1, iu::LinearHostMemory_32f_C2& src2, float2& result)
{
  iu::LinearHostMemory_32f_C2 dst(src1.size());
  iuprivate::math::complex::multiplyConjugate(src1, src2, dst);
  iuprivate::math::summation(dst, iu::type_trait<float2>::make(0), result);
}
void dotProduct(iu::LinearHostMemory<float2, 2>& src1, iu::LinearHostMemory<float2, 2>& src2, float2& result)
{
  iu::LinearHostMemory<float2, 2> dst(src1.size());
  iuprivate::math::complex::multiplyConjugate(src1, src2, dst);
  iuprivate::math::summation(dst, iu::type_trait<float2>::make(0), result);
}
void dotProduct(iu::LinearHostMemory<float2, 3>& src1, iu::LinearHostMemory<float2, 3>& src2, float2& result)
{
  iu::LinearHostMemory<float2, 3> dst(src1.size());
  iuprivate::math::complex::multiplyConjugate(src1, src2, dst);
  iuprivate::math::summation(dst, iu::type_trait<float2>::make(0), result);
}
void dotProduct(iu::LinearHostMemory<float2, 4>& src1, iu::LinearHostMemory<float2, 4>& src2, float2& result)
{
  iu::LinearHostMemory<float2, 4> dst(src1.size());
  iuprivate::math::complex::multiplyConjugate(src1, src2, dst);
  iuprivate::math::summation(dst, iu::type_trait<float2>::make(0), result);
}
void dotProduct(iu::LinearHostMemory<float2, 5>& src1, iu::LinearHostMemory<float2, 5>& src2, float2& result)
{
  iu::LinearHostMemory<float2, 5> dst(src1.size());
  iuprivate::math::complex::multiplyConjugate(src1, src2, dst);
  iuprivate::math::summation(dst, iu::type_trait<float2>::make(0), result);
}

void dotProduct(iu::ImageGpu_64f_C2& src1, iu::ImageGpu_64f_C2& src2, double2& result)
{
  iu::ImageGpu_64f_C2 dst(src1.size());
  iuprivate::math::complex::multiplyConjugate(src1, src2, dst);
  iuprivate::math::summation(dst, iu::type_trait<double2>::make(0), result);
}
void dotProduct(iu::VolumeGpu_64f_C2& src1, iu::VolumeGpu_64f_C2& src2, double2& result)
{
  iu::VolumeGpu_64f_C2 dst(src1.size());
  iuprivate::math::complex::multiplyConjugate(src1, src2, dst);
  iuprivate::math::summation(dst, iu::type_trait<double2>::make(0), result);
}
void dotProduct(iu::LinearDeviceMemory_64f_C2& src1, iu::LinearDeviceMemory_64f_C2& src2, double2& result)
{
  iu::LinearDeviceMemory_64f_C2 dst(src1.size());
  iuprivate::math::complex::multiplyConjugate(src1, src2, dst);
  iuprivate::math::summation(dst, iu::type_trait<double2>::make(0), result);
}
void dotProduct(iu::LinearDeviceMemory<double2, 2>& src1, iu::LinearDeviceMemory<double2, 2>& src2, double2& result)
{
  iu::LinearDeviceMemory<double2, 2> dst(src1.size());
  iuprivate::math::complex::multiplyConjugate(src1, src2, dst);
  iuprivate::math::summation(dst, iu::type_trait<double2>::make(0), result);
}
void dotProduct(iu::LinearDeviceMemory<double2, 3>& src1, iu::LinearDeviceMemory<double2, 3>& src2, double2& result)
{
  iu::LinearDeviceMemory<double2, 3> dst(src1.size());
  iuprivate::math::complex::multiplyConjugate(src1, src2, dst);
  iuprivate::math::summation(dst, iu::type_trait<double2>::make(0), result);
}
void dotProduct(iu::LinearDeviceMemory<double2, 4>& src1, iu::LinearDeviceMemory<double2, 4>& src2, double2& result)
{
  iu::LinearDeviceMemory<double2, 4> dst(src1.size());
  iuprivate::math::complex::multiplyConjugate(src1, src2, dst);
  iuprivate::math::summation(dst, iu::type_trait<double2>::make(0), result);
}
void dotProduct(iu::LinearDeviceMemory<double2, 5>& src1, iu::LinearDeviceMemory<double2, 5>& src2, double2& result)
{
  iu::LinearDeviceMemory<double2, 5> dst(src1.size());
  iuprivate::math::complex::multiplyConjugate(src1, src2, dst);
  iuprivate::math::summation(dst, iu::type_trait<double2>::make(0), result);
}

void dotProduct(iu::ImageCpu_64f_C2& src1, iu::ImageCpu_64f_C2& src2, double2& result)
{
  iu::ImageCpu_64f_C2 dst(src1.size());
  iuprivate::math::complex::multiplyConjugate(src1, src2, dst);
  iuprivate::math::summation(dst, iu::type_trait<double2>::make(0), result);
}
void dotProduct(iu::VolumeCpu_64f_C2& src1, iu::VolumeCpu_64f_C2& src2, double2& result)
{
  iu::VolumeCpu_64f_C2 dst(src1.size());
  iuprivate::math::complex::multiplyConjugate(src1, src2, dst);
  iuprivate::math::summation(dst, iu::type_trait<double2>::make(0), result);
}
void dotProduct(iu::LinearHostMemory_64f_C2& src1, iu::LinearHostMemory_64f_C2& src2, double2& result)
{
  iu::LinearHostMemory_64f_C2 dst(src1.size());
  iuprivate::math::complex::multiplyConjugate(src1, src2, dst);
  iuprivate::math::summation(dst, iu::type_trait<double2>::make(0), result);
}
void dotProduct(iu::LinearHostMemory<double2, 2>& src1, iu::LinearHostMemory<double2, 2>& src2, double2& result)
{
  iu::LinearHostMemory<double2, 2> dst(src1.size());
  iuprivate::math::complex::multiplyConjugate(src1, src2, dst);
  iuprivate::math::summation(dst, iu::type_trait<double2>::make(0), result);
}
void dotProduct(iu::LinearHostMemory<double2, 3>& src1, iu::LinearHostMemory<double2, 3>& src2, double2& result)
{
  iu::LinearHostMemory<double2, 3> dst(src1.size());
  iuprivate::math::complex::multiplyConjugate(src1, src2, dst);
  iuprivate::math::summation(dst, iu::type_trait<double2>::make(0), result);
}
void dotProduct(iu::LinearHostMemory<double2, 4>& src1, iu::LinearHostMemory<double2, 4>& src2, double2& result)
{
  iu::LinearHostMemory<double2, 4> dst(src1.size());
  iuprivate::math::complex::multiplyConjugate(src1, src2, dst);
  iuprivate::math::summation(dst, iu::type_trait<double2>::make(0), result);
}
void dotProduct(iu::LinearHostMemory<double2, 5>& src1, iu::LinearHostMemory<double2, 5>& src2, double2& result)
{
  iu::LinearHostMemory<double2, 5> dst(src1.size());
  iuprivate::math::complex::multiplyConjugate(src1, src2, dst);
  iuprivate::math::summation(dst, iu::type_trait<double2>::make(0), result);
}
}  //namespace complex

namespace fft{
// fftshift2
void fftshift2(const iu::ImageGpu_32f_C1& src, iu::ImageGpu_32f_C1& dst) {iuprivate::math::fft::fftshift2(src, dst);}
void fftshift2(const iu::ImageGpu_32f_C2& src, iu::ImageGpu_32f_C2& dst) {iuprivate::math::fft::fftshift2(src, dst);}
void fftshift2(const iu::VolumeGpu_32f_C1& src, iu::VolumeGpu_32f_C1& dst) {iuprivate::math::fft::fftshift2(src, dst);}
void fftshift2(const iu::VolumeGpu_32f_C2& src, iu::VolumeGpu_32f_C2& dst) {iuprivate::math::fft::fftshift2(src, dst);}
void fftshift2(const iu::LinearDeviceMemory<float, 2>& src, iu::LinearDeviceMemory<float, 2>& dst) {iuprivate::math::fft::fftshift2(src, dst);}
void fftshift2(const iu::LinearDeviceMemory<float2, 2>& src, iu::LinearDeviceMemory<float2, 2>& dst) {iuprivate::math::fft::fftshift2(src, dst);}
void fftshift2(const iu::LinearDeviceMemory<float, 3>& src, iu::LinearDeviceMemory<float, 3>& dst) {iuprivate::math::fft::fftshift2(src, dst);}
void fftshift2(const iu::LinearDeviceMemory<float2, 3>& src, iu::LinearDeviceMemory<float2, 3>& dst) {iuprivate::math::fft::fftshift2(src, dst);}
void fftshift2(const iu::LinearDeviceMemory<float, 4>& src, iu::LinearDeviceMemory<float, 4>& dst) {iuprivate::math::fft::fftshift2(src, dst);}
void fftshift2(const iu::LinearDeviceMemory<float2, 4>& src, iu::LinearDeviceMemory<float2, 4>& dst) {iuprivate::math::fft::fftshift2(src, dst);}

void fftshift2(const iu::ImageGpu_64f_C1& src, iu::ImageGpu_64f_C1& dst) {iuprivate::math::fft::fftshift2(src, dst);}
void fftshift2(const iu::ImageGpu_64f_C2& src, iu::ImageGpu_64f_C2& dst) {iuprivate::math::fft::fftshift2(src, dst);}
void fftshift2(const iu::VolumeGpu_64f_C1& src, iu::VolumeGpu_64f_C1& dst) {iuprivate::math::fft::fftshift2(src, dst);}
void fftshift2(const iu::VolumeGpu_64f_C2& src, iu::VolumeGpu_64f_C2& dst) {iuprivate::math::fft::fftshift2(src, dst);}
void fftshift2(const iu::LinearDeviceMemory<double, 2>& src, iu::LinearDeviceMemory<double, 2>& dst) {iuprivate::math::fft::fftshift2(src, dst);}
void fftshift2(const iu::LinearDeviceMemory<double2, 2>& src, iu::LinearDeviceMemory<double2, 2>& dst) {iuprivate::math::fft::fftshift2(src, dst);}
void fftshift2(const iu::LinearDeviceMemory<double, 3>& src, iu::LinearDeviceMemory<double, 3>& dst) {iuprivate::math::fft::fftshift2(src, dst);}
void fftshift2(const iu::LinearDeviceMemory<double2, 3>& src, iu::LinearDeviceMemory<double2, 3>& dst) {iuprivate::math::fft::fftshift2(src, dst);}
void fftshift2(const iu::LinearDeviceMemory<double, 4>& src, iu::LinearDeviceMemory<double, 4>& dst) {iuprivate::math::fft::fftshift2(src, dst);}
void fftshift2(const iu::LinearDeviceMemory<double2, 4>& src, iu::LinearDeviceMemory<double2, 4>& dst) {iuprivate::math::fft::fftshift2(src, dst);}

// ifftshift2
void ifftshift2(const iu::ImageGpu_32f_C1& src, iu::ImageGpu_32f_C1& dst) {iuprivate::math::fft::ifftshift2(src, dst);}
void ifftshift2(const iu::ImageGpu_32f_C2& src, iu::ImageGpu_32f_C2& dst) {iuprivate::math::fft::ifftshift2(src, dst);}
void ifftshift2(const iu::VolumeGpu_32f_C1& src, iu::VolumeGpu_32f_C1& dst) {iuprivate::math::fft::ifftshift2(src, dst);}
void ifftshift2(const iu::VolumeGpu_32f_C2& src, iu::VolumeGpu_32f_C2& dst) {iuprivate::math::fft::ifftshift2(src, dst);}
void ifftshift2(const iu::LinearDeviceMemory<float, 2>& src, iu::LinearDeviceMemory<float, 2>& dst) {iuprivate::math::fft::ifftshift2(src, dst);}
void ifftshift2(const iu::LinearDeviceMemory<float2, 2>& src, iu::LinearDeviceMemory<float2, 2>& dst) {iuprivate::math::fft::ifftshift2(src, dst);}
void ifftshift2(const iu::LinearDeviceMemory<float, 3>& src, iu::LinearDeviceMemory<float, 3>& dst) {iuprivate::math::fft::ifftshift2(src, dst);}
void ifftshift2(const iu::LinearDeviceMemory<float2, 3>& src, iu::LinearDeviceMemory<float2, 3>& dst) {iuprivate::math::fft::ifftshift2(src, dst);}
void ifftshift2(const iu::LinearDeviceMemory<float, 4>& src, iu::LinearDeviceMemory<float, 4>& dst) {iuprivate::math::fft::ifftshift2(src, dst);}
void ifftshift2(const iu::LinearDeviceMemory<float2, 4>& src, iu::LinearDeviceMemory<float2, 4>& dst) {iuprivate::math::fft::ifftshift2(src, dst);}

void ifftshift2(const iu::ImageGpu_64f_C1& src, iu::ImageGpu_64f_C1& dst) {iuprivate::math::fft::ifftshift2(src, dst);}
void ifftshift2(const iu::ImageGpu_64f_C2& src, iu::ImageGpu_64f_C2& dst) {iuprivate::math::fft::ifftshift2(src, dst);}
void ifftshift2(const iu::VolumeGpu_64f_C1& src, iu::VolumeGpu_64f_C1& dst) {iuprivate::math::fft::ifftshift2(src, dst);}
void ifftshift2(const iu::VolumeGpu_64f_C2& src, iu::VolumeGpu_64f_C2& dst) {iuprivate::math::fft::ifftshift2(src, dst);}
void ifftshift2(const iu::LinearDeviceMemory<double, 2>& src, iu::LinearDeviceMemory<double, 2>& dst) {iuprivate::math::fft::ifftshift2(src, dst);}
void ifftshift2(const iu::LinearDeviceMemory<double2, 2>& src, iu::LinearDeviceMemory<double2, 2>& dst) {iuprivate::math::fft::ifftshift2(src, dst);}
void ifftshift2(const iu::LinearDeviceMemory<double, 3>& src, iu::LinearDeviceMemory<double, 3>& dst) {iuprivate::math::fft::ifftshift2(src, dst);}
void ifftshift2(const iu::LinearDeviceMemory<double2, 3>& src, iu::LinearDeviceMemory<double2, 3>& dst) {iuprivate::math::fft::ifftshift2(src, dst);}
void ifftshift2(const iu::LinearDeviceMemory<double, 4>& src, iu::LinearDeviceMemory<double, 4>& dst) {iuprivate::math::fft::ifftshift2(src, dst);}
void ifftshift2(const iu::LinearDeviceMemory<double2, 4>& src, iu::LinearDeviceMemory<double2, 4>& dst) {iuprivate::math::fft::ifftshift2(src, dst);}

// fft2
void fft2(const iu::ImageGpu_32f_C2& src, iu::ImageGpu_32f_C2& dst, bool scale_sqrt) {iuprivate::math::fft::fft2(src,  dst, scale_sqrt);}
void fft2(const iu::VolumeGpu_32f_C2& src, iu::VolumeGpu_32f_C2& dst, bool scale_sqrt) {iuprivate::math::fft::fft2(src,  dst, scale_sqrt);}
void fft2(const iu::ImageGpu_32f_C1& src, iu::ImageGpu_32f_C2& dst, bool scale_sqrt) {iuprivate::math::fft::fft2(src,  dst, scale_sqrt);}
void fft2(const iu::VolumeGpu_32f_C1& src, iu::VolumeGpu_32f_C2& dst, bool scale_sqrt) {iuprivate::math::fft::fft2(src,  dst, scale_sqrt);}
void fft2(const iu::LinearDeviceMemory<float2, 2>& src, iu::LinearDeviceMemory<float2, 2>& dst, bool scale_sqrt) {iuprivate::math::fft::fft2(src,  dst, scale_sqrt);}
void fft2(const iu::LinearDeviceMemory<float2, 3>& src, iu::LinearDeviceMemory<float2, 3>& dst, bool scale_sqrt) {iuprivate::math::fft::fft2(src,  dst, scale_sqrt);}
void fft2(const iu::LinearDeviceMemory<float2, 4>& src, iu::LinearDeviceMemory<float2, 4>& dst, bool scale_sqrt) {iuprivate::math::fft::fft2(src,  dst, scale_sqrt);}
void fft2(const iu::LinearDeviceMemory<float, 2>& src, iu::LinearDeviceMemory<float2, 2>& dst, bool scale_sqrt) {iuprivate::math::fft::fft2(src,  dst, scale_sqrt);}
void fft2(const iu::LinearDeviceMemory<float, 3>& src, iu::LinearDeviceMemory<float2, 3>& dst, bool scale_sqrt) {iuprivate::math::fft::fft2(src,  dst, scale_sqrt);}
void fft2(const iu::LinearDeviceMemory<float, 4>& src, iu::LinearDeviceMemory<float2, 4>& dst, bool scale_sqrt) {iuprivate::math::fft::fft2(src,  dst, scale_sqrt);}

void fft2(const iu::ImageGpu_64f_C2& src, iu::ImageGpu_64f_C2& dst, bool scale_sqrt) {iuprivate::math::fft::fft2(src,  dst, scale_sqrt);}
void fft2(const iu::VolumeGpu_64f_C2& src, iu::VolumeGpu_64f_C2& dst, bool scale_sqrt) {iuprivate::math::fft::fft2(src,  dst, scale_sqrt);}
void fft2(const iu::ImageGpu_64f_C1& src, iu::ImageGpu_64f_C2& dst, bool scale_sqrt) {iuprivate::math::fft::fft2(src,  dst, scale_sqrt);}
void fft2(const iu::VolumeGpu_64f_C1& src, iu::VolumeGpu_64f_C2& dst, bool scale_sqrt) {iuprivate::math::fft::fft2(src,  dst, scale_sqrt);}
void fft2(const iu::LinearDeviceMemory<double2, 2>& src, iu::LinearDeviceMemory<double2, 2>& dst, bool scale_sqrt) {iuprivate::math::fft::fft2(src,  dst, scale_sqrt);}
void fft2(const iu::LinearDeviceMemory<double2, 3>& src, iu::LinearDeviceMemory<double2, 3>& dst, bool scale_sqrt) {iuprivate::math::fft::fft2(src,  dst, scale_sqrt);}
void fft2(const iu::LinearDeviceMemory<double2, 4>& src, iu::LinearDeviceMemory<double2, 4>& dst, bool scale_sqrt) {iuprivate::math::fft::fft2(src,  dst, scale_sqrt);}
void fft2(const iu::LinearDeviceMemory<double, 2>& src, iu::LinearDeviceMemory<double2, 2>& dst, bool scale_sqrt) {iuprivate::math::fft::fft2(src,  dst, scale_sqrt);}
void fft2(const iu::LinearDeviceMemory<double, 3>& src, iu::LinearDeviceMemory<double2, 3>& dst, bool scale_sqrt) {iuprivate::math::fft::fft2(src,  dst, scale_sqrt);}
void fft2(const iu::LinearDeviceMemory<double, 4>& src, iu::LinearDeviceMemory<double2, 4>& dst, bool scale_sqrt) {iuprivate::math::fft::fft2(src,  dst, scale_sqrt);}

// ifft2
void ifft2(const iu::ImageGpu_32f_C2& src, iu::ImageGpu_32f_C2& dst, bool scale_sqrt) {iuprivate::math::fft::ifft2(src,  dst, scale_sqrt);}
void ifft2(const iu::VolumeGpu_32f_C2& src, iu::VolumeGpu_32f_C2& dst, bool scale_sqrt) {iuprivate::math::fft::ifft2(src,  dst, scale_sqrt);}
void ifft2(const iu::ImageGpu_32f_C2& src, iu::ImageGpu_32f_C1& dst, bool scale_sqrt) {iuprivate::math::fft::ifft2(src,  dst, scale_sqrt);}
void ifft2(const iu::VolumeGpu_32f_C2& src, iu::VolumeGpu_32f_C1& dst, bool scale_sqrt) {iuprivate::math::fft::ifft2(src,  dst, scale_sqrt);}
void ifft2(const iu::LinearDeviceMemory<float2, 2>& src, iu::LinearDeviceMemory<float2, 2>& dst, bool scale_sqrt) {iuprivate::math::fft::ifft2(src,  dst, scale_sqrt);}
void ifft2(const iu::LinearDeviceMemory<float2, 3>& src, iu::LinearDeviceMemory<float2, 3>& dst, bool scale_sqrt) {iuprivate::math::fft::ifft2(src,  dst, scale_sqrt);}
void ifft2(const iu::LinearDeviceMemory<float2, 4>& src, iu::LinearDeviceMemory<float2, 4>& dst, bool scale_sqrt) {iuprivate::math::fft::ifft2(src,  dst, scale_sqrt);}
void ifft2(const iu::LinearDeviceMemory<float2, 2>& src, iu::LinearDeviceMemory<float, 2>& dst, bool scale_sqrt) {iuprivate::math::fft::ifft2(src,  dst, scale_sqrt);}
void ifft2(const iu::LinearDeviceMemory<float2, 3>& src, iu::LinearDeviceMemory<float, 3>& dst, bool scale_sqrt) {iuprivate::math::fft::ifft2(src,  dst, scale_sqrt);}
void ifft2(const iu::LinearDeviceMemory<float2, 4>& src, iu::LinearDeviceMemory<float, 4>& dst, bool scale_sqrt) {iuprivate::math::fft::ifft2(src,  dst, scale_sqrt);}

void ifft2(const iu::ImageGpu_64f_C2& src, iu::ImageGpu_64f_C2& dst, bool scale_sqrt) {iuprivate::math::fft::ifft2(src,  dst, scale_sqrt);}
void ifft2(const iu::VolumeGpu_64f_C2& src, iu::VolumeGpu_64f_C2& dst, bool scale_sqrt) {iuprivate::math::fft::ifft2(src,  dst, scale_sqrt);}
void ifft2(const iu::ImageGpu_64f_C2& src, iu::ImageGpu_64f_C1& dst, bool scale_sqrt) {iuprivate::math::fft::ifft2(src,  dst, scale_sqrt);}
void ifft2(const iu::VolumeGpu_64f_C2& src, iu::VolumeGpu_64f_C1& dst, bool scale_sqrt) {iuprivate::math::fft::ifft2(src,  dst, scale_sqrt);}
void ifft2(const iu::LinearDeviceMemory<double2, 2>& src, iu::LinearDeviceMemory<double2, 2>& dst, bool scale_sqrt) {iuprivate::math::fft::ifft2(src,  dst, scale_sqrt);}
void ifft2(const iu::LinearDeviceMemory<double2, 3>& src, iu::LinearDeviceMemory<double2, 3>& dst, bool scale_sqrt) {iuprivate::math::fft::ifft2(src,  dst, scale_sqrt);}
void ifft2(const iu::LinearDeviceMemory<double2, 4>& src, iu::LinearDeviceMemory<double2, 4>& dst, bool scale_sqrt) {iuprivate::math::fft::ifft2(src,  dst, scale_sqrt);}
void ifft2(const iu::LinearDeviceMemory<double2, 2>& src, iu::LinearDeviceMemory<double, 2>& dst, bool scale_sqrt) {iuprivate::math::fft::ifft2(src,  dst, scale_sqrt);}
void ifft2(const iu::LinearDeviceMemory<double2, 3>& src, iu::LinearDeviceMemory<double, 3>& dst, bool scale_sqrt) {iuprivate::math::fft::ifft2(src,  dst, scale_sqrt);}
void ifft2(const iu::LinearDeviceMemory<double2, 4>& src, iu::LinearDeviceMemory<double, 4>& dst, bool scale_sqrt) {iuprivate::math::fft::ifft2(src,  dst, scale_sqrt);}

// fft2c
void fft2c(const iu::ImageGpu_32f_C2& src, iu::ImageGpu_32f_C2& dst, bool scale_sqrt) {iuprivate::math::fft::fft2c(src,  dst, scale_sqrt);}
void fft2c(const iu::VolumeGpu_32f_C2& src, iu::VolumeGpu_32f_C2& dst, bool scale_sqrt) {iuprivate::math::fft::fft2c(src,  dst, scale_sqrt);}
void fft2c(const iu::LinearDeviceMemory<float2, 2>& src, iu::LinearDeviceMemory<float2, 2>& dst, bool scale_sqrt) {iuprivate::math::fft::fft2c(src,  dst, scale_sqrt);}
void fft2c(const iu::LinearDeviceMemory<float2, 3>& src, iu::LinearDeviceMemory<float2, 3>& dst, bool scale_sqrt) {iuprivate::math::fft::fft2c(src,  dst, scale_sqrt);}
void fft2c(const iu::LinearDeviceMemory<float2, 4>& src, iu::LinearDeviceMemory<float2, 4>& dst, bool scale_sqrt) {iuprivate::math::fft::fft2c(src,  dst, scale_sqrt);}

void fft2c(const iu::ImageGpu_64f_C2& src, iu::ImageGpu_64f_C2& dst, bool scale_sqrt) {iuprivate::math::fft::fft2c(src,  dst, scale_sqrt);}
void fft2c(const iu::VolumeGpu_64f_C2& src, iu::VolumeGpu_64f_C2& dst, bool scale_sqrt) {iuprivate::math::fft::fft2c(src,  dst, scale_sqrt);}
void fft2c(const iu::LinearDeviceMemory<double2, 2>& src, iu::LinearDeviceMemory<double2, 2>& dst, bool scale_sqrt) {iuprivate::math::fft::fft2c(src,  dst, scale_sqrt);}
void fft2c(const iu::LinearDeviceMemory<double2, 3>& src, iu::LinearDeviceMemory<double2, 3>& dst, bool scale_sqrt) {iuprivate::math::fft::fft2c(src,  dst, scale_sqrt);}
void fft2c(const iu::LinearDeviceMemory<double2, 4>& src, iu::LinearDeviceMemory<double2, 4>& dst, bool scale_sqrt) {iuprivate::math::fft::fft2c(src,  dst, scale_sqrt);}

// ifft2c
void ifft2c(const iu::ImageGpu_32f_C2& src, iu::ImageGpu_32f_C2& dst, bool scale_sqrt) {iuprivate::math::fft::ifft2c(src,  dst, scale_sqrt);}
void ifft2c(const iu::VolumeGpu_32f_C2& src, iu::VolumeGpu_32f_C2& dst, bool scale_sqrt) {iuprivate::math::fft::ifft2c(src,  dst, scale_sqrt);}
void ifft2c(const iu::LinearDeviceMemory<float2, 2>& src, iu::LinearDeviceMemory<float2, 2>& dst, bool scale_sqrt) {iuprivate::math::fft::ifft2c(src,  dst, scale_sqrt);}
void ifft2c(const iu::LinearDeviceMemory<float2, 3>& src, iu::LinearDeviceMemory<float2, 3>& dst, bool scale_sqrt) {iuprivate::math::fft::ifft2c(src,  dst, scale_sqrt);}
void ifft2c(const iu::LinearDeviceMemory<float2, 4>& src, iu::LinearDeviceMemory<float2, 4>& dst, bool scale_sqrt) {iuprivate::math::fft::ifft2c(src,  dst, scale_sqrt);}

void ifft2c(const iu::ImageGpu_64f_C2& src, iu::ImageGpu_64f_C2& dst, bool scale_sqrt) {iuprivate::math::fft::ifft2c(src,  dst, scale_sqrt);}
void ifft2c(const iu::VolumeGpu_64f_C2& src, iu::VolumeGpu_64f_C2& dst, bool scale_sqrt) {iuprivate::math::fft::ifft2c(src,  dst, scale_sqrt);}
void ifft2c(const iu::LinearDeviceMemory<double2, 2>& src, iu::LinearDeviceMemory<double2, 2>& dst, bool scale_sqrt) {iuprivate::math::fft::ifft2c(src,  dst, scale_sqrt);}
void ifft2c(const iu::LinearDeviceMemory<double2, 3>& src, iu::LinearDeviceMemory<double2, 3>& dst, bool scale_sqrt) {iuprivate::math::fft::ifft2c(src,  dst, scale_sqrt);}
void ifft2c(const iu::LinearDeviceMemory<double2, 4>& src, iu::LinearDeviceMemory<double2, 4>& dst, bool scale_sqrt) {iuprivate::math::fft::ifft2c(src,  dst, scale_sqrt);}
} //namespace fft


} //namespace math
} //namespace iu

