/*
 * Class       : $RCSfile$
 * Language    : C++
 * Description : Definition of
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUPRIVATE_CUBICTEXTURE_KERNELS_CU
#define IUPRIVATE_CUBICTEXTURE_KERNELS_CU

namespace iuprivate {

//! Bicubic interpolated texture lookup, using unnormalized coordinates.
//! Fast implementation, using 4 trilinear lookups.
//! @param tex  2D texture
//! @param x  unnormalized x texture coordinate
//! @param y  unnormalized y texture coordinate
template<class floatN, class T, enum cudaTextureReadMode mode>
__device__ floatN CUBICTEX2D(texture<T, 2, mode> tex, float x, float y)
{
  // transform the coordinate from [0,extent] to [-0.5, extent-0.5]
  const float2 coord_grid = make_float2(x - 0.5f, y - 0.5f);
  const float2 index = floor(coord_grid);
  const float2 fraction = coord_grid - index;
  float2 w0, w1, w2, w3;
  WEIGHTS(fraction, w0, w1, w2, w3);

  const float2 g0 = w0 + w1;
  const float2 g1 = w2 + w3;
  const float2 h0 = (w1 / g0) - make_float2(0.5f) + index;  //h0 = w1/g0 - 1, move from [-0.5, extent-0.5] to [0, extent]
  const float2 h1 = (w3 / g1) + make_float2(1.5f) + index;  //h1 = w3/g1 + 1, move from [-0.5, extent-0.5] to [0, extent]

  // fetch the four linear interpolations
  floatN tex00 = tex2D(tex, h0.x, h0.y);
  floatN tex10 = tex2D(tex, h1.x, h0.y);
  floatN tex01 = tex2D(tex, h0.x, h1.y);
  floatN tex11 = tex2D(tex, h1.x, h1.y);

  // weigh along the y-direction
  tex00 = g0.y * tex00 + g1.y * tex01;
  tex10 = g0.y * tex10 + g1.y * tex11;

  // weigh along the x-direction
  return (g0.x * tex00 + g1.x * tex10);
}


// Specializations

// These specializations fill in the floatN and T class types and therefore
// allow the cubicTex2D function to be called without any template arguments,
// thus with any <> brackets.

// 1-dimensional pixels
template<enum cudaTextureReadMode mode> __device__ float CUBICTEX2D(texture<float, 2, mode> tex, float x, float y) {return CUBICTEX2D<float, float, mode>(tex, x, y);}
template<enum cudaTextureReadMode mode> __device__ float CUBICTEX2D(texture<unsigned char, 2, mode> tex, float x, float y) {return CUBICTEX2D<float, unsigned char, mode>(tex, x, y);}
template<enum cudaTextureReadMode mode> __device__ float CUBICTEX2D(texture<char, 2, mode> tex, float x, float y) {return CUBICTEX2D<float, char, mode>(tex, x, y);}
template<enum cudaTextureReadMode mode> __device__ float CUBICTEX2D(texture<unsigned short, 2, mode> tex, float x, float y) {return CUBICTEX2D<float, unsigned short, mode>(tex, x, y);}
template<enum cudaTextureReadMode mode> __device__ float CUBICTEX2D(texture<short, 2, mode> tex, float x, float y) {return CUBICTEX2D<float, short, mode>(tex, x, y);}
template<enum cudaTextureReadMode mode> __device__ float CUBICTEX2D(texture<unsigned int, 2, mode> tex, float x, float y) {return CUBICTEX2D<float, uint, mode>(tex, x, y);}
template<enum cudaTextureReadMode mode> __device__ float CUBICTEX2D(texture<int, 2, mode> tex, float x, float y) {return CUBICTEX2D<float, int, mode>(tex, x, y);}
//// 2-dimensional pixels
//template<enum cudaTextureReadMode mode> __device__ float2 CUBICTEX2D(texture<float2, 2, mode> tex, float x, float y) {return CUBICTEX2D<float2, float2, mode>(tex, x, y);}
//template<enum cudaTextureReadMode mode> __device__ float2 CUBICTEX2D(texture<unsigned char2, 2, mode> tex, float x, float y) {return CUBICTEX2D<float2, unsigned char2, mode>(tex, x, y);}
//template<enum cudaTextureReadMode mode> __device__ float2 CUBICTEX2D(texture<char2, 2, mode> tex, float x, float y) {return CUBICTEX2D<float2, char2, mode>(tex, x, y);}
//template<enum cudaTextureReadMode mode> __device__ float2 CUBICTEX2D(texture<unsigned short2, 2, mode> tex, float x, float y) {return CUBICTEX2D<float2, unsigned short2, mode>(tex, x, y);}
//template<enum cudaTextureReadMode mode> __device__ float2 CUBICTEX2D(texture<short2, 2, mode> tex, float x, float y) {return CUBICTEX2D<float2, short2, mode>(tex, x, y);}
//template<enum cudaTextureReadMode mode> __device__ float2 CUBICTEX2D(texture<uint2, 2, mode> tex, float x, float y) {return CUBICTEX2D<float2, uint2, mode>(tex, x, y);}
//template<enum cudaTextureReadMode mode> __device__ float2 CUBICTEX2D(texture<int2, 2, mode> tex, float x, float y) {return CUBICTEX2D<float2, int2, mode>(tex, x, y);}
//// 3-dimensional pixels
//template<enum cudaTextureReadMode mode> __device__ float3 CUBICTEX2D(texture<float3, 2, mode> tex, float x, float y) {return CUBICTEX2D<float3, float3, mode>(tex, x, y);}
//template<enum cudaTextureReadMode mode> __device__ float3 CUBICTEX2D(texture<unsigned char3, 2, mode> tex, float x, float y) {return CUBICTEX2D<float3, unsigned char3, mode>(tex, x, y);}
//template<enum cudaTextureReadMode mode> __device__ float3 CUBICTEX2D(texture<char3, 2, mode> tex, float x, float y) {return CUBICTEX2D<float3, char3, mode>(tex, x, y);}
//template<enum cudaTextureReadMode mode> __device__ float3 CUBICTEX2D(texture<unsigned short3, 2, mode> tex, float x, float y) {return CUBICTEX2D<float3, unsigned short3, mode>(tex, x, y);}
//template<enum cudaTextureReadMode mode> __device__ float3 CUBICTEX2D(texture<short3, 2, mode> tex, float x, float y) {return CUBICTEX2D<float3, short3, mode>(tex, x, y);}
//template<enum cudaTextureReadMode mode> __device__ float3 CUBICTEX2D(texture<uint3, 2, mode> tex, float x, float y) {return CUBICTEX2D<float3, uint3, mode>(tex, x, y);}
//template<enum cudaTextureReadMode mode> __device__ float3 CUBICTEX2D(texture<int3, 2, mode> tex, float x, float y) {return CUBICTEX2D<float3, int3, mode>(tex, x, y);}
//// 4-dimensional pixels
//template<enum cudaTextureReadMode mode> __device__ float4 CUBICTEX2D(texture<float4, 2, mode> tex, float x, float y) {return CUBICTEX2D<float4, float4, mode>(tex, x, y);}
//template<enum cudaTextureReadMode mode> __device__ float4 CUBICTEX2D(texture<unsigned char4, 2, mode> tex, float x, float y) {return CUBICTEX2D<float4, unsigned char4, mode>(tex, x, y);}
//template<enum cudaTextureReadMode mode> __device__ float4 CUBICTEX2D(texture<char4, 2, mode> tex, float x, float y) {return CUBICTEX2D<float4, char4, mode>(tex, x, y);}
//template<enum cudaTextureReadMode mode> __device__ float4 CUBICTEX2D(texture<unsigned short4, 2, mode> tex, float x, float y) {return CUBICTEX2D<float4, unsigned short4, mode>(tex, x, y);}
//template<enum cudaTextureReadMode mode> __device__ float4 CUBICTEX2D(texture<short4, 2, mode> tex, float x, float y) {return CUBICTEX2D<float4, short4, mode>(tex, x, y);}
//template<enum cudaTextureReadMode mode> __device__ float4 CUBICTEX2D(texture<uint4, 2, mode> tex, float x, float y) {return CUBICTEX2D<float4, uint4, mode>(tex, x, y);}
//template<enum cudaTextureReadMode mode> __device__ float4 CUBICTEX2D(texture<int4, 2, mode> tex, float x, float y) {return CUBICTEX2D<float4, int4, mode>(tex, x, y);}

}

#endif // IUPRIVATE_CUBICTEXTURE_KERNELS_CU
