#ifndef IUCORE_TEXTURES_CUH
#define IUCORE_TEXTURES_CUH

namespace iuprivate {

// cuda textures (have to be global - do not mess around with them)
texture<unsigned char, 2, cudaReadModeElementType> tex1_8u_C1__;
texture<uchar2, 2, cudaReadModeElementType> tex1_8u_C2__;
texture<uchar4, 2, cudaReadModeElementType> tex1_8u_C4__;

texture<float, 2, cudaReadModeElementType> tex1_32f_C1__;
texture<float2, 2, cudaReadModeElementType> tex1_32f_C2__;
//texture<float3, 2, cudaReadModeElementType> tex1_32f_C3__;
texture<float4, 2, cudaReadModeElementType> tex1_32f_C4__;

texture<float, 2, cudaReadModeElementType> tex2_32f_C1__;

}

#endif // IUCORE_TEXTURES_CUH
