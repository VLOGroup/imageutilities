
//#include <helper_math.h>
#include "coredefs.h"
#include "memorydefs.h"
//#include "iutextures.cuh"
#include <assert.h>
#include <cuda.h>


/**
  Atomic add for double using comapre-and-swap (CAS)
  Taken from nVidia CUDA Programming Guide
  http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
  */
//__device__ double atomicAdd(double* address, double val)
//{

//    unsigned long long int* address_as_ull = (unsigned long long int*)address;
//    unsigned long long int old = *address_as_ull, assumed;
//    do
//    {
//        assumed = old;
//        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));

//    } while (assumed != old);  // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)

//    return __longlong_as_double(old);
//}

namespace iuprivate {


/* ***************************************************************************
 *  CUDA KERNELS
 * ***************************************************************************/

//-----------------------------------------------------------------------------
/** convert kernel 32f_C3 -> 32f_C4 (float3 -> float4)
 */
__global__ void cuConvertC3ToC4Kernel(const float3* src, size_t src_stride,
                                      float4* dst, size_t dst_stride, int width, int height)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;
  int c_src = y*src_stride + x;
  int c_dst = y*dst_stride + x;

  if (x<width && y<height)
  {
    float3 val=src[c_src];
    dst[c_dst] =  make_float4(val.x, val.y, val.z, 1.0f);
  }
}

//-----------------------------------------------------------------------------
/** convert kernel 32f_C4 -> 32f_C3 (float4 -> float3)
 */
__global__ void cuConvertC4ToC3Kernel(const float4* src, size_t src_stride,
                                      float3* dst, size_t dst_stride, int width, int height)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;
  int src_c = y*src_stride + x;
  int dst_c = y*dst_stride + x;

  if (x<width && y<height)
  {
    float4 val=src[src_c];
    dst[dst_c] = make_float3(val.x, val.y, val.z);
  }
}


//-----------------------------------------------------------------------------
/** convert kernel 8u_C1 -> 32f_C1 (unsigned char -> float)
 */
__global__ void cuConvert8uC1To32fC1Kernel(const unsigned char *src, size_t src_stride,
                                           float* dst, size_t dst_stride, float mul_constant,
                                           float add_constant, int width, int height)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;
  int src_c = y*src_stride + x;
  int dst_c = y*dst_stride + x;

  if (x<width && y<height)
  {
    dst[dst_c] = src[src_c] * mul_constant + add_constant;
  }
}


//-----------------------------------------------------------------------------
/** convert kernel 32u_C1 -> 32f_C1 (unsigned int -> float)
 */
__global__ void cuConvert32uC1To32fC1Kernel(const unsigned int *src, size_t src_stride,
                                           float* dst, size_t dst_stride, float mul_constant,
                                           float add_constant, int width, int height)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;
  int src_c = y*src_stride + x;
  int dst_c = y*dst_stride + x;

  if (x<width && y<height)
  {
    dst[dst_c] = src[src_c] * mul_constant + add_constant;
  }
}



/** convert kernel 8u_C3 -> 32f_C4 (unsigned char -> float)
 */
__global__ void cuConvert8uC3To32fC4Kernel(const unsigned char *src, size_t src_pitch,
                                           float4* dst, size_t dst_stride, float mul_constant,
                                           float add_constant, int width, int height)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;
  int src_c = y*src_pitch + x*3;
  int dst_c = y*dst_stride + x;

  if (x<width && y<height)
  {
    dst[dst_c] = make_float4(src[src_c]/255.0f, src[src_c+1]/255.0f, src[src_c+2]/255.0f, 1.0f);// * mul_constant + add_constant;
  }
}


//-----------------------------------------------------------------------------
/** convert kernel 32f_C1 -> 8u_C1 (float -> unsigned char)
 */
__global__ void cuConvert32fC1To8uC1Kernel(const float* src, size_t src_stride,
                                           unsigned char* dst, size_t dst_stride, float mul_constant,
                                           unsigned char add_constant, int width, int height)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;
  int src_c = y*src_stride + x;
  int dst_c = y*dst_stride + x;

  if (x<width && y<height)
  {
    dst[dst_c] = src[src_c] * mul_constant + add_constant;
  }
}

//-----------------------------------------------------------------------------
/** convert kernel 32f_C4 -> 8u_C4 (float4 -> unsigned char4)
 */
__global__ void cuConvert32fC4To8uC4Kernel(const float4* src, size_t src_stride,
                                           uchar4* dst, size_t dst_stride, float mul_constant,
                                           unsigned char add_constant, int width, int height)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;
  int src_c = y*src_stride + x;
  int dst_c = y*dst_stride + x;

  if (x<width && y<height)
  {
    float4 val = src[src_c];
    uchar4 res;
    res.x = val.x * mul_constant + add_constant;
    res.y = val.y * mul_constant + add_constant;
    res.z = val.z * mul_constant + add_constant;
    res.w = val.w * mul_constant + add_constant;
    dst[dst_c] = res;
  }
}

//-----------------------------------------------------------------------------
/** convert kernel rgb -> hsv
 */
__global__ void cuConvertRGBToHSVKernel(const float4* src, float4* dst, size_t stride,
                                        int width, int height, bool normalize)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;
  int c = y*stride + x;

  if (x<width && y<height)
  {
    // Read
    float4 in = src[c];
    float R = in.x;
    float G = in.y;
    float B = in.z;

    float Ma = fmaxf(R, fmaxf(G, B));
    float mi = fminf(R, fminf(G, B));
    float C = Ma-mi;

    // Hue
    float H = 0.0f;
    if (C != 0.0f)
    {
      if (Ma == R)
        H = fmod((G - B)/C, 6.0f);
      if (Ma == G)
        H = (B - R)/C + 2.0f;
      if (Ma == B)
        H = (R - G)/C + 4.0f;
    }

    H *= 60.0f;

    // Value
    float V = Ma;

    // Saturation
    float S = 0.0f;
    if (C != 0.0f)
      S = C/V;

    if (H < 0.0f)
      H += 360.0f;

    // Normalize
    if (normalize)
      H /= 360.0f;

    // Write Back
    dst[c] = make_float4(H, S, V, in.w);
  }
}

//-----------------------------------------------------------------------------
/** convert kernel hsv -> rgb
 */
__global__ void cuConvertHSVToRGBKernel(const float4* src, float4* dst, size_t stride,
                                        int width, int height, bool denormalize)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;
  int c = y*stride + x;

  if (x<width && y<height)
  {
    // Read
    float4 in = src[c];
    float H = in.x;
    float S = in.y;
    float V = in.z;

    float4 rgb = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    //    float C = V*S;

    //    // Denormalize
    //    if (denormalize)
    //      H = H*360.0f;

    //    // RGB
    //    H /= 60.0f;
    //    float X = C*(1.0f - abs(fmod(H, 2.0f) - 1.0f));


    //    if (H >= 0.0f)
    //    {
    //      if (H < 1.0f)
    //        rgb = make_float4(C, X, 0.0f, 0.0f);
    //      else if (H < 2.0f)
    //        rgb = make_float4(X, C, 0.0f, 0.0f);
    //      else if (H < 3.0f)
    //        rgb = make_float4(0.0f, C, X, 0.0f);
    //      else if (H < 4.0f)
    //        rgb = make_float4(0.0f, X, C, 0.0f);
    //      else if (H < 5.0f)
    //        rgb = make_float4(X, 0.0f, C, 0.0f);
    //      else if (H <= 6.0f)
    //        rgb = make_float4(C, 0.0f, X, 0.0f);
    //    }

    //    float m = V-C;
    //    rgb += m;

    if (S == 0)
    {
      rgb = make_float4(V, V, V, in.w);
      dst[c] = rgb;
      return;
    }

     H /= 60.0f;
     int i = floor(H);
     float f = H-i;
     float p = V*(1.0f - S);
     float q = V*(1.0f - S*f);
     float t = V*(1.0f - S*(1.0f-f));

     if (i == 0)
       rgb = make_float4(V, t, p, in.w);
     else if (i == 1)
       rgb = make_float4(q, V, p, in.w);
     else if (i == 2)
       rgb = make_float4(p, V, t, in.w);
     else if (i == 3)
       rgb = make_float4(p, q, V, in.w);
     else if (i == 4)
       rgb = make_float4(t, p, V, in.w);
     else if (i == 5)
       rgb = make_float4(V, p, q, in.w);



    // Write Back
    rgb.w = in.w;
    dst[c] = rgb;
  }
}


//-----------------------------------------------------------------------------
/** convert kernel rgb -> lab
 */
__global__ void cuConvertRGBToLABKernel(const float4* src, float4* dst, size_t stride,
                                        int width, int height, bool isNormalized)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;
  int c = y*stride + x;

  if (x<width && y<height)
  {
    // Read
    float4 in = src[c];
    if (!isNormalized)
    {
      in.x /= 255.0f;
      in.y /= 255.0f;
      in.z /= 255.0f;
      in.w /= 255.0f;
    }
    
    float R = in.x;
    float G = in.y;
    float B = in.z;

    
    // convert to XYZ
    float4 XYZ;
    XYZ.x = 0.4124564f*R + 0.3575761f*G + 0.1804375f*B;
    XYZ.y = 0.2126729f*R + 0.7151522f*G + 0.0721750f*B;
    XYZ.z = 0.0193339f*R + 0.1191920f*G + 0.9503041f*B;
    
    // normalize for D65 white point
    XYZ.x /= 0.950456f;
    XYZ.z /= 1.088754f;
    
    float cubeRootX, cubeRootY, cubeRootZ;
    const float T1 = 216/24389.0f;
    const float T2 = 24389/27.0f;
    
   if (XYZ.x > T1)
      cubeRootX = cbrtf(XYZ.x);
    else
      cubeRootX = (T2 * XYZ.x + 16) / 116;
    
    if (XYZ.y > T1)
      cubeRootY = cbrtf(XYZ.y);
    else
      cubeRootY = (T2 * XYZ.y + 16) / 116;
    
    if (XYZ.z > T1)
      cubeRootZ = cbrtf(XYZ.z);
    else
      cubeRootZ = (T2 * XYZ.z + 16) / 116;
    
    
    
    dst[c] = make_float4(116*cubeRootY-16, 500*(cubeRootX-cubeRootY), 200*(cubeRootY-cubeRootZ), in.w);
  }
}


//-----------------------------------------------------------------------------
/** convert kernel lab -> rgb
 */
__global__ void cuConvertLABToRGBKernel(const float4* src, float4* dst, size_t stride,
                                        int width, int height)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;
  int c = y*stride + x;

  if (x<width && y<height)
  {
    // Read
    float4 in = src[c];
    
    float L = in.x;
    float a = in.y;
    float b = in.z;

    
    // convert to XYZ
    const float T1 = cbrtf(216/24389.0f);
    const float fy = (L+16) / 116.0f;
    
    float4 XYZ;
    if (L > 8)
      XYZ.y = fy*fy*fy;
    else
      XYZ.y = L / (24389/27.0f);
    
    float fx = a/500.0f + fy;
    if (fx > T1)
      XYZ.x = fx*fx*fx;
    else
      XYZ.x = (116*fx-16) / (24389/27.0f);
    
    float fz = fy - b/200.0f;
    if (fz > T1)
      XYZ.z = fz*fz*fz;
    else
      XYZ.z = (116*fz-16) / (24389/27.0f);
    
    
    // Normalize for D65 white point
    XYZ.x *= 0.950456f;
    XYZ.z *= 1.088754f;
    
    float4 rgb;
    rgb.x = 3.2404542f*XYZ.x + -1.5371385f*XYZ.y + -0.4985314f*XYZ.z;
    rgb.y = -0.9692660f*XYZ.x + 1.8760108f*XYZ.y + 0.0415560f*XYZ.z;
    rgb.z = 0.0556434f*XYZ.x + -0.2040259f*XYZ.y + 1.0572252f*XYZ.z;
    rgb.w = in.w;
    
    dst[c] = rgb;
  }
}



/* ***************************************************************************
 *  CUDA WRAPPERS
 * ***************************************************************************/

//-----------------------------------------------------------------------------
/** convert kernel 32f_C3 -> 32f_C4 (float3 -> float4)
 */
void cuConvert(const iu::ImageGpu_32f_C3* src,
                   iu::ImageGpu_32f_C4* dst)
{
    assert(src->size() == dst->size());
  // fragmentation
  const unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(src->width(), dimBlock.x),
               iu::divUp(src->height(), dimBlock.y));

  cuConvertC3ToC4Kernel<<<dimGrid, dimBlock>>>(src->data(), src->stride(),
                                               dst->data(), dst->stride(), src->width(), src->height());

}

//-----------------------------------------------------------------------------
/** convert kernel 32f_C4 -> 32f_C3 (float4 -> float3)
 */
void cuConvert(const iu::ImageGpu_32f_C4* src,
                   iu::ImageGpu_32f_C3* dst)
{
    assert(src->size() == dst->size());
  // fragmentation
  const unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(src->width(), dimBlock.x),
               iu::divUp(src->height(), dimBlock.y));

  cuConvertC4ToC3Kernel<<<dimGrid, dimBlock>>>(src->data(), src->stride(),
                                               dst->data(), dst->stride(), src->width(), src->height());

}


//-----------------------------------------------------------------------------
void cuConvert_8u_32f(const iu::ImageGpu_8u_C1* src,
                          iu::ImageGpu_32f_C1* dst, float mul_constant,
                          float add_constant)
{
    assert(src->size() == dst->size());
  // fragmentation
  const unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(src->width(), dimBlock.x),
               iu::divUp(src->height(), dimBlock.y));

  cuConvert8uC1To32fC1Kernel<<<dimGrid, dimBlock>>>(src->data(), src->stride(), dst->data(), dst->stride(),
                                                    mul_constant, add_constant, src->width(), src->height());

}

void cuConvert_32u_32f(const iu::ImageGpu_32u_C1* src,
                                 iu::ImageGpu_32f_C1* dst,
                                 float mul_constant,  float add_constant)
{
    assert(src->size() == dst->size());
  // fragmentation
  const unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(src->width(), dimBlock.x),
               iu::divUp(src->height(), dimBlock.y));

  cuConvert32uC1To32fC1Kernel<<<dimGrid, dimBlock>>>(src->data(), src->stride(), dst->data(), dst->stride(),
                                                    mul_constant, add_constant, src->width(), src->height());
}

//-----------------------------------------------------------------------------
struct int2float
{
	__host__ __device__ float operator()(int x) const
	{
		return (float)x;
	}
};

void cuConvert_32s_32f_lin(iu::LinearDeviceMemory_32s_C1 *src, iu::LinearDeviceMemory_32f_C1 *dest)
{
	thrust::transform(src->begin(), src->end(), dest->begin(), int2float());
}

//-----------------------------------------------------------------------------
void cuConvert_8u_32f_C3C4(const iu::ImageGpu_8u_C3* src,
                          iu::ImageGpu_32f_C4* dst, float mul_constant,
                          float add_constant)
{
    assert(src->size() == dst->size());
  // fragmentation
  const unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(src->width(), dimBlock.x),
               iu::divUp(src->height(), dimBlock.y));
  
  cuConvert8uC3To32fC4Kernel<<<dimGrid, dimBlock>>>(&(src->data()->x), src->pitch(), dst->data(), dst->stride(),
                                                    mul_constant, add_constant, src->width(), src->height());

}


//-----------------------------------------------------------------------------
void cuConvert_32f_8u(const iu::ImageGpu_32f_C1* src,
                          iu::ImageGpu_8u_C1* dst, float mul_constant,
                          unsigned char add_constant)
{
    assert(src->size() == dst->size());
  // fragmentation
  const unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(src->width(), dimBlock.x),
               iu::divUp(src->height(), dimBlock.y));

  cuConvert32fC1To8uC1Kernel<<<dimGrid, dimBlock>>>(src->data(), src->stride(), dst->data(), dst->stride(),
                                                    mul_constant, add_constant, src->width(), src->height());

}

//-----------------------------------------------------------------------------
void cuConvert_32f_8u(const iu::ImageGpu_32f_C4* src,
                          iu::ImageGpu_8u_C4* dst, float mul_constant,
                          unsigned char add_constant)
{
    assert(src->size() == dst->size());
  // fragmentation
  const unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(src->width(), dimBlock.x),
               iu::divUp(src->height(), dimBlock.y));

  cuConvert32fC4To8uC4Kernel<<<dimGrid, dimBlock>>>(src->data(), src->stride(), dst->data(), dst->stride(),
                                                    mul_constant, add_constant, src->width(), src->height());

}

//-----------------------------------------------------------------------------
void cuConvert_rgb_to_hsv(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst,
                              bool normalize)
{
    assert(src->size() == dst->size());
  // fragmentation
  const unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(src->width(), dimBlock.x),
               iu::divUp(src->height(), dimBlock.y));

  cuConvertRGBToHSVKernel<<<dimGrid, dimBlock>>>(src->data(), dst->data(), src->stride(),
                                                 src->width(), src->height(), normalize);

}

//-----------------------------------------------------------------------------
void cuConvert_hsv_to_rgb(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst,
                              bool denormalize)
{
    assert(src->size() == dst->size());
  // fragmentation
  const unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(src->width(), dimBlock.x),
               iu::divUp(src->height(), dimBlock.y));

  cuConvertHSVToRGBKernel<<<dimGrid, dimBlock>>>(src->data(), dst->data(), src->stride(),
                                                 src->width(), src->height(), denormalize);

}


//-----------------------------------------------------------------------------
void cuConvert_rgb_to_lab(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst, bool isNormalized)
{
    assert(src->size() == dst->size());
  // fragmentation
  const unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(src->width(), dimBlock.x),
               iu::divUp(src->height(), dimBlock.y));

  cuConvertRGBToLABKernel<<<dimGrid, dimBlock>>>(src->data(), dst->data(), src->stride(),
                                                 src->width(), src->height(), isNormalized);

}

//-----------------------------------------------------------------------------
void cuConvert_lab_to_rgb(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst)
{
    assert(src->size() == dst->size());
  // fragmentation
  const unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(src->width(), dimBlock.x),
               iu::divUp(src->height(), dimBlock.y));

  cuConvertLABToRGBKernel<<<dimGrid, dimBlock>>>(src->data(), dst->data(), src->stride(),
                                                 src->width(), src->height());

}





//typedef float num_t;

//__global__ void sum_reduce(iu::ImageGpu_32f_C1::KernelData g_data, num_t* sum)
//{
//    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
//    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
//    const int linId = threadIdx.x + threadIdx.y*blockDim.x;

//    extern volatile __shared__ num_t reduction[];

//    if (x<g_data.width_ && y<g_data.height_)
//    {
//        reduction[linId] = g_data(x,y);
//    }
//    else
//        reduction[linId] = 0;

//    __syncthreads();

//    if (linId < 128) { reduction[linId] += reduction[linId+128]; }
//    __syncthreads();
//    if (linId < 64) { reduction[linId] += reduction[linId+64]; }
//    __syncthreads();

//    if (linId < 32)
//    {
//        reduction[linId] += reduction[linId+32];
//        reduction[linId] += reduction[linId+16];
//        reduction[linId] += reduction[linId+8];
//        reduction[linId] += reduction[linId+4];
//        reduction[linId] += reduction[linId+2];
//        reduction[linId] += reduction[linId+1];
//    }
//    __syncthreads();



//    if (linId == 0)
//        atomicAdd(sum, reduction[0]);
//}


//double cuSummation(iu::ImageGpu_32f_C1* src)
//{
//    // fragmentation
//    unsigned int block_size = 16;
//    dim3 dimBlock(block_size, block_size);
//    dim3 dimGrid(iu::divUp(src->width(), dimBlock.x), iu::divUp(src->height(), dimBlock.y));

//    num_t* s;
//    cudaMalloc(&s, sizeof(num_t));
//    cudaMemset(s, 0, sizeof(num_t));

//    int shared_sz = block_size*block_size*sizeof(num_t);
//    sum_reduce<<< dimGrid, dimBlock, shared_sz >>> (*src, s);

//    num_t sum = 0;
//    cudaMemcpy(&sum, s, sizeof(num_t), cudaMemcpyDeviceToHost);

//    cudaFree(s);
//    return sum;
//}


} // namespace iuprivate


