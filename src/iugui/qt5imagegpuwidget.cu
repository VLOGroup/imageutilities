
#include "../iucore.h"



__global__ void to_pbo_kernel(iu::ImageGpu_8u_C1::KernelData in, iu::ImageGpu_8u_C4::KernelData out)
{
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x<in.width_ && y<in.height_)
    {
        unsigned char value = in(x,y);
        out(x,y) = make_uchar4(value, value, value, 1);
    }
}

__global__ void to_pbo_kernel(iu::ImageGpu_8u_C4::KernelData in, iu::ImageGpu_8u_C4::KernelData out)
{
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x<in.width_ && y<in.height_)
    {
        out(x,y) = in(x,y);
    }
}

__global__ void to_pbo_kernel1(unsigned char* g_in, int stride_in, uchar4* g_out, int stride_out,
                               int width, int height)
{
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x<width && y<height)
    {
        unsigned char value = g_in[y*stride_in+x];
        g_out[y*stride_out+x] = make_uchar4(value, value, value, 1);
    }
}

__global__ void to_pbo_kernel(iu::ImageGpu_32f_C1::KernelData in, iu::ImageGpu_8u_C4::KernelData out, float minVal, float maxVal)
{
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x<in.width_ && y<in.height_)
    {
        float value = (in(x,y)-minVal)/(maxVal-minVal)*255.f;
        out(x,y) = make_uchar4(value, value, value, 1);
    }
}

__global__ void to_pbo_jet_kernel(iu::ImageGpu_32f_C1::KernelData in, iu::ImageGpu_8u_C4::KernelData out, float minVal, float maxVal)
{
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x<in.width_ && y<in.height_)
    {
        float value = (in(x,y)-minVal)/(maxVal-minVal)*4.f;
        out(x,y) = make_uchar4(255.f*min(max(min(value-1.5f,-value+4.5f),0.f),1.f),
                               255.f*min(max(min(value-0.5f,-value+3.5f),0.f),1.f),
                               255.f*min(max(min(value+0.5f,-value+2.5f),0.f),1.f),1);
    }
}


namespace iuprivate {

void copy_to_PBO(iu::ImageGpu_8u_C1& img, iu::ImageGpu_8u_C4& pbo)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid(iu::divUp(img.width(), dimBlock.x),
                 iu::divUp(img.height(), dimBlock.y));

//    to_pbo_kernel <<< dimGrid, dimBlock >>> (img, pbo);
    to_pbo_kernel1 <<< dimGrid, dimBlock >>> (img.data(), img.stride(), pbo.data(), pbo.stride(),
                                              img.width(), img.height());
}

void copy_to_PBO(iu::ImageGpu_8u_C4& img, iu::ImageGpu_8u_C4& pbo)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid(iu::divUp(img.width(), dimBlock.x),
                 iu::divUp(img.height(), dimBlock.y));

    to_pbo_kernel <<< dimGrid, dimBlock >>> (img, pbo);
}

void copy_to_PBO(iu::ImageGpu_32f_C1& img, iu::ImageGpu_8u_C4& pbo, float minVal, float maxVal, bool colormap)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid(iu::divUp(img.width(), dimBlock.x),
                 iu::divUp(img.height(), dimBlock.y));

    if(!colormap)
        to_pbo_kernel <<< dimGrid, dimBlock >>> (img, pbo, minVal, maxVal);
    else
        to_pbo_jet_kernel <<< dimGrid, dimBlock >>> (img, pbo, minVal, maxVal);
//    to_pbo_kernel1 <<< dimGrid, dimBlock >>> (img.data(), img.stride(), pbo.data(), pbo.stride(),
//                                              img.width(), img.height());
}


} // namespace
