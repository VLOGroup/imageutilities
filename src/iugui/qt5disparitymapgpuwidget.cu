
#include "../iucore.h"



__global__ void to_vbo_kernel(iu::ImageGpu_32f_C1::KernelData disparities, iu::ImageGpu_32f_C1::KernelData color, float f, float cx, float cy, float B, float point_size, iu::ImageGpu_32f_C4::KernelData out)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x<disparities.width_&& y<disparities.height_)
    {
        float Z;
        if(B<10e-6f)
            Z = disparities(x,y);
        else
            Z = f*B/disparities(x,y);
        float X = (x-cx)*Z/f;
        float Y = (y-cy)*Z/f;
        float c = color(x,y);

        out(x*6,  y) = make_float4(X-point_size, Y-point_size, Z, c);
        out(x*6+1,y) = make_float4(X+point_size, Y-point_size, Z, c);
        out(x*6+2,y) = make_float4(X+point_size, Y+point_size, Z, c);
        out(x*6+3,y) = make_float4(X+point_size, Y+point_size, Z, c);
        out(x*6+4,y) = make_float4(X-point_size, Y+point_size, Z, c);
        out(x*6+5,y) = make_float4(X-point_size, Y-point_size, Z, c);
    }
}



namespace iuprivate {

void copy_to_VBO(iu::ImageGpu_32f_C1& disparitites, iu::ImageGpu_32f_C1& color, float f, float cx, float cy, float B, float point_size, iu::ImageGpu_32f_C4& vbo)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid(iu::divUp(disparitites.width(), dimBlock.x),
                 iu::divUp(disparitites.height(), dimBlock.y));
    to_vbo_kernel <<< dimGrid, dimBlock >>> (disparitites,color,f,cx,cy,B,point_size,vbo);

}

} // namespace
