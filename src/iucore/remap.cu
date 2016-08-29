//#include <iostream>
#include <iudefs.h>
#include <iucutil.h>
#include <iucore/iutextures.cuh>
#include <common/bind_textures.cuh>
#include <common/bsplinetexture_kernels.cuh>


namespace iuprivate {

// local textures
texture<float, 2, cudaReadModeElementType> tex_remap_dx_32f_C1__;
texture<float, 2, cudaReadModeElementType> tex_remap_dy_32f_C1__;

///////////////////////////////////////////////////////////////////////////////
// 32f_C4
///////////////////////////////////////////////////////////////////////////////

// linear interpolation
// 32f_C1
__global__ void cuRemapKernel_32f_C1(float *dst, size_t stride, int width, int height)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;

  // texutre coordinates
  const float xx = x+0.5f;
  const float yy = y+0.5f;
  // warped texutre coordinates
  const float wx = xx + tex2D(tex_remap_dx_32f_C1__, xx, yy);
  const float wy = yy + tex2D(tex_remap_dy_32f_C1__, xx, yy);

  if (x<width && y<height) // Check if out coordinates lie inside output image
  {
    dst[y*stride+x] = tex2D(tex1_32f_C1__, wx, wy);
  }
}

// cubic interpolation
//__global__ void cuRemapCubicKernel_32f_C1(float *dst, size_t stride, int width, int height)
//{
//  const int x = blockIdx.x*blockDim.x + threadIdx.x;
//  const int y = blockIdx.y*blockDim.y + threadIdx.y;

//  // texutre coordinates
//  const float xx = x+0.5f;
//  const float yy = y+0.5f;
//  // warped texutre coordinates
//  const float wx = xx + tex2D(tex_remap_dx_32f_C1__, xx, yy);
//  const float wy = yy + tex2D(tex_remap_dy_32f_C1__, xx, yy);

//  if (x<width && y<height) // Check if out coordinates lie inside output image
//  {
//    dst[y*stride+x] = iu::cubicTex2DSimple(tex1_32f_C1__, wx, wy);
//  }
//}
// cubic spline interpolation
//__global__ void cuRemapCubicSplineKernel_32f_C1(float *dst, size_t stride, int width, int height)
//{
//  const int x = blockIdx.x*blockDim.x + threadIdx.x;
//  const int y = blockIdx.y*blockDim.y + threadIdx.y;

//  // texutre coordinates
//  const float xx = x+0.5f;
//  const float yy = y+0.5f;
//  // warped texutre coordinates
//  const float wx = xx + tex2D(tex_remap_dx_32f_C1__, xx, yy);
//  const float wy = yy + tex2D(tex_remap_dy_32f_C1__, xx, yy);

//  if (x<width && y<height) // Check if out coordinates lie inside output image
//  {
//    dst[y*stride+x] = iu::cubicTex2D(tex1_32f_C1__, wx, wy);
//  }
//}


//-----------------------------------------------------------------------------
void cuRemap(iu::ImageGpu_32f_C1* src,
             iu::ImageGpu_32f_C1* dx_map, iu::ImageGpu_32f_C1* dy_map,
             iu::ImageGpu_32f_C1* dst, IuInterpolationType interpolation)
{
  tex1_32f_C1__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C1__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C1__.normalized = false;

  tex_remap_dx_32f_C1__.addressMode[0] = cudaAddressModeClamp;
  tex_remap_dx_32f_C1__.addressMode[1] = cudaAddressModeClamp;
  tex_remap_dx_32f_C1__.normalized = false;
  tex_remap_dx_32f_C1__.filterMode = cudaFilterModePoint;

  tex_remap_dy_32f_C1__.addressMode[0] = cudaAddressModeClamp;
  tex_remap_dy_32f_C1__.addressMode[1] = cudaAddressModeClamp;
  tex_remap_dy_32f_C1__.normalized = false;
  tex_remap_dy_32f_C1__.filterMode = cudaFilterModePoint;


  // bind src image to texture and use as input for reduction
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
  cudaBindTexture2D(0, &tex1_32f_C1__, src->data(), &channel_desc, src->width(), src->height(), src->pitch());
  cudaBindTexture2D(0, &tex_remap_dx_32f_C1__, dx_map->data(), &channel_desc, dx_map->width(), dx_map->height(), dx_map->pitch());
  cudaBindTexture2D(0, &tex_remap_dy_32f_C1__, dy_map->data(), &channel_desc, dy_map->width(), dy_map->height(), dy_map->pitch());

  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGridOut(iu::divUp(dst->width(), dimBlock.x), iu::divUp(dst->height(), dimBlock.y));

  switch(interpolation)
  {
  case IU_INTERPOLATE_NEAREST:
    tex1_32f_C1__.filterMode = cudaFilterModePoint;
    break;
  case IU_INTERPOLATE_LINEAR:
    tex1_32f_C1__.filterMode = cudaFilterModeLinear;
    break;
  }

  switch(interpolation)
  {
  case IU_INTERPOLATE_NEAREST:
  case IU_INTERPOLATE_LINEAR: // fallthrough intended
    cuRemapKernel_32f_C1 <<< dimGridOut, dimBlock >>> (
                                                      dst->data(), dst->stride(), dst->width(), dst->height());
    break;
//  case IU_INTERPOLATE_CUBIC:
//    cuRemapCubicKernel_32f_C1 <<< dimGridOut, dimBlock >>> (
//                                                           dst->data(), dst->stride(), dst->width(), dst->height());
//    break;
//  case IU_INTERPOLATE_CUBIC_SPLINE:
//    cuRemapCubicSplineKernel_32f_C1 <<< dimGridOut, dimBlock >>> (
//                                                                 dst->data(), dst->stride(), dst->width(), dst->height());
//    break;
  }

  cudaUnbindTexture(&tex1_32f_C1__);
  cudaUnbindTexture(&tex_remap_dx_32f_C1__);
  cudaUnbindTexture(&tex_remap_dy_32f_C1__);

//  IU_CUDA_CHECK();
}

///////////////////////////////////////////////////////////////////////////////
// 32f_C4
///////////////////////////////////////////////////////////////////////////////

// 32f_C1
__global__ void cuRemapKernel_32f_C4(float4 *dst, size_t stride, int width, int height)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;

  // texutre coordinates
  const float xx = x+0.5f;
  const float yy = y+0.5f;
  // warped texutre coordinates
  const float wx = xx + tex2D(tex_remap_dx_32f_C1__, xx, yy);
  const float wy = yy + tex2D(tex_remap_dy_32f_C1__, xx, yy);

  if (x<width && y<height) // Check if out coordinates lie inside output image
  {
    dst[y*stride+x] = tex2D(tex1_32f_C4__, wx, wy);
  }
}

//-----------------------------------------------------------------------------
void cuRemap(iu::ImageGpu_32f_C4* src,
             iu::ImageGpu_32f_C1* dx_map, iu::ImageGpu_32f_C1* dy_map,
             iu::ImageGpu_32f_C4* dst, IuInterpolationType interpolation)
{
  cudaTextureFilterMode filter_mode = cudaFilterModePoint;

  switch(interpolation)
  {
  case IU_INTERPOLATE_LINEAR:
    filter_mode = cudaFilterModeLinear;
    break;
  case IU_INTERPOLATE_NEAREST:
  default:
    filter_mode = cudaFilterModePoint;
    break;
  }

  // bind src image to texture and use as input for reduction
  bindTexture(tex1_32f_C4__, src, filter_mode);
  bindTexture(tex_remap_dx_32f_C1__, dx_map);
  bindTexture(tex_remap_dy_32f_C1__, dy_map);

  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGridOut(iu::divUp(dst->width(), dimBlock.x), iu::divUp(dst->height(), dimBlock.y));


    cuRemapKernel_32f_C4
        <<< dimGridOut, dimBlock >>> (dst->data(), dst->stride(), dst->width(), dst->height());

//  IU_CUDA_CHECK();
}


//-----------------------------------------------------------------------------
// 8u_C1
__global__ void cuRemapLinearInterpKernel_8u_C1(unsigned char*dst, size_t stride, int width, int height)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;

  // texutre coordinates
  const float xx = x+0.5f;
  const float yy = y+0.5f;
  // warped texutre coordinates
  const float wx = xx + tex2D(tex_remap_dx_32f_C1__, xx, yy);
  const float wy = yy + tex2D(tex_remap_dy_32f_C1__, xx, yy);

  if (x<width && y<height) // Check if out coordinates lie inside output image
  {
    int wx1 = fmaxf(0, static_cast<int>(wx));
    int wx2 = fminf(width, wx1+1);
    int wy1 = fmaxf(0, static_cast<int>(wy));
    int wy2 = fminf(height, wy1+1);
    float dx = wx2-xx;
    float dy = wy2-yy;

    float val1 = dx*dy*static_cast<float>(tex2D(tex1_8u_C1__,wx1,wy1))/255.0f;
    float val2 = dx*(1-dy)*static_cast<float>(tex2D(tex1_8u_C1__,wx1,wy2))/255.0f;
    float val3 = (1-dx)*dy*static_cast<float>(tex2D(tex1_8u_C1__,wx2,wy1))/255.0f;
    float val4 = (1-dx)*(1-dy)*static_cast<float>(tex2D(tex1_8u_C1__,wx2,wy2))/255.0f;

    dst[y*stride+x] = (val1 + val2 + val3 + val4) * 255;
    //dst[y*stride+x] = tex2D(tex1_8u_C1__, wx, wy);
  }
}

//-----------------------------------------------------------------------------
// 8u_C1
__global__ void cuRemapPointInterpKernel_8u_C1(unsigned char*dst, size_t stride, int width, int height)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;

  // texutre coordinates
  const float xx = x+0.5f;
  const float yy = y+0.5f;
  // warped texutre coordinates
  const float wx = xx + tex2D(tex_remap_dx_32f_C1__, xx, yy);
  const float wy = yy + tex2D(tex_remap_dy_32f_C1__, xx, yy);

  if (x<width && y<height) // Check if out coordinates lie inside output image
  {
    dst[y*stride+x] = tex2D(tex1_8u_C1__, wx, wy);
  }
}

//-----------------------------------------------------------------------------
void cuRemap(iu::ImageGpu_8u_C1* src,
             iu::ImageGpu_32f_C1* dx_map, iu::ImageGpu_32f_C1* dy_map,
             iu::ImageGpu_8u_C1* dst, IuInterpolationType interpolation)
{
  tex1_8u_C1__.addressMode[0] = cudaAddressModeClamp;
  tex1_8u_C1__.addressMode[1] = cudaAddressModeClamp;
  tex1_8u_C1__.normalized = false;
  tex1_8u_C1__.filterMode = cudaFilterModePoint;

  tex_remap_dx_32f_C1__.addressMode[0] = cudaAddressModeClamp;
  tex_remap_dx_32f_C1__.addressMode[1] = cudaAddressModeClamp;
  tex_remap_dx_32f_C1__.normalized = false;
  tex_remap_dx_32f_C1__.filterMode = cudaFilterModePoint;

  tex_remap_dy_32f_C1__.addressMode[0] = cudaAddressModeClamp;
  tex_remap_dy_32f_C1__.addressMode[1] = cudaAddressModeClamp;
  tex_remap_dy_32f_C1__.normalized = false;
  tex_remap_dy_32f_C1__.filterMode = cudaFilterModePoint;


  // bind src image to texture and use as input for reduction
  cudaChannelFormatDesc channel_desc_8u_C1 = cudaCreateChannelDesc<unsigned char>();
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
  cudaBindTexture2D(0, &tex1_8u_C1__, src->data(), &channel_desc_8u_C1, src->width(), src->height(), src->pitch());
  cudaBindTexture2D(0, &tex_remap_dx_32f_C1__, dx_map->data(), &channel_desc, dx_map->width(), dx_map->height(), dx_map->pitch());
  cudaBindTexture2D(0, &tex_remap_dy_32f_C1__, dy_map->data(), &channel_desc, dy_map->width(), dy_map->height(), dy_map->pitch());

  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGridOut(iu::divUp(dst->width(), dimBlock.x), iu::divUp(dst->height(), dimBlock.y));

    switch(interpolation)
    {
    case IU_INTERPOLATE_NEAREST:
      tex1_8u_C1__.filterMode = cudaFilterModePoint;
      break;
    case IU_INTERPOLATE_LINEAR:
      tex1_8u_C1__.filterMode = cudaFilterModeLinear;
      break;
    }

  //  switch(interpolation)
  //  {
  //  case IU_INTERPOLATE_LINEAR: // fallthrough intended
  //    cuRemapLinearInterpKernel_8u_C1 <<< dimGridOut, dimBlock >>> (
  //        dst->data(), dst->stride(), dst->width(), dst->height());
  //    break;
  //  default:
  //  case IU_INTERPOLATE_NEAREST:
  cuRemapPointInterpKernel_8u_C1 <<< dimGridOut, dimBlock >>> (
                                                              dst->data(), dst->stride(), dst->width(), dst->height());
  //    break;
  //  }

  cudaUnbindTexture(&tex1_8u_C1__);
  cudaUnbindTexture(&tex_remap_dx_32f_C1__);
  cudaUnbindTexture(&tex_remap_dy_32f_C1__);

//  IU_CUDA_CHECK();
}


__global__ void cuRemapAffineKernel_32f_C1(float *dst, int stride, int width, int height,
                                           float a1, float a2, float a3, float a4, 
                                           float b1, float  b2)
{
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;
  
  // texutre coordinates
  const float xx = x;
  const float yy = y;
  // warped texutre coordinates
  const float wx = a1*xx + a2*yy + b1 + 0.5f;
  const float wy = a3*xx + a4*yy + b2 + 0.5f;

  if (x<width && y<height) // Check if out coordinates lie inside output image
  {
    dst[y*stride+x] = tex2D(tex1_32f_C1__, wx, wy);
  }
}

void cuRemapAffine(iu::ImageGpu_32f_C1* src,
                          float a1, float a2, float a3, float a4,
                          float b1, float b2,
                          iu::ImageGpu_32f_C1* dst)
{
  tex1_32f_C1__.addressMode[0] = cudaAddressModeClamp;
  tex1_32f_C1__.addressMode[1] = cudaAddressModeClamp;
  tex1_32f_C1__.normalized = false;
  tex1_32f_C1__.filterMode = cudaFilterModeLinear;


  // bind src image to texture and use as input for reduction
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
  cudaBindTexture2D(0, &tex1_32f_C1__, src->data(), &channel_desc, src->width(), src->height(), src->pitch());
  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGridOut(iu::divUp(dst->width(), dimBlock.x), iu::divUp(dst->height(), dimBlock.y));

  cuRemapAffineKernel_32f_C1 <<< dimGridOut, dimBlock >>> 
    (dst->data(), dst->stride(), dst->width(), dst->height(),
     a1, a2, a3, a4, b1, b2);

  cudaUnbindTexture(&tex1_32f_C1__);

//  IU_CUDA_CHECK();
}

} // namespace iuprivate


