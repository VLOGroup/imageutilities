#ifndef IUSPARSECOMPARE_CU
#define IUSPARSECOMPARE_CU

#include <list>
#include <iostream>
#include <iudefs.h>
#include <iucore.h>
#include <iucutil.h>
#include <iumath.h>
#include <iusparse.h>

// textures
texture<float, 2, cudaReadModeElementType> rof_tex_u;
texture<float, 2, cudaReadModeElementType> rof_tex_u_;
texture<float, 2, cudaReadModeElementType> rof_tex_f;
texture<float2, 2, cudaReadModeElementType> rof_tex_p;
texture<float2, 2, cudaReadModeElementType> rof_tex_gradient;
texture<float, 2, cudaReadModeElementType> rof_tex_divergence;

const unsigned int BSX=32;
const unsigned int BSY=16;

////////////////////////////////////////////////////////////////////////////////
void bindTexture(texture<float, 2>& tex, iu::ImageGpu_32f_C1* mem)
{
  tex.addressMode[0] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.addressMode[1] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.filterMode = cudaFilterModeLinear;
  tex.normalized = false;
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
  ( cudaBindTexture2D( 0, &tex, mem->data(), &channel_desc,
                      mem->width(), mem->height(), mem->pitch()));
}

////////////////////////////////////////////////////////////////////////////////
void bindTexture(texture<float2, 2>& tex, iu::ImageGpu_32f_C2* mem)
{
  tex.addressMode[0] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.addressMode[1] = cudaAddressModeClamp; // Neumann Boundary Conditions
  tex.filterMode = cudaFilterModeLinear;
  tex.normalized = false;
  cudaChannelFormatDesc channel_desc_float2 = cudaCreateChannelDesc<float2>();
  ( cudaBindTexture2D( 0, &tex, mem->data(), &channel_desc_float2,
                      mem->width(), mem->height(), mem->pitch()));
}

////////////////////////////////////////////////////////////////////////////////
//############################################################################//
//OOOOOOOOOOOOOOOOOOOOOOOOOOO  STANDARD MODEL  OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO//
//############################################################################//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
__global__ void update_primal(float* device_u, float* device_u_,
                              float tau, float theta, float lambda,
                              int width, int height, int xstride)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  int c = y*xstride + x;

  if(x<width && y<height)
  {
    // texture fetches
    float f = tex2D(rof_tex_f, x+0.5f, y+0.5f);
    float u = tex2D(rof_tex_u, x+0.5f, y+0.5f);
    float u_ = u;

    float divergence = dp_ad(rof_tex_p, x, y, width, height);

    // update primal variable
    u = (u + tau*(divergence + lambda*f))/(1.0f+tau*lambda);

    device_u[c] = u;
    device_u_[c] = u + theta*(u-u_);
  }
}

////////////////////////////////////////////////////////////////////////////////
__global__ void update_dual(float2* device_p, float sigma,
                            int width, int height, int stride)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  if(x<width && y<height)
  {
    float2 p = tex2D(rof_tex_p, x+0.5f, y+0.5f);
    float2 grad_u = dp(rof_tex_u_, x, y);

    // update dual variable
    p = p + sigma*grad_u;
    p = p/max(1.0f, length(p));

    device_p[y*stride + x] = p;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void rof_primal_dual(iu::ImageGpu_32f_C1* device_f, iu::ImageGpu_32f_C1* device_u,
                     iu::ImageGpu_32f_C1* device_u_, iu::ImageGpu_32f_C2* device_p,
                     float lambda, int max_iter)
{
  int width = device_f->width();
  int height = device_f->height();

  // fragmentation
  int nb_x = width/BSX;
  int nb_y = height/BSY;
  if (nb_x*BSX < width) nb_x++;
  if (nb_y*BSY < height) nb_y++;

  dim3 dimBlock(BSX,BSY);
  dim3 dimGrid(nb_x,nb_y);

  bindTexture(rof_tex_f, device_f);
  bindTexture(rof_tex_u, device_u);
  bindTexture(rof_tex_u_, device_u_);
  bindTexture(rof_tex_p, device_p);

  float L = sqrtf(8.0f);
  float tau = 1/L;
  float sigma = 1/L;
  float theta;

  for (int k = 0; k < max_iter; ++k)
  {
    update_dual<<<dimGrid, dimBlock>>>(device_p->data(), sigma,
                                       width, height, device_p->stride());

    if (sigma < 1000.0f)
      theta = 1/sqrtf(1.0f+0.7f*lambda*tau);
    else
      theta = 1.0f;

    update_primal<<<dimGrid, dimBlock>>>(device_u->data(), device_u_->data(),
                                         tau, theta, lambda, width, height,
                                         device_u->stride());
    sigma /= theta;
    tau *= theta;
  }
  IuStatus status = iu::checkCudaErrorState(true);
  if(status != IU_NO_ERROR)
  {
    std::cerr << "An error occured while solving the ROF model." << std::endl;
  }
}

////////////////////////////////////////////////////////////////////////////////
//############################################################################//
//OOOOOOOOOOOOOOOOOOOOOOOOOOOOO  NO TEXTURE  OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO//
//############################################################################//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
__global__ void update_primal_notex(float* device_u, float* device_u_,
                                    float* device_f, float2* device_p,
                                    float tau, float theta, float lambda,
                                    int width, int height, int xstride, int xstride2)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  int c = y*xstride + x;

  if(x<width && y<height)
  {
    // texture fetches
    float f = device_f[c];
    float u = device_u[c];
    float u_ = u;

    float2 cval = device_p[y*xstride2 + x];
    float2 wval = device_p[y*xstride2 + max(0,x-1)];
    float2 nval = device_p[max(0,y-1)*xstride2 + x];
    if (x == 0)
      wval.x = 0.0f;
    else if (x >= width-1)
      cval.x = 0.0f;
    if (y == 0)
      nval.y = 0.0f;
    else if (y >= height-1)
      cval.y = 0.0f;
    float divergence = cval.x - wval.x + cval.y - nval.y;

    // update primal variable
    u = (u + tau*(divergence + lambda*f))/(1.0f+tau*lambda);

    device_u[c] = u;
    device_u_[c] = u + theta*(u-u_);
  }
}

////////////////////////////////////////////////////////////////////////////////
__global__ void update_dual_notex(float2* device_p, float* device_u_, float sigma,
                                  int width, int height, int xstride, int xstride2)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  int c2 = y*xstride2 + x;

  if(x<width && y<height)
  {
    float2 p = device_p[c2];
    float2 grad_u = make_float2(0.0f, 0.0f);
    float cval = device_u_[y*xstride + x];
    grad_u.x = device_u_[y*xstride + min(width-1,x+1)] - cval;
    grad_u.y = device_u_[min(height-1,y+1)*xstride + x] - cval;

    // update dual variable
    p = p + sigma*grad_u;
    p = p/max(1.0f, length(p));

    device_p[c2] = p;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void rof_primal_dual_notex(iu::ImageGpu_32f_C1* device_f, iu::ImageGpu_32f_C1* device_u,
                           iu::ImageGpu_32f_C1* device_u_, iu::ImageGpu_32f_C2* device_p,
                           float lambda, int max_iter)
{
  int width = device_f->width();
  int height = device_f->height();

  // fragmentation
  int nb_x = width/BSX;
  int nb_y = height/BSY;
  if (nb_x*BSX < width) nb_x++;
  if (nb_y*BSY < height) nb_y++;

  dim3 dimBlock(BSX,BSY);
  dim3 dimGrid(nb_x,nb_y);

  float L = sqrtf(8.0f);
  float tau = 1/L;
  float sigma = 1/L;
  float theta;


cudaFuncSetCacheConfig("_Z17update_dual_notexP6float2Pffiiii", cudaFuncCachePreferShared);
IuStatus status = iu::checkCudaErrorState(true);
if(status != IU_NO_ERROR)
{
  std::cerr << "An error occured while solving the ROF model." << std::endl;
}
cudaFuncSetCacheConfig("_Z19update_primal_notexPfS_S_P6float2fffiiii", cudaFuncCachePreferShared);
status = iu::checkCudaErrorState(true);
if(status != IU_NO_ERROR)
{
  std::cerr << "An error occured while solving the ROF model." << std::endl;
}

  for (int k = 0; k < max_iter; ++k)
  {
    update_dual_notex<<<dimGrid, dimBlock>>>(device_p->data(), device_u_->data(),
                                             sigma, width, height,
                                             device_u_->stride(), device_p->stride());

    if (sigma < 1000.0f)
      theta = 1/sqrtf(1.0f+0.7f*lambda*tau);
    else
      theta = 1.0f;

    update_primal_notex<<<dimGrid, dimBlock>>>(device_u->data(), device_u_->data(),
                                               device_f->data(), device_p->data(),
                                               tau, theta, lambda, width, height,
                                               device_u->stride(), device_p->stride());
    sigma /= theta;
    tau *= theta;
  }
  status = iu::checkCudaErrorState(true);
  if(status != IU_NO_ERROR)
  {
    std::cerr << "An error occured while solving the ROF model." << std::endl;
  }
}

////////////////////////////////////////////////////////////////////////////////
//############################################################################//
//OOOOOOOOOOOOOOOOOOOOOOOOOOO  SPARSE MATRIX  OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO//
//############################################################################//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
__global__ void update_primal_sparse(float* device_u, float* device_u_,
                                     float tau, float theta, float lambda,
                                     int width, int height, int xstride)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  int c = y*xstride + x;

  if(x<width && y<height)
  {
    // texture fetches
    float u = tex2D(rof_tex_u, x+0.5f, y+0.5f);
    float u_ = u;

    // update primal variable
    u = (u + tau*(-tex2D(rof_tex_divergence, x+0.5f, y+0.5f) + lambda*tex2D(rof_tex_f, x+0.5f, y+0.5f)))/(1.0f+tau*lambda);

    device_u[c] = u;
    device_u_[c] = u + theta*(u-u_);
  }
}

////////////////////////////////////////////////////////////////////////////////
__global__ void update_dual_sparse(float2* device_p,  float sigma,
                                   int width, int height, int stride)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  if(x<width && y<height)
  {
    float2 p = tex2D(rof_tex_p, x+0.5f, y+0.5f) + sigma*tex2D(rof_tex_gradient, x+0.5f, y+0.5f);
    device_p[y*stride + x] = p/max(1.0f, length(p));
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void rof_primal_dual_sparse(iu::SparseMatrixGpu_32f* G,
                            iu::ImageGpu_32f_C1* device_f, iu::ImageGpu_32f_C1* device_u,
                            iu::ImageGpu_32f_C1* device_u_, iu::ImageGpu_32f_C2* device_p,
                            float lambda, int max_iter)
{
  int width = device_f->width();
  int height = device_f->height();

  // fragmentation
  int nb_x = width/BSX;
  int nb_y = height/BSY;
  if (nb_x*BSX < width) nb_x++;
  if (nb_y*BSY < height) nb_y++;

  dim3 dimBlock(BSX,BSY);
  dim3 dimGrid(nb_x,nb_y);

  iu::ImageGpu_32f_C2 gradient(width, height);
  iu::ImageGpu_32f_C1 divergence(width, height);

  float L = sqrtf(8.0f);
  float tau = 1/L;
  float sigma = 1/L;
  float theta;

  for (int k = 0; k < max_iter; ++k)
  {
    // Gradient
    iu::sparseMultiplication(G->handle(), G, device_u_, &gradient);

    // Dual update
    bindTexture(rof_tex_p, device_p);
    bindTexture(rof_tex_gradient, &gradient);
    update_dual_sparse<<< dimGrid, dimBlock >>>(device_p->data(), sigma,
                                                width, height, device_p->stride());
    cudaUnbindTexture(&rof_tex_p);
    cudaUnbindTexture(&rof_tex_gradient);


    // Update Timesteps
    if (sigma < 1000.0f)
      theta = 1/sqrtf(1.0f+0.7f*lambda*tau);
    else
      theta = 1.0f;

    // Divergence
    iu::sparseMultiplication(G->handle(), G, device_p, &divergence, CUSPARSE_OPERATION_TRANSPOSE);

    // Primal update
    bindTexture(rof_tex_f, device_f);
    bindTexture(rof_tex_u, device_u);
    bindTexture(rof_tex_divergence, &divergence);
    update_primal_sparse<<< dimGrid, dimBlock >>>(device_u->data(), device_u_->data(),
                                                  tau, theta, lambda, width, height,
                                                  device_u->stride());
    cudaUnbindTexture(&rof_tex_f);
    cudaUnbindTexture(&rof_tex_u);
    cudaUnbindTexture(&rof_tex_divergence);

    // Update Timesteps
    sigma /= theta;
    tau *= theta;
  }
  IuStatus status = iu::checkCudaErrorState(true);
  if(status != IU_NO_ERROR)
  {
    std::cerr << "An error occured while solving the ROF model." << std::endl;
  }
}

////////////////////////////////////////////////////////////////////////////////
//############################################################################//
//OOOOOOOOOOOOOOOOOOOOOOOOOOO  SHARED MEMORY  OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO//
//############################################################################//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
__global__ void update_primal_shared(float* device_u, float* device_u_,
                                     float tau, float theta, float lambda,
                                     int width, int height, int xstride)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  int c = y*xstride + x;

  if(x<width && y<height)
  {
    // texture fetches
    float f = tex2D(rof_tex_f, x+0.5f, y+0.5f);
    float u = tex2D(rof_tex_u, x+0.5f, y+0.5f);
    float u_ = u;

    __shared__ float2 p[BSX+1][BSY+1];

    p[threadIdx.x+1][threadIdx.y+1] = tex2D(rof_tex_p, x+0.5f, y+0.5f);

    if (threadIdx.x==0)
    {
      p[threadIdx.x][threadIdx.y+1] = tex2D(rof_tex_p, x-0.5f, y+0.5f);
      if (x==0)
        p[threadIdx.x][threadIdx.y+1].x = 0.0f;
    }
    if (threadIdx.y==0)
    {
      p[threadIdx.x+1][threadIdx.y] = tex2D(rof_tex_p, x+0.5f, y-0.5f);
      if (y==0)
        p[threadIdx.x+1][threadIdx.y].y = 0.0f;
    }

    if (x >= width-1)
      p[threadIdx.x+1][threadIdx.y+1].x = 0.0f;
    if (y >= height-1)
      p[threadIdx.x+1][threadIdx.y+1].y = 0.0f;

    //        syncthreads();

    float divergence = p[threadIdx.x+1][threadIdx.y+1].x - p[threadIdx.x][threadIdx.y+1].x +
        p[threadIdx.x+1][threadIdx.y+1].y - p[threadIdx.x+1][threadIdx.y].y;

    // update primal variable
    u = (u + tau*(divergence + lambda*f))/(1.0f+tau*lambda);

    device_u[c] = u;
    device_u_[c] = u + theta*(u-u_);
  }
}

////////////////////////////////////////////////////////////////////////////////
__global__ void update_dual_shared(float2* device_p, float sigma,
                                   int width, int height, int stride)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  if(x<width && y<height)
  {
    float2 p = tex2D(rof_tex_p, x+0.5f, y+0.5f);

    __shared__ float u_[BSX+1][BSY+1];
    u_[threadIdx.x][threadIdx.y] = tex2D(rof_tex_u_, x+0.5f, y+0.5f);

    if (threadIdx.x==BSX-1)
      u_[threadIdx.x+1][threadIdx.y] = tex2D(rof_tex_u_, x+1.5f, y+0.5f);
    if (threadIdx.y==BSY-1)
      u_[threadIdx.x][threadIdx.y+1] = tex2D(rof_tex_u_, x+0.5f, y+1.5f);

    //    syncthreads();

    float2 grad_u = make_float2(u_[threadIdx.x+1][threadIdx.y] - u_[threadIdx.x][threadIdx.y],
                                u_[threadIdx.x][threadIdx.y+1] - u_[threadIdx.x][threadIdx.y] );

    // update dual variable
    p = p + sigma*grad_u;
    p = p/max(1.0f, length(p));

    device_p[y*stride + x] = p;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void rof_primal_dual_shared(iu::ImageGpu_32f_C1* device_f, iu::ImageGpu_32f_C1* device_u,
                            iu::ImageGpu_32f_C1* device_u_, iu::ImageGpu_32f_C2* device_p,
                            float lambda, int max_iter)
{
  int width = device_f->width();
  int height = device_f->height();

  // fragmentation
  int nb_x = width/BSX;
  int nb_y = height/BSY;
  if (nb_x*BSX < width) nb_x++;
  if (nb_y*BSY < height) nb_y++;

  dim3 dimBlock(BSX,BSY);
  dim3 dimGrid(nb_x,nb_y);

  bindTexture(rof_tex_f, device_f);
  bindTexture(rof_tex_u, device_u);
  bindTexture(rof_tex_u_, device_u_);
  bindTexture(rof_tex_p, device_p);

  float L = sqrtf(8.0f);
  float tau = 1/L;
  float sigma = 1/L;
  float theta;

  for (int k = 0; k < max_iter; ++k)
  {
    update_dual_shared<<<dimGrid, dimBlock>>>(device_p->data(), sigma,
                                              width, height, device_p->stride());

    if (sigma < 1000.0f)
      theta = 1/sqrtf(1.0f+0.7f*lambda*tau);
    else
      theta = 1.0f;

    update_primal_shared<<<dimGrid, dimBlock>>>(device_u->data(), device_u_->data(),
                                                tau, theta, lambda, width, height,
                                                device_u->stride());
    sigma /= theta;
    tau *= theta;
  }
  IuStatus status = iu::checkCudaErrorState(true);
  if(status != IU_NO_ERROR)
  {
    std::cerr << "An error occured while solving the ROF model." << std::endl;
  }
}


////////////////////////////////////////////////////////////////////////////////
//############################################################################//
//OOOOOOOOOOOOOOOOOOOOOOOO SINGLE SHARED MEMORY  OOOOOOOOOOOOOOOOOOOOOOOOOOOOO//
//############################################################################//
////////////////////////////////////////////////////////////////////////////////

const unsigned int BSX_SINGLE=16;
const unsigned int BSY_SINGLE=12;

////////////////////////////////////////////////////////////////////////////////
__global__ void update_shared_single(float* device_u, float2* device_p,
                                     float tau, float theta, float sigma, float lambda,
                                     int width, int height, int stride1, int stride2)
{
  int x = blockIdx.x*(blockDim.x-1) + threadIdx.x;
  int y = blockIdx.y*(blockDim.y-1) + threadIdx.y;

  __shared__ float u_[BSX_SINGLE][BSY_SINGLE];

  float2 p = tex2D(rof_tex_p, x+0.5f, y+0.5f);
  float2 pwval = tex2D(rof_tex_p, x-0.5f, y+0.5f);
  float2 pnval = tex2D(rof_tex_p, x+0.5f, y-0.5f);
  if (x == 0)
    pwval.x = 0.0f;
  else if (x >= width-1)
    p.x = 0.0f;
  if (y == 0)
    pnval.y = 0.0f;
  else if (y >= height-1)
    p.y = 0.0f;

  float f = tex2D(rof_tex_f, x+0.5f, y+0.5f);
  float u = tex2D(rof_tex_u, x+0.5f, y+0.5f);

  // Remember old u
  u_[threadIdx.x][threadIdx.y] = u;

  // Primal update
  u = (u + tau*((p.x - pwval.x + p.y - pnval.y) + lambda*f))/(1.0f+tau*lambda);

  // Overrelaxation
  u_[threadIdx.x][threadIdx.y] = u + theta*(u-u_[threadIdx.x][threadIdx.y]);

  __syncthreads();

  if(x<width && y<height)
  {
    if ( (threadIdx.x < BSX_SINGLE-1) && (threadIdx.y < BSY_SINGLE-1) )
    {
      // Dual Update
      float2 grad_u = make_float2(u_[threadIdx.x+1][threadIdx.y] - u_[threadIdx.x][threadIdx.y],
                                  u_[threadIdx.x][threadIdx.y+1] - u_[threadIdx.x][threadIdx.y] );
      p = p + sigma*grad_u;
      p = p/max(1.0f, length(p));

      // Write Back
      device_p[y*stride2 + x] = p;
      device_u[y*stride1 + x] = u;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void rof_primal_dual_shared_single(iu::ImageGpu_32f_C1* device_f, iu::ImageGpu_32f_C1* device_u,
                                   iu::ImageGpu_32f_C2* device_p,
                                   float lambda, int max_iter)
{
  int width = device_f->width();
  int height = device_f->height();

  // fragmentation
  int nb_x = width/(BSX_SINGLE-1);
  int nb_y = height/(BSY_SINGLE-1);
  if (nb_x*(BSX_SINGLE-1) < width) nb_x++;
  if (nb_y*(BSY_SINGLE-1) < height) nb_y++;

  dim3 dimBlock(BSX_SINGLE,BSY_SINGLE);
  dim3 dimGrid(nb_x,nb_y);

  bindTexture(rof_tex_f, device_f);
  bindTexture(rof_tex_u, device_u);
  bindTexture(rof_tex_p, device_p);

  float L = sqrtf(8.0f);
  float tau = 1/L;
  float sigma = 1/L;
  float theta;

  for (int k = 0; k < max_iter; ++k)
  {
    if (sigma < 1000.0f)
      theta = 1/sqrtf(1.0f+0.7f*lambda*tau);
    else
      theta = 1.0f;

    update_shared_single<<<dimGrid, dimBlock>>>(device_u->data(), device_p->data(),
                                                tau, theta, sigma, lambda, width, height,
                                                device_u->stride(), device_p->stride());
    sigma /= theta;
    tau *= theta;
  }
  IuStatus status = iu::checkCudaErrorState(true);
  if(status != IU_NO_ERROR)
  {
    std::cerr << "An error occured while solving the ROF model." << std::endl;
  }
}

////////////////////////////////////////////////////////////////////////////////
//############################################################################//
//OOOOOOOOOOOOOOOOOOOOOO TWO SINGLE SHARED MEMORY  OOOOOOOOOOOOOOOOOOOOOOOOOOO//
//############################################################################//
////////////////////////////////////////////////////////////////////////////////

const unsigned int BSX_SINGLE2=16;
const unsigned int BSY_SINGLE2=10;

////////////////////////////////////////////////////////////////////////////////
__global__ void update_shared_single2(float* device_u, float2* device_p,
                                      float tau, float theta, float sigma, float lambda,
                                      int width, int height, int stride1, int stride2)
{
  int x = blockIdx.x*(blockDim.x-3) + threadIdx.x - 1;
  int y = blockIdx.y*(blockDim.y-3) + threadIdx.y - 1;

  __shared__ float u_[BSX_SINGLE2][BSY_SINGLE2];
  __shared__ float2 p[BSX_SINGLE2][BSY_SINGLE2];

  float2 pc = tex2D(rof_tex_p, x+0.5f, y+0.5f);
  float2 pwval = tex2D(rof_tex_p, x-0.5f, y+0.5f);
  float2 pnval = tex2D(rof_tex_p, x+0.5f, y-0.5f);
  if (x == 0)
    pwval.x = 0.0f;
  else if (x >= width-1)
    pc.x = 0.0f;
  if (y == 0)
    pnval.y = 0.0f;
  else if (y >= height-1)
    pc.y = 0.0f;

  float f = tex2D(rof_tex_f, x+0.5f, y+0.5f);
  float u = tex2D(rof_tex_u, x+0.5f, y+0.5f);

  // Remember old u
  u_[threadIdx.x][threadIdx.y] = u;

  // Primal update
  u = (u + tau*((pc.x - pwval.x + pc.y - pnval.y) + lambda*f))/(1.0f+tau*lambda);

  // Overrelaxation
  u_[threadIdx.x][threadIdx.y] = u + theta*(u-u_[threadIdx.x][threadIdx.y]);

  __syncthreads();

  if ( (threadIdx.x < BSX_SINGLE2-1) && (threadIdx.y < BSY_SINGLE2-1) )
  {
    // Dual Update
    float2 grad_u = make_float2(u_[threadIdx.x+1][threadIdx.y] - u_[threadIdx.x][threadIdx.y],
                                u_[threadIdx.x][threadIdx.y+1] - u_[threadIdx.x][threadIdx.y] );
    pc = pc + sigma*grad_u;
    p[threadIdx.x][threadIdx.y] = pc/max(1.0f, length(pc));
  }

  __syncthreads();

  if (threadIdx.x>0 && threadIdx.y>0)
  {
    // Remember old u
    u_[threadIdx.x][threadIdx.y] = u;

    // Primal update
    float div = p[threadIdx.x][threadIdx.y].x - p[threadIdx.x-1][threadIdx.y].x +
        p[threadIdx.x][threadIdx.y].y - p[threadIdx.x][threadIdx.y-1].y;
    u = (u + tau*(div + lambda*f))/(1.0f+tau*lambda);

    // Overrelaxation
    u_[threadIdx.x][threadIdx.y] = u + theta*(u-u_[threadIdx.x][threadIdx.y]);
  }

  __syncthreads();

  if(x<width && y<height)
  {
    if ( (threadIdx.x < BSX_SINGLE2-2) && (threadIdx.y < BSY_SINGLE2-2) )
    {
      // Dual Update
      float2 grad_u = make_float2(u_[threadIdx.x+1][threadIdx.y] - u_[threadIdx.x][threadIdx.y],
                                  u_[threadIdx.x][threadIdx.y+1] - u_[threadIdx.x][threadIdx.y] );
      pc = p[threadIdx.x][threadIdx.y] + sigma*grad_u;
      p[threadIdx.x][threadIdx.y] = pc/max(1.0f, length(pc));

      if (threadIdx.x>0 && threadIdx.y>0)
      {
        // Write Back
        device_p[y*stride2 + x] = p[threadIdx.x][threadIdx.y];
        device_u[y*stride1 + x] = u;
      }
    }
  }


}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void rof_primal_dual_shared_single2(iu::ImageGpu_32f_C1* device_f, iu::ImageGpu_32f_C1* device_u,
                                    iu::ImageGpu_32f_C2* device_p,
                                    float lambda, int max_iter)
{
  int width = device_f->width();
  int height = device_f->height();

  // fragmentation
  int nb_x = width/(BSX_SINGLE2-3);
  int nb_y = height/(BSY_SINGLE2-3);
  if (nb_x*(BSX_SINGLE2-3) < width) nb_x++;
  if (nb_y*(BSY_SINGLE2-3) < height) nb_y++;

  dim3 dimBlock(BSX_SINGLE2,BSY_SINGLE2);
  dim3 dimGrid(nb_x,nb_y);

  bindTexture(rof_tex_f, device_f);
  bindTexture(rof_tex_u, device_u);
  bindTexture(rof_tex_p, device_p);

  float L = sqrtf(8.0f);
  float tau = 1/L;
  float sigma = 1/L;
  float theta;

  for (int k = 0; k < max_iter; k+=2)
  {
    if (sigma < 1000.0f)
      theta = 1/sqrtf(1.0f+0.7f*lambda*tau);
    else
      theta = 1.0f;

    update_shared_single2<<<dimGrid, dimBlock>>>(device_u->data(), device_p->data(),
                                                 tau, theta, sigma, lambda, width, height,
                                                 device_u->stride(), device_p->stride());
    sigma /= theta;
    tau *= theta;
  }
  IuStatus status = iu::checkCudaErrorState(true);
  if(status != IU_NO_ERROR)
  {
    std::cerr << "An error occured while solving the ROF model." << std::endl;
  }
}

#endif // IUSPARSECOMPARE_CU

