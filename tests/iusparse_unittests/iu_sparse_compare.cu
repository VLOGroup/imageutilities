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

const unsigned int BSX=16;
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
//! Primal energy kernel
////////////////////////////////////////////////////////////////////////////////
__global__ void primal_energy_kernel(float* primal, float lambda,
                                     int width, int height, int xstride)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  float xx = x + 0.5f;
  float yy = y + 0.5f;

  // texture fetches
  float f = tex2D(rof_tex_f, xx, yy);
  float u = tex2D(rof_tex_u, xx, yy);
  float2 grad_u = dp(rof_tex_u, x, y);

  // Compute pixel wise energy
  if ((x<width) && (y<height))
  {
    primal[y*xstride + x] = length(grad_u) + lambda/2.0f * (u-f)*(u-f);
  }
}

////////////////////////////////////////////////////////////////////////////////
//! Dual energy kernel for the 2D TVSEG model
////////////////////////////////////////////////////////////////////////////////
__global__ void dual_energy_kernel(float* dual, float lambda,
                                   int width, int height, int xstride)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  float xx = x + 0.5f;
  float yy = y + 0.5f;

  // texture fetches
  float f = tex2D(rof_tex_f, xx, yy);
  float divergence = dp_ad(rof_tex_p, x, y, width, height);

  // Compute pixel wise energy
  if ((x<width) && (y<height))
  {
    dual[y*xstride + x] = -divergence*divergence/(2.0f*lambda) - divergence*f;
  }
}

////////////////////////////////////////////////////////////////////////////////
void rof_energy( iu::ImageGpu_32f_C1* device_f, iu::ImageGpu_32f_C1* device_u,
                iu::ImageGpu_32f_C2* device_p,
                float lambda, double& primal_energy, double& dual_energy)
{
  int width = device_f->width();
  int height = device_f->height();

  // compute number of Blocks
  int nb_x = width/BSX;
  int nb_y = height/BSY;
  if (nb_x*BSX < width) nb_x++;
  if (nb_y*BSY < height) nb_y++;

  dim3 dimBlock(BSX,BSY);
  dim3 dimGrid(nb_x,nb_y);

  bindTexture(rof_tex_f, device_f);
  bindTexture(rof_tex_u, device_u);
  bindTexture(rof_tex_p, device_p);

  // Temporary variable for energies
  iu::ImageGpu_32f_C1 temp_energy(device_f->size());

  // Calculate primal energy
  primal_energy_kernel
      <<<dimGrid, dimBlock>>>(temp_energy.data(), lambda,
                              width, height, temp_energy.stride());
  iu::summation(&temp_energy, device_f->roi(), primal_energy);

  // Calculate dual energy
  dual_energy_kernel
      <<<dimGrid, dimBlock>>>(temp_energy.data(), lambda,
                              width, height, temp_energy.stride());
  iu::summation(&temp_energy, device_f->roi(), dual_energy);
  iu::checkCudaErrorState();
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
    float divergence;

    divergence = dp_ad(rof_tex_p, x, y, width, height);

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

  float L = sqrt(8.0f);
  float tau = 1/L;
  float sigma = 1/L;
  float theta;

  for (int k = 0; k < max_iter; ++k)
  {
    update_dual<<<dimGrid, dimBlock>>>(device_p->data(), sigma,
                                       width, height, device_p->stride());

    if (sigma < 1000.0f)
      theta = 1/sqrt(1.0f+0.7f*lambda*tau);
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

  float L = sqrt(8.0f);
  float tau = 1/L;
  float sigma = 1/L;
  float theta;

  for (int k = 0; k < max_iter; ++k)
  {
    update_dual_notex<<<dimGrid, dimBlock>>>(device_p->data(), device_u_->data(),
                                             sigma, width, height,
                                             device_u_->stride(), device_p->stride());

    if (sigma < 1000.0f)
      theta = 1/sqrt(1.0f+0.7f*lambda*tau);
    else
      theta = 1.0f;

    update_primal_notex<<<dimGrid, dimBlock>>>(device_u->data(), device_u_->data(),
                                               device_f->data(), device_p->data(),
                                               tau, theta, lambda, width, height,
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

  float L = sqrt(8.0f);
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
      theta = 1/sqrt(1.0f+0.7f*lambda*tau);
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

  float L = sqrt(8.0f);
  float tau = 1/L;
  float sigma = 1/L;
  float theta;

  for (int k = 0; k < max_iter; ++k)
  {
    update_dual_shared<<<dimGrid, dimBlock>>>(device_p->data(), sigma,
                                              width, height, device_p->stride());

    if (sigma < 1000.0f)
      theta = 1/sqrt(1.0f+0.7f*lambda*tau);
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

#endif // IUSPARSECOMPARE_CU

