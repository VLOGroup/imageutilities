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

const unsigned int ROF_BLOCK_SIZE=16;

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
  int nb_x = width/ROF_BLOCK_SIZE;
  int nb_y = height/ROF_BLOCK_SIZE;
  if (nb_x*ROF_BLOCK_SIZE < width) nb_x++;
  if (nb_y*ROF_BLOCK_SIZE < height) nb_y++;

  dim3 dimBlock(ROF_BLOCK_SIZE,ROF_BLOCK_SIZE);
  dim3 dimGrid(nb_x,nb_y);

  bindTexture(rof_tex_f, device_f);
  bindTexture(rof_tex_u, device_u);
  bindTexture(rof_tex_p, device_p);

  // Temporary variable for energies
  iu::ImageGpu_32f_C1 temp_energy(device_f->size());

  // Calculate primal energy
  primal_energy_kernel
      <<< dimGrid, dimBlock >>> (temp_energy.data(), lambda,
                                 width, height, temp_energy.stride());
  iu::summation(&temp_energy, device_f->roi(), primal_energy);

  // Calculate dual energy
  dual_energy_kernel
      <<< dimGrid, dimBlock >>> (temp_energy.data(), lambda,
                                 width, height, temp_energy.stride());
  iu::summation(&temp_energy, device_f->roi(), dual_energy);
  iu::checkCudaErrorState();
}


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
  unsigned int block_size = ROF_BLOCK_SIZE;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(width, dimBlock.x),
               iu::divUp(height, dimBlock.y));

  bindTexture(rof_tex_f, device_f);
  bindTexture(rof_tex_u, device_u);
  bindTexture(rof_tex_u_, device_u_);
  bindTexture(rof_tex_p, device_p);

  float L = sqrt(8.0f);
  float tau = 1/L;
  float sigma = 1/L;
  float theta;

  for (int k = 0; k <= max_iter; ++k)
  {
    update_dual <<< dimGrid, dimBlock >>> (device_p->data(), sigma,
                                           width, height, device_p->stride());

    if (sigma < 1000.0f)
      theta = 1/sqrt(1.0f+0.7f*lambda*tau);
    else
      theta = 1.0f;

    update_primal <<< dimGrid, dimBlock >>> (device_u->data(), device_u_->data(),
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
    u = (u + tau*(tex2D(rof_tex_divergence, x+0.5f, y+0.5f) + lambda*tex2D(rof_tex_f, x+0.5f, y+0.5f)))/(1.0f+tau*lambda);

    device_u[c] = u;
    device_u_[c] = u + theta*(u-u_);
  }
}

////////////////////////////////////////////////////////////////////////////////
__global__ void update_dual_sparse(float2* device_p, float sigma,
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
  unsigned int block_size = ROF_BLOCK_SIZE;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(width, dimBlock.x),
               iu::divUp(height, dimBlock.y));

  iu::ImageGpu_32f_C2 gradient(width, height);
  iu::ImageGpu_32f_C1 divergence(width, height);

  bindTexture(rof_tex_f, device_f);
  bindTexture(rof_tex_u, device_u);
  //  bindTexture(rof_tex_u_, device_u_);
  bindTexture(rof_tex_p, device_p);

  bindTexture(rof_tex_gradient, &gradient);
  bindTexture(rof_tex_divergence, &divergence);

  float L = sqrt(8.0f);
  float tau = 1/L;
  float sigma = 1/L;
  float theta;

  for (int k = 0; k <= max_iter; ++k)
  {
    // Gradient
    iu::sparseMultiplication(G->handle(), G, device_u_, &gradient);
    cudaThreadSynchronize();

    // Dual update
    bindTexture(rof_tex_p, device_p);
    bindTexture(rof_tex_gradient, &gradient);
    cudaThreadSynchronize();
    update_dual_sparse<<< dimGrid, dimBlock >>>(device_p->data(), sigma,
                                                width, height, device_p->stride());
    cudaUnbindTexture(&rof_tex_p);
    cudaUnbindTexture(&rof_tex_gradient);
    cudaThreadSynchronize();

    // Update Timesteps
    if (sigma < 1000.0f)
      theta = 1/sqrt(1.0f+0.7f*lambda*tau);
    else
      theta = 1.0f;

    // Divergence
    iu::sparseMultiplication(G->handle(), G, device_p, &divergence, CUSPARSE_OPERATION_TRANSPOSE);
    cudaThreadSynchronize();

    // Primal update
    bindTexture(rof_tex_f, device_f);
    bindTexture(rof_tex_u, device_u);
    bindTexture(rof_tex_divergence, &divergence);
    cudaThreadSynchronize();
    update_primal_sparse<<< dimGrid, dimBlock >>>(device_u->data(), device_u_->data(),
                                                  tau, theta, lambda, width, height,
                                                  device_u->stride());
    cudaUnbindTexture(&rof_tex_f);
    cudaUnbindTexture(&rof_tex_u);
    cudaUnbindTexture(&rof_tex_divergence);
    cudaThreadSynchronize();

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

#endif // IUSPARSECOMPARE_CU

