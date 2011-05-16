#include <math.h>
#include <iostream>

#include <cuda.h>
#include <cusparse.h>

#include <iucore.h>
#include <iusparse.h>
#include <iuio.h>
#include <iutransform.h>

#include "print_helpers.h"


#define NSIZES 2
#define NALG   6

//cusparseHandle_t handle = 0;


// Forward declarations
void rof_primal_dual(iu::ImageGpu_32f_C1* device_f, iu::ImageGpu_32f_C1* device_u,
                     iu::ImageGpu_32f_C1* device_u_, iu::ImageGpu_32f_C2* device_p,
                     float lambda, int max_iter);

void rof_primal_dual_sparse(iu::SparseMatrixGpu_32f* G,
                            iu::ImageGpu_32f_C1* device_f, iu::ImageGpu_32f_C1* device_u,
                            iu::ImageGpu_32f_C1* device_u_, iu::ImageGpu_32f_C2* device_p,
                            float lambda, int max_iter);

void rof_primal_dual_notex(iu::ImageGpu_32f_C1* device_f, iu::ImageGpu_32f_C1* device_u,
                           iu::ImageGpu_32f_C1* device_u_, iu::ImageGpu_32f_C2* device_p,
                           float lambda, int max_iter);

void rof_primal_dual_shared(iu::ImageGpu_32f_C1* device_f, iu::ImageGpu_32f_C1* device_u,
                            iu::ImageGpu_32f_C1* device_u_, iu::ImageGpu_32f_C2* device_p,
                            float lambda, int max_iter);

void rof_primal_dual_shared_single(iu::ImageGpu_32f_C1* device_f, iu::ImageGpu_32f_C1* device_u,
                            iu::ImageGpu_32f_C2* device_p, float lambda, int max_iter);

void rof_primal_dual_shared_single2(iu::ImageGpu_32f_C1* device_f, iu::ImageGpu_32f_C1* device_u,
                                    iu::ImageGpu_32f_C2* device_p, float lambda, int max_iter);


// Calls
void calcROF(iu::ImageGpu_32f_C1* f, iu::ImageGpu_32f_C1* u,
             float lambda, int max_iter, double* init, double* alg, double* complete,
             bool notex=false)
{
  cudaThreadSynchronize();
  double start = iu::getTime();

  iu::ImageGpu_32f_C1 u_(u->size());
  iu::copy(u, &u_);
  iu::ImageGpu_32f_C2 p(u->size());
  iu::setValue(make_float2(0.0f, 0.0f), &p, p.roi());

  cudaThreadSynchronize();
  double interm = iu::getTime();
  *init = interm - start;

  if (notex)
    rof_primal_dual_notex(f, u, &u_, &p, lambda, max_iter);
  else
    rof_primal_dual(f, u, &u_, &p, lambda, max_iter);

  cudaThreadSynchronize();
  *alg = iu::getTime() - interm;
  *complete = iu::getTime() - start;

  return;
}

void calcROFshared(iu::ImageGpu_32f_C1* f, iu::ImageGpu_32f_C1* u,
                   float lambda, int max_iter, double* init, double* alg, double* complete,
                   bool single=false, int internal=2)
{
  cudaThreadSynchronize();
  double start = iu::getTime();

  iu::ImageGpu_32f_C1 u_(u->size());
  iu::copy(u, &u_);
  iu::ImageGpu_32f_C2 p(u->size());
  iu::setValue(make_float2(0.0f, 0.0f), &p, p.roi());

  cudaThreadSynchronize();
  double interm = iu::getTime();
  *init = interm - start;

  if (single)
  {
    if (internal == 1)
      rof_primal_dual_shared_single(f, u, &p, lambda, max_iter);
    else if (internal == 2)
      rof_primal_dual_shared_single2(f, u, &p, lambda, max_iter);
  }
  else
    rof_primal_dual_shared(f, u, &u_, &p, lambda, max_iter);

  cudaThreadSynchronize();
  *alg = iu::getTime() - interm;
  *complete = iu::getTime() - start;

  return;
}


void calcROFSparse(iu::ImageGpu_32f_C1* f, iu::ImageGpu_32f_C1* u,
                   float lambda, int max_iter, double* init, double* alg, double* complete)
{
  // Sparse matrix
  cusparseHandle_t handle = 0;
  cusparseCreate(&handle);

  cudaThreadSynchronize();
  double start = iu::getTime();

  iu::ImageGpu_32f_C1 u_(u->size());
  iu::copy(u, &u_);
  iu::ImageGpu_32f_C2 p(u->size());
  iu::setValue(make_float2(0.0f, 0.0f), &p, p.roi());

  // Generate Matrix G
  iu::LinearHostMemory<int> rowG(u->stride()*u->height()*4);
  iu::LinearHostMemory<int> colG(u->stride()*u->height()*4);
  iu::LinearHostMemory<float> valG(u->stride()*u->height()*4);

  int cind = 0;
  for (int y=0; y<u->height(); y++)
  {
    for (int x=0; x<u->width(); x++)
    {
      if (x<u->width()-1)
      {
        *rowG.data(cind) = y*p.stride()*2 + x*2;
        *colG.data(cind) = y*u->stride() + x;
        *valG.data(cind) = -1.0f;
        cind++;

        *rowG.data(cind) = y*p.stride()*2 + x*2;
        *colG.data(cind) = y*u->stride() + x+1;
        *valG.data(cind) = 1.0f;
        cind++;
      }
      if (y<u->height()-1)
      {
        *rowG.data(cind) = y*p.stride()*2 + x*2+1;
        *colG.data(cind) = y*u->stride() + x;
        *valG.data(cind) = -1.0f;
        cind++;

        *rowG.data(cind) = y*p.stride()*2 + x*2+1;
        *colG.data(cind) = (y+1)*u->stride() + x;
        *valG.data(cind) = 1.0f;
        cind++;
      }
    }
  }
  iu::LinearDeviceMemory<int> rowG_d(cind);
  iu::copy(&rowG, &rowG_d);
  iu::LinearDeviceMemory<int> colG_d(cind);
  iu::copy(&colG, &colG_d);
  iu::LinearDeviceMemory<float> valG_d(cind);
  iu::copy(&valG, &valG_d);
  iu::SparseMatrixGpu_32f G_d(&handle, &valG_d, &rowG_d, p.stride()*2*p.height(), &colG_d, u->stride()*u->height());

  G_d.changeSparseFormat(CSR);

  cudaThreadSynchronize();
  double interm = iu::getTime();
  *init = interm - start;

  rof_primal_dual_sparse(&G_d, f, u, &u_, &p, lambda, max_iter);

  cudaThreadSynchronize();
  *alg = iu::getTime() - interm;
  *complete = iu::getTime() - start;

  cusparseDestroy(handle);

  return;
}


//////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
  //  // Sparse matrix
  //  cusparseCreate(&handle);

  // full
  iu::ImageGpu_32f_C1* input_full = iu::imread_cu32f_C1("../../Data/test/cat.pgm");

  iu::ImageGpu_32f_C1 input_256(256, 256*input_full->height()/input_full->width());
  iu::reduce(input_full, &input_256);

  iu::ImageGpu_32f_C1 output_256(input_256.size());
  iu::ImageGpu_32f_C1 output_256_notex(input_256.size());
  iu::ImageGpu_32f_C1 output_256_shared(input_256.size());
  iu::ImageGpu_32f_C1 output_256_shared_single(input_256.size());
  iu::ImageGpu_32f_C1 output_256_shared_single2(input_256.size());
  iu::ImageGpu_32f_C1 output_256_sparse(input_256.size());
  iu::copy(&input_256, &output_256);
  iu::copy(&input_256, &output_256_notex);
  iu::copy(&input_256, &output_256_shared);
  iu::copy(&input_256, &output_256_shared_single);
  iu::copy(&input_256, &output_256_shared_single2);
  iu::copy(&input_256, &output_256_sparse);

  iu::ImageGpu_32f_C1 input_1024(1024, 1024*input_full->height()/input_full->width());
  iu::reduce(input_full, &input_1024);

  iu::ImageGpu_32f_C1 output_1024(input_1024.size());
  iu::ImageGpu_32f_C1 output_1024_notex(input_1024.size());
  iu::ImageGpu_32f_C1 output_1024_shared(input_1024.size());
  iu::ImageGpu_32f_C1 output_1024_shared_single(input_1024.size());
  iu::ImageGpu_32f_C1 output_1024_shared_single2(input_1024.size());
  iu::ImageGpu_32f_C1 output_1024_sparse(input_1024.size());
  iu::copy(&input_1024, &output_1024);
  iu::copy(&input_1024, &output_1024_notex);
  iu::copy(&input_1024, &output_1024_shared);
  iu::copy(&input_1024, &output_1024_shared_single);
  iu::copy(&input_1024, &output_1024_shared_single2);
  iu::copy(&input_1024, &output_1024_sparse);

  float lambda    = 1.0f;
  int   max_iter  = 1000;

  double init[NSIZES][NALG];
  double alg[NSIZES][NALG];
  double complete[NSIZES][NALG];
  for (int a=0; a<NALG; a++)
  {
    for(int sz=0; sz<NSIZES; sz++)
    {
      init[sz][a] = -1.0;
      alg[sz][a] = -1.0;
      complete[sz][a] = -1.0;
    }
  }
  char* name[] = {"Standard: ", "No Tex:   ", "Shared:   ", "Single:   ", "Single2:  ", "Sparse:   "};

  int calg = 0;
  // Co calculations of standard ROF model
  calcROF(&input_256, &output_256,  lambda, max_iter, &init[0][calg], &alg[0][calg], &complete[0][calg]);
  calcROF(&input_1024, &output_1024, lambda, max_iter, &init[1][calg], &alg[1][calg], &complete[1][calg]);
  calg++;

  // Co calculations of standard ROF with no texture model
  calcROF(&input_256, &output_256_notex,  lambda, max_iter, &init[0][calg], &alg[0][calg], &complete[0][calg], true);
  calcROF(&input_1024, &output_1024_notex, lambda, max_iter, &init[1][calg], &alg[1][calg], &complete[1][calg], true);
  calg++;

  // Co calculations of standard ROF with shared memory
  calcROFshared(&input_256, &output_256_shared,  lambda, max_iter, &init[0][calg], &alg[0][calg], &complete[0][calg]);
  calcROFshared(&input_1024, &output_1024_shared, lambda, max_iter, &init[1][calg], &alg[1][calg], &complete[1][calg]);
  calg++;

  // Co calculations of standard ROF with shared memory in single kernel
  calcROFshared(&input_256, &output_256_shared_single,  lambda, max_iter, &init[0][calg], &alg[0][calg], &complete[0][calg], true, 1);
  calcROFshared(&input_1024, &output_1024_shared_single, lambda, max_iter, &init[1][calg], &alg[1][calg], &complete[1][calg], true, 1);
  calg++;

  // Co calculations of standard ROF with shared memory in single kernel with two iterations
  calcROFshared(&input_256, &output_256_shared_single2,  lambda, max_iter, &init[0][calg], &alg[0][calg], &complete[0][calg], true, 2);
  calcROFshared(&input_1024, &output_1024_shared_single2, lambda, max_iter, &init[1][calg], &alg[1][calg], &complete[1][calg], true, 2);
  calg++;

  //  // Co calculations of sparse ROF model
  //  calcROFSparse(&input_256, &output_256_sparse,  lambda, max_iter, &init[0][calg], &alg[0][calg], &complete[0][calg]);
  //  calcROFSparse(&input_1024, &output_1024_sparse, lambda, max_iter, &init[1][calg], &alg[1][calg], &complete[1][calg]);
  //  calg++;

  std::cout << "          256              1024" << std::endl;
  for (int a=0; a<NALG; a++)
  {
    std::cout << name[a];
    for(int sz=0; sz<NSIZES; sz++)
    {
      std::cout << complete[sz][a] << "(" << alg[sz][a] << ")  ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  iu::imsave(&input_256, "256_input.png");
  iu::imsave(&output_256, "256_output.png");
  iu::imsave(&output_256_notex, "256_notex_output.png");
  iu::imsave(&output_256_shared, "256_shared_output.png");
  iu::imsave(&output_256_shared_single, "256_shared_single_output.png");
  iu::imsave(&output_256_shared_single2, "256_shared_single2_output.png");
  iu::imsave(&output_256_sparse, "256_sparse_output.png");

  iu::imsave(&input_1024, "1024_input.png");
  iu::imsave(&output_1024, "1024_output.png");
  iu::imsave(&output_1024_notex, "1024_notex_output.png");
  iu::imsave(&output_1024_shared, "1024_shared_output.png");
  iu::imsave(&output_1024_shared_single, "1024_shared_single_output.png");
  iu::imsave(&output_1024_shared_single2, "1024_shared_single2_output.png");
  iu::imsave(&output_1024_sparse, "1024_sparse_output.png");

  // Clean up
  //  cusparseDestroy(handle);
  delete input_full;

  return 0;
}
