#include <math.h>
#include <iostream>

#include <cuda.h>
#include <cusparse.h>

#include <iucore.h>
#include <iusparse.h>
#include <iuio.h>
#include <iutransform.h>

#include "print_helpers.h"


#define NSIZES 3
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
  for (unsigned int y=0; y<u->height(); y++)
  {
    for (unsigned int x=0; x<u->width(); x++)
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

  char* name[] = {"Standard: ", "No Tex:   ", "Shared:   ", "Single:   ", "Single2:  ", "Sparse:   "};
  int sizes[] = {256, 1024, 2048};

  float lambda    = 5.0f;
  int   max_iter  = 100;

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

  // Read input
  iu::ImageGpu_32f_C1* input_full = iu::imread_cu32f_C1("../../Data/test/cat.pgm");
//  iu::ImageGpu_32f_C1* input_full = iu::imread_cu32f_C1("/home/markus/Downloads/lena_bw_2048.pgm");


  for (int s=0; s<NSIZES; s++)
  {
    // Downsample and create images
    iu::ImageGpu_32f_C1 input_resized(sizes[s], sizes[s]*input_full->height()/input_full->width());
    iu::reduce(input_full, &input_resized);

    iu::ImageGpu_32f_C1 output(input_resized.size());
    iu::ImageGpu_32f_C1 output_notex(input_resized.size());
    iu::ImageGpu_32f_C1 output_shared(input_resized.size());
    iu::ImageGpu_32f_C1 output_shared_single(input_resized.size());
    iu::ImageGpu_32f_C1 output_shared_single2(input_resized.size());
    iu::ImageGpu_32f_C1 output_sparse(input_resized.size());
    iu::copy(&input_resized, &output);
    iu::copy(&input_resized, &output_notex);
    iu::copy(&input_resized, &output_shared);
    iu::copy(&input_resized, &output_shared_single);
    iu::copy(&input_resized, &output_shared_single2);
    iu::copy(&input_resized, &output_sparse);

    int calg = 0;
    // Co calculations of standard ROF model
    calcROF(&input_resized, &output,  lambda, max_iter, &init[s][calg], &alg[s][calg], &complete[s][calg]);
    calg++;

    // Co calculations of standard ROF with no texture model
    calcROF(&input_resized, &output_notex,  lambda, max_iter, &init[s][calg], &alg[s][calg], &complete[s][calg], true);
    calg++;

    // Co calculations of standard ROF with shared memory
    calcROFshared(&input_resized, &output_shared,  lambda, max_iter, &init[s][calg], &alg[s][calg], &complete[s][calg]);
    calg++;

    // Co calculations of standard ROF with shared memory in single kernel
    calcROFshared(&input_resized, &output_shared_single,  lambda, max_iter, &init[s][calg], &alg[s][calg], &complete[s][calg], true, 1);
    calg++;

    // Co calculations of standard ROF with shared memory in single kernel with two iterations
    calcROFshared(&input_resized, &output_shared_single2,  lambda, max_iter, &init[s][calg], &alg[s][calg], &complete[s][calg], true, 2);
    calg++;

  //    // Co calculations of sparse ROF model
  //    calcROFSparse(&input_resized, &output_sparse,  lambda, max_iter, &init[s][calg], &alg[s][calg], &complete[s][calg]);
  //    calg++;

    // Save Output
    char buffer [50];
    sprintf(buffer, "%d_input.png", sizes[s]);
    iu::imsave(&input_resized, buffer);
    sprintf(buffer, "%d_std_output.png", sizes[s]);
    iu::imsave(&output, buffer);
    sprintf(buffer, "%d_notex_output.png", sizes[s]);
    iu::imsave(&output_notex, buffer);
    sprintf(buffer, "%d_shared_output.png", sizes[s]);
    iu::imsave(&output_shared, buffer);
    sprintf(buffer, "%d_shared_single_output.png", sizes[s]);
    iu::imsave(&output_shared_single, buffer);
    sprintf(buffer, "%d_shared_single2_output.png", sizes[s]);
    iu::imsave(&output_shared_single2, buffer);
//    sprintf(buffer, "%d_sparse_output.png", sizes[s]);
//    iu::imsave(&output_sparse, buffer);
}

  std::cout << "\t\t";
  for(int sz=0; sz<NSIZES; sz++)
  {
    char buffer [50];
    sprintf(buffer, "%d", sizes[sz]);
    std::cout.width(8);
    std::cout << buffer << "\t\t";
  }
  std::cout << std::endl;
  for (int a=0; a<NALG; a++)
  {
    std::cout << name[a]<< "\t";
    for(int sz=0; sz<NSIZES; sz++)
    {
      std::cout.width(7);
      std::cout << complete[sz][a] << " (";
      std::cout.width(7);
      std::cout << alg[sz][a] << ") \t";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;


  // Clean up
  //  cusparseDestroy(handle);
  delete input_full;

  return 0;
}
