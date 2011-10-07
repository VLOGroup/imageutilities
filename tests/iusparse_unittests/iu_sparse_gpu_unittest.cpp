#include <math.h>

#include <cuda.h>
#include <cusparse.h>

#include <iucore.h>
#include <iusparse.h>

#include "print_helpers.h"

//////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
  try
  {
    // Sparse matrix
    cusparseHandle_t handle = 0;
    cusparseCreate(&handle);

    // INPUT ///////////////////////////////////////////////////////
    cout << "== INPUT ==" << endl;
    // Generate Vector a
    iu::LinearHostMemory<float> vecA(3);
    vecA.data()[0] = 1.0f;
    vecA.data()[1] = 2.0f;
    vecA.data()[2] = 3.0f;

    iu::LinearDeviceMemory<float> vecA_d(vecA.length());
    iu::copy(&vecA, &vecA_d);

    printVector(&vecA_d, "a");

    // Generate Image I
    iu::ImageCpu_32f_C1 I(6, 4);
    iu::setValue(0.0f, &I, I.roi());
    *I.data(0,0) = 1.0f;
    *I.data(1,1) = 1.0f;
    *I.data(2,2) = 1.0f;
    *I.data(3,3) = 1.0f;
    *I.data(2,0) = 0.5f;
    *I.data(3,1) = 0.5f;
    *I.data(4,2) = 0.5f;
    *I.data(5,3) = 0.5f;
    for (int x=0; x<I.width(); x++)
      *I.data(x,2) = (x+1.0f)/10.f;
    iu::ImageGpu_32f_C1 I_d(6,4);
    iu::copy(&I, &I_d);

    printImage(&I_d, "I");

    // Generate Matrix B
    iu::LinearHostMemory<int> rowB(5);
    iu::LinearHostMemory<int> colB(5);
    iu::LinearHostMemory<float> valB(5);

    rowB.data()[0] = 0;
    rowB.data()[1] = 0;
    rowB.data()[2] = 1;
    rowB.data()[3] = 1;
    rowB.data()[4] = 2;

    colB.data()[0] = 0;
    colB.data()[1] = 1;
    colB.data()[2] = 1;
    colB.data()[3] = 2;
    colB.data()[4] = 1;

    valB.data()[0] = 1.0f;
    valB.data()[1] = 3.0f;
    valB.data()[2] = 4.0f;
    valB.data()[3] = 1.0f;
    valB.data()[4] = 2.0f;

    iu::LinearDeviceMemory<int> rowB_d(rowB.length());
    iu::copy(&rowB, &rowB_d);
    iu::LinearDeviceMemory<int> colB_d(colB.length());
    iu::copy(&colB, &colB_d);
    iu::LinearDeviceMemory<float> valB_d(valB.length());
    iu::copy(&valB, &valB_d);

    iu::SparseMatrixGpu_32f B_d(&handle, &valB_d, &rowB_d, 3, &colB_d, 3);

    printSparseMatrixElements(&B_d, "B");

    // Generate Matrix C
    int NC = 10;
    iu::LinearHostMemory<int> rowC(NC);
    iu::LinearHostMemory<int> colC(NC);
    iu::LinearHostMemory<float> valC(NC);

    //  colC.data()[0] = 0;
    //  colC.data()[1] = 0;
    //  colC.data()[2] = 1;
    //  colC.data()[3] = 1;
    //  colC.data()[4] = 2;
    //  colC.data()[5] = 3;
    //  colC.data()[6] = 3;
    //  colC.data()[7] = 2;
    //  colC.data()[8] = 9;

    //  rowC.data()[0] = 0;
    //  rowC.data()[1] = 1;
    //  rowC.data()[2] = 1;
    //  rowC.data()[3] = 2;
    //  rowC.data()[4] = 1;
    //  rowC.data()[5] = 0;
    //  rowC.data()[6] = 1;
    //  rowC.data()[7] = 7;
    //  rowC.data()[8] = 19;

    //  valC.data()[0] = 1.0f;
    //  valC.data()[1] = 3.0f;
    //  valC.data()[2] = 4.0f;
    //  valC.data()[3] = 6.0f;
    //  valC.data()[4] = 5.0f;
    //  valC.data()[5] = 2.0f;
    //  valC.data()[6] = 7.0f;
    //  valC.data()[7] = 6.0f;
    //  valC.data()[8] = 9.0f;

    colC.data()[0] = 0;
    colC.data()[1] = 3;
    colC.data()[2] = 0;
    colC.data()[3] = 1;
    colC.data()[4] = 2;
    colC.data()[5] = 3;
    colC.data()[6] = 1;
    colC.data()[7] = 2;
    colC.data()[8] = 5;
    colC.data()[9] = 8;

    rowC.data()[0] = 0;
    rowC.data()[1] = 0;
    rowC.data()[2] = 1;
    rowC.data()[3] = 1;
    rowC.data()[4] = 1;
    rowC.data()[5] = 1;
    rowC.data()[6] = 2;
    rowC.data()[7] = 7;
    rowC.data()[8] = 6;
    rowC.data()[9] = 18;

    valC.data()[0] = 1.0f;
    valC.data()[1] = 2.0f;
    valC.data()[2] = 3.0f;
    valC.data()[3] = 4.0f;
    valC.data()[4] = 5.0f;
    valC.data()[5] = 7.0f;
    valC.data()[6] = 6.0f;
    valC.data()[7] = 6.0f;
    valC.data()[8] = 9.0f;
    valC.data()[9] = 8.0f;

    iu::LinearDeviceMemory<int> rowC_d(rowC.length());
    iu::copy(&rowC, &rowC_d);
    iu::LinearDeviceMemory<int> colC_d(colC.length());
    iu::copy(&colC, &colC_d);
    iu::LinearDeviceMemory<float> valC_d(valC.length());
    iu::copy(&valC, &valC_d);

    iu::SparseMatrixGpu_32f C_d(&handle, &valC_d, &rowC_d, 20, &colC_d, 10);

    printSparseMatrixElements(&C_d, "C");
    C_d.changeSparseFormat(CSC);
    printSparseMatrixElements(&C_d, "C CSC");
    C_d.changeSparseFormat(CSR);
    printSparseMatrixElements(&C_d, "C CSR");

    // Generate Matrix G
    iu::LinearHostMemory<int> rowG(I_d.stride()*I_d.height()*4);
    iu::LinearHostMemory<int> colG(I_d.stride()*I_d.height()*4);
    iu::LinearHostMemory<float> valG(I_d.stride()*I_d.height()*4);

    int cind = 0;
    // Gradient in x direction
    for (int y=0; y<I_d.height(); y++)
    {
      for (int x=0; x<I_d.width()-1; x++)
      {
        *rowG.data(cind) = y*I_d.stride() + x;
        *colG.data(cind) = y*I_d.stride() + x;
        *valG.data(cind) = -1.0f;
        cind++;

        *rowG.data(cind) = y*I_d.stride() + x;
        *colG.data(cind) = y*I_d.stride() + x+1;
        *valG.data(cind) = 1.0f;
        cind++;
      }
    }
    int ofs = I_d.stride()*I_d.height();
    // Gradient in y direction
    for (int y=0; y<I_d.height()-1; y++)
    {
      for (int x=0; x<I_d.width(); x++)
      {
        *rowG.data(cind) = ofs + y*I_d.stride() + x;
        *colG.data(cind) = y*I_d.stride() + x;
        *valG.data(cind) = -1.0f;
        cind++;

        *rowG.data(cind) = ofs +     y*I_d.stride() + x;
        *colG.data(cind) = (y+1)*I_d.stride() + x;
        *valG.data(cind) = 1.0f;
        cind++;
      }
    }
    iu::LinearDeviceMemory<int> rowG_d(cind);
    iu::copy(&rowG, &rowG_d);
    iu::LinearDeviceMemory<int> colG_d(cind);
    iu::copy(&colG, &colG_d);
    iu::LinearDeviceMemory<float> valG_d(cind);
    iu::copy(&valG, &valG_d);
    iu::SparseMatrixGpu_32f G_d(&handle, &valG_d, &rowG_d, I_d.stride()*I_d.height()*2, &colG_d, I_d.stride()*I_d.height());

    cout << "Matrix G is a simple gradient operator that maps to float" << endl << endl;
    //  printSparseMatrixElements(&G_d, "G");


    // OUTPUT ///////////////////////////////////////////////////////
    // Generate Vector c (by sparse matrix multiplication)
    iu::LinearDeviceMemory<float> vecC_d(vecA.length());
    iu::LinearDeviceMemory<float> vecD_d(vecA.length());
    iu::ImageGpu_32f_C1 O_d(I_d.width(), I_d.height()*2);
    iu::ImageGpu_32f_C2 O2_d(I_d.width(), I_d.height());

    // Generate Matrix G2
    iu::LinearHostMemory<int> rowG2(I_d.stride()*I_d.height()*4);
    iu::LinearHostMemory<int> colG2(I_d.stride()*I_d.height()*4);
    iu::LinearHostMemory<float> valG2(I_d.stride()*I_d.height()*4);

    cind = 0;
    for (int y=0; y<I_d.height(); y++)
    {
      for (int x=0; x<I_d.width(); x++)
      {
        if (x<I_d.width()-1)
        {
          *rowG2.data(cind) = y*O2_d.stride()*2 + x*2;
          *colG2.data(cind) = y*I_d.stride() + x;
          *valG2.data(cind) = -1.0f;
          cind++;

          *rowG2.data(cind) = y*O2_d.stride()*2 + x*2;
          *colG2.data(cind) = y*I_d.stride() + x+1;
          *valG2.data(cind) = 1.0f;
          cind++;
        }
        if (y<I_d.height()-1)
        {
          *rowG2.data(cind) = y*O2_d.stride()*2 + x*2+1;
          *colG2.data(cind) = y*I_d.stride() + x;
          *valG2.data(cind) = -1.0f;
          cind++;

          *rowG2.data(cind) = y*O2_d.stride()*2 + x*2+1;
          *colG2.data(cind) = (y+1)*I_d.stride() + x;
          *valG2.data(cind) = 1.0f;
          cind++;
        }
      }
    }
    iu::LinearDeviceMemory<int> rowG2_d(cind);
    iu::copy(&rowG2, &rowG2_d);
    iu::LinearDeviceMemory<int> colG2_d(cind);
    iu::copy(&colG2, &colG2_d);
    iu::LinearDeviceMemory<float> valG2_d(cind);
    iu::copy(&valG2, &valG2_d);
    iu::SparseMatrixGpu_32f G2_d(&handle, &valG2_d, &rowG2_d, O2_d.stride()*O2_d.height()*2, &colG2_d, I_d.stride()*I_d.height());

    cout << "Matrix G2 is a simple gradient operator that maps to float2" << endl << endl;
    //  printSparseMatrixElements(&G2_d, "G2");

    // CONVERSIONS /////////////////////////////////////////////////
    cout << "== Conversions == " << endl;
    //  B_d.changeSparseFormat(CSC);
    //  printSparseMatrixElements(&B_d, "B_CSC");
    //  B_d.changeSparseFormat(CSR);
    //  printSparseMatrixElements(&B_d, "B_CSR");

    //    printSparseMatrixElements(&G_d, "G_CSR");
    //  G_d.changeSparseFormat(CSC);
    //  printSparseMatrixElements(&G_d, "G_CSC");

    // CALCULATIONS /////////////////////////////////////////////////
    cout << "== Sparse Multiplications == " << endl;
    iu::sparseMultiplication(&handle, &B_d, &vecA_d, &vecC_d);
    printVector(&vecC_d, "c=B*a");
    iu::sparseMultiplication(&handle, &B_d, &vecA_d, &vecD_d, CUSPARSE_OPERATION_TRANSPOSE);
    printVector(&vecD_d, "d=B'*a");
    iu::sparseMultiplication(&handle, &G_d, &I_d, &O_d);
    printImage(&O_d, "O=I*G");
    iu::sparseMultiplication(&handle, &G2_d, &I_d, &O2_d);
    printImage(&O2_d, "O2=I*G2");

    // SUM TESTS /////////////////////////////////////////////////
    cout << "== Sum up along axis == " << endl;
    iu::LinearDeviceMemory_32f_C1 e1(20);
    iu::LinearDeviceMemory_32f_C1 e2(10);
    iu::sumSparseRow(&C_d, &e1);
    printVector(&e1, "e1=sumRow(C)");
    iu::sumSparseCol(&C_d, &e2);
    printVector(&e2, "e2=sumCol(C)");

    // Clean up
    cusparseDestroy(handle);

  }
  catch (IuException& e)
  {
    std::cerr << e.what() << std::endl;
  }


  return 0;
}
