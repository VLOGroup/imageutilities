// system includes
#include <iostream>
#include <iomanip>
#include <stdlib.h>

using namespace std;

void printVector(iu::LinearDeviceMemory<float>* vec_d, string title)
{
  iu::LinearHostMemory<float> vec(vec_d->length());
  iu::copy(vec_d, &vec);

  cout << "Vector " << title << ": (length=" << vec_d->length() << ")" <<  setprecision(1) << fixed << endl;
  for (unsigned int i=0; i<vec.length(); i++)
  {
    cout << *vec.data(i) << " ";
  }
  cout << endl << endl;
}

void printImage(iu::ImageGpu_32f_C1* img_d, string title)
{
  iu::ImageCpu_32f_C1 img(img_d->size());
  iu::copy(img_d, &img);

  cout << "Image " << title << ": (size=" << img_d->width() << "x" << img_d->height()
      << ", stride=" << img_d->stride() << ")"  <<  setprecision(2) << fixed << endl;
  for (unsigned int y=0; y<img.height(); y++)
  {
    for (unsigned int x=0; x<img.width(); x++)
    {
      cout << setw(4) << *img.data(x, y) << " ";
    }
    cout << endl;
  }
  cout << endl;
}

void printImage(iu::ImageGpu_32f_C2* img_d, string title)
{
  iu::ImageCpu_32f_C2 img(img_d->size());
  iu::copy(img_d, &img);

  cout << "Image " << title << ": (size=" << img_d->width() << "x" << img_d->height()
      << ", stride=" << img_d->stride() << ")"  <<  setprecision(2) << fixed << endl;
  for (unsigned int y=0; y<img.height(); y++)
  {
    for (unsigned int x=0; x<img.width(); x++)
    {
      cout << "(" << setw(4) << img.data(x, y)->x << "," << setw(4) << img.data(x, y)->y << ") ";
    }
    cout << endl;
  }
  cout << endl;
}

//void printSparseMatrix(iu::SparseMatrixGpu_32f* mat, string title)
//{
//  iu::LinearHostMemory<int>   row(mat->row()->length());
//  iu::LinearHostMemory<int>   col(mat->col()->length());
//  iu::LinearHostMemory<float> val(mat->value()->length());
//  iu::copy(mat->row(), &row);
//  iu::copy(mat->col(), &col);
//  iu::copy(mat->value(), &val);

//  cout << "Sparse Matrix " << title << ": (size=" << mat->n_col() << "x" << mat->n_row() << ")"
//      <<  setprecision(1) << fixed << endl;
//  unsigned int colInd = 0;
//  for (unsigned int r=0; r<mat->n_row(); r++)
//  {
//    unsigned int nRowElem = *row.data(r+1) - *row.data(r);
//    cout << nRowElem << " == ";
//    if (nRowElem > 0)
//    {
//      for (unsigned int c=0; c<mat->n_col(); c++)
//      {
//        unsigned int match = -1;
//        for (unsigned int ci=colInd; ci<colInd+nRowElem; ci++)
//          if (*col.data(ci) == c)
//            match = ci;

//        if (match >= 0)
//          cout << *val.data(match) << " ";
//        else
//          cout << " 0  ";
//      }
//      colInd = colInd+nRowElem;
//    }
//    else
//      for (unsigned int c=0; c<mat->n_col(); c++)
//        cout << " 0  ";

//    cout << endl;
//  }
//  cout << endl;
//}

void printSparseMatrixElements(iu::LinearHostMemory<int>* row, iu::LinearHostMemory<int>* col, iu::LinearHostMemory<float>* val)
{
  unsigned int maxvals = max(row->length(), col->length());
  cout << "  num   val   col   row" << endl;
  for (unsigned int i=0; i<maxvals; i++)
  {
    cout << setw(5) << i << " ";
    if (i < val->length())
      cout << setw(5) << *val->data(i) << " ";
    else
      cout << "      ";
    if (i < col->length())
      cout<< setw(5) << *col->data(i) << " ";
    else
      cout << "      ";
    if (i < row->length())
      cout<< setw(5) << *row->data(i) << " ";
    else
      cout << "      ";
    cout << endl;
  }
  cout << endl;
}

void printSparseMatrixElements(iu::SparseMatrixGpu_32f* mat, string title)
{
  iu::LinearHostMemory<int>   row(mat->row()->length());
  iu::LinearHostMemory<int>   col(mat->col()->length());
  iu::LinearHostMemory<float> val(mat->value()->length());
  iu::copy(mat->row(), &row);
  iu::copy(mat->col(), &col);
  iu::copy(mat->value(), &val);

  cout << "Sparse Matrix " << title << " Elements: (size=" << mat->n_col() << "x" << mat->n_row() << ")"
      << " IuSparseFormat=" << mat->sparseFormat() <<  setprecision(2) << fixed << endl;

  printSparseMatrixElements(&row, &col, &val);
}

void printSparseMatrix(iu::SparseMatrixGpu_32f* mat, string title, unsigned int width, unsigned int height)
{
  iu::LinearHostMemory<int>   row(mat->row()->length());
  iu::LinearHostMemory<int>   col(mat->col()->length());
  iu::LinearHostMemory<float> val(mat->value()->length());
  iu::copy(mat->row(), &row);
  iu::copy(mat->col(), &col);
  iu::copy(mat->value(), &val);

  cout << "Sparse Matrix " << title << " cropped: (crop=" << width << "x" << height << ", size="
      << mat->n_col() << "x" << mat->n_row() << ")" <<  setprecision(1) << fixed << endl;
  unsigned int colInd = 0;
  for (unsigned int r=0; r<mat->n_row(); r++)
  {
    //    unsigned int nRowElem = *row.data(r+1) - *row.data(r);
    for (unsigned int c=0; c<mat->n_col(); c++)
    {
      unsigned int cc = *col.data(colInd);
      if (cc == c)
      {
        if (c<width && r<height)
          cout << *val.data(colInd) << " ";
        colInd++;
      }
      else
      {
        if (c<width && r<height)
          cout << "0.0 ";
      }
    }
    cout << endl;
  }
  cout << endl;
}
