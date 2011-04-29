/*
 * Copyright (c) ICG. All rights reserved.
 *
 * Institute for Computer Graphics and Vision
 * Graz University of Technology / Austria
 *
 *
 * This software is distributed WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 * PURPOSE.  See the above copyright notices for more information.
 *
 *
 * Project     : ImageUtilities
 * Module      : Core
 * Class       : LinearHostMemory
 * Language    : C++
 * Description : Inline implementation of a linear host memory class
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUSPARSE_SPARSEMATRIXCPU_H
#define IUSPARSE_SPARSEMATRIXCPU_H

#include <stdio.h>
#include <assert.h>
#include <cstdlib>

#include <iucore/linearhostmemory.h>

namespace iu {

template<typename PixelType>
class SparseMatrixCpu
{
public:
  SparseMatrixCpu() :
      n_row_(0), n_col_(0), n_elements_(0),
      value_(0), row_(0), col_(0), sformat_(CSR), ext_data_pointer_(false)
  {
  }

  virtual ~SparseMatrixCpu()
  {
    if((!ext_data_pointer_) && (value_!=NULL))
    {
      delete value_;
      value_ = 0;
    }
    if((!ext_data_pointer_) && (row_!=NULL))
    {
      delete row_;
      row_ = 0;
    }
    if((!ext_data_pointer_) && (col_!=NULL))
    {
      delete col_;
      col_ = 0;
    }
  }

  SparseMatrixCpu(LinearHostMemory<PixelType>* value, LinearHostMemory<int>* row,
                  LinearHostMemory<int>* col, int n_row, int n_col,
                  IuSparseFormat sformat, bool ext_data_pointer = false) :
     n_row_(n_row), n_col_(n_col), n_elements_(0),
     value_(0), row_(0), col_(0), sformat_(sformat), ext_data_pointer_(ext_data_pointer)
  {
    if(ext_data_pointer_)
    {
      // This uses the external memory as internal one.
      value_ = value;
      row_ = row;
      col_ = col;
    }
    else
    {
      // allocates internal memory
      if ((value == 0) || (row == 0) || (col == 0))
        return;

      value_ = new LinearHostMemory<PixelType>(*value);
      row_ = new LinearHostMemory<int>(*row);
      col_ = new LinearHostMemory<int>(*col);
    }

    n_elements_ = value->length();
  }

  const LinearHostMemory<PixelType>* value() const
  {
    return reinterpret_cast<const LinearHostMemory<PixelType>*>(value_);
  }

  const LinearHostMemory<int>* row() const
  {
    return reinterpret_cast<const LinearHostMemory<int>*>(row_);
  }

  const LinearHostMemory<int>* col() const
  {
    return reinterpret_cast<const LinearHostMemory<int>*>(col_);
  }

  int n_elements()
  {
    return n_elements_;
  }

  int n_row()
  {
    return n_row_;
  }

  int n_col()
  {
    return n_col_;
  }

  IuSparseFormat sparseFormat()
  {
    return sformat_;
  }

protected:

private:
  int n_row_;      /**< Number of rows in the sparse matrix */
  int n_col_;      /**< Number of columns in the sparse matrix */
  int n_elements_; /**< Number of non-zero elements in the sparse matrix */

  LinearHostMemory<PixelType>* value_;
  LinearHostMemory<int>* row_;
  LinearHostMemory<int>* col_;

  IuSparseFormat sformat_;

  bool ext_data_pointer_; /**< Flag if data pointer is handled outside the image class. */

};

} // namespace iu

#endif // IUSPARSE_SPARSEMATRIXCPU_H
