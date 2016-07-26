/*
 * iuvector.h
 *
 *  Created on: Jul 22, 2016
 *      Author: kerstin
 */

#ifndef IUVECTOR_H_
#define IUVECTOR_H_

#include "coredefs.h"

namespace iu {
template<typename PixelType, int Ndim>
class VectorBase
{
IU_ASSERT(Ndim > 0)

public:
  VectorBase()
  {
    for (int i = 0; i < Ndim; i++)
    {
      data_[i] = 0.0;
    }
  }

  ~VectorBase()
  {
  }

  inline PixelType operator[](int i) const
  {
    return data_[i];
  }

  inline PixelType &operator[](int i)
  {
    return data_[i];
  }

  static unsigned int ndim()
  {
    return Ndim;
  }

  void fill(const PixelType& value)
  {
    for (int i = 0; i < Ndim; i++)
    {
      data_[i] = value;
    }
  }

  friend std::ostream& operator<<(std::ostream & out,
                                  VectorBase<PixelType, Ndim> const& v)
  {
    out << "[";
    for (int i = 0; i < Ndim - 1; i++)
    {
      out << v[i] << ", ";
    }
    out << v[Ndim - 1] << "]";
    return out;
  }

protected:
  PixelType data_[Ndim];

private:
  VectorBase(const VectorBase& from);
  VectorBase& operator=(const VectorBase& from);

};

template<typename PixelType, int Ndim>
bool operator==(const VectorBase<PixelType, Ndim> &v1,
                const VectorBase<PixelType, Ndim> &v2)
{
  for (int i = 0; i < Ndim; i++)
  {
    if (v1[i] != v2[i])
      return false;
  }
  return true;
}

template<typename PixelType, int Ndim>
bool operator!=(const VectorBase<PixelType, Ndim> &v1,
                const VectorBase<PixelType, Ndim> &v2)
{
  for (int i = 0; i < Ndim; i++)
  {
    if (v1[i] != v2[i])
      return true;
  }
  return false;
}

template<int Ndim>
class SizeBase: public VectorBase<int, Ndim>
{
public:
  SizeBase() :
      VectorBase<int, Ndim>()
  {
  }

  SizeBase(int value) : VectorBase<int, Ndim>()
  {
    this->fill(value);
  }

  ~SizeBase()
  {
  }
  unsigned int numel() const
  {
    int num_elements = this->data_[0];
    for (int i = 1; i < Ndim; i++)
    {
      num_elements *= this->data_[i];
    }
    return num_elements;
  }

  template<typename ScalarType>
  SizeBase operator*(const ScalarType& scalar) const
  {
    SizeBase<Ndim> v;
    for (int i = 0; i < Ndim; i++)
    {
      v[i] = static_cast<int>(this->data_[i] * scalar + 0.5f);
    }
    return v;
  }

  template<typename ScalarType>
  void operator*=(const ScalarType &scalar)
  {
    for (int i = 0; i < Ndim; i++)
    {
      this->data_[i] = static_cast<int>(this->data_[i] * scalar + 0.5f);
      ;
    }
  }

  template<typename ScalarType>
  SizeBase operator/(const ScalarType& scalar) const
  {
    IU_ASSERT(scalar != 0);
    double invFactor = 1.0 / static_cast<double>(scalar);
    return operator*(invFactor);
  }

  template<typename ScalarType>
  void operator/=(const ScalarType &scalar)
  {
    IU_ASSERT(scalar != 0);
    double invFactor = 1.0 / static_cast<double>(scalar);
    operator*=(invFactor);
  }

  friend std::ostream& operator<<(std::ostream & out, SizeBase<Ndim> const& v)
  {
    out << "[";
    for (int i = 0; i < Ndim - 1; i++)
      out << v[i] << ", ";
    out << v[Ndim - 1] << "];";
    return out;
  }

  SizeBase(const SizeBase& from)
  {
    for (int i = 0; i < Ndim; i++)
      this->data_[i] = from[i];
  }

  SizeBase& operator=(const SizeBase& from)
  {
    for (int i = 0; i < Ndim; i++)
      this->data_[i] = from[i];
    return *this;
  }

  // this is only used in kernel data of device memory
  int* ptr()
  {
    return this->data_;
  }

  const int* ptr() const
  {
    return reinterpret_cast<const int*>(this->data_);
  }
};

template<int Ndim>
class Size: public SizeBase<Ndim>
{
public:
  Size() : SizeBase<Ndim>() {}

  ~Size() {}

  Size(const Size& from) : SizeBase<Ndim>(from) {}

  Size& operator=(const Size& from)
  {
    SizeBase<Ndim>::operator=(from);
    return *this;
  }

  Size& operator=(const SizeBase<Ndim>& from)
  {
    SizeBase<Ndim>::operator=(from);
    return *this;
  }

};

template<>
class Size<3> : public SizeBase<3>
{
public:
  int& width;
  int& height;
  int& depth;

  Size() : SizeBase<3>(),
      width(this->data_[0]), height(this->data_[1]), depth(this->data_[2])
  {
  }

  Size(int width, int height, int depth=0) : SizeBase<3>(),
      width(this->data_[0]), height(this->data_[1]), depth(this->data_[2])
  {
    data_[0] = width;
    data_[1] = height;
    data_[2] = depth;
  }

  Size(const Size& from) : SizeBase<3>(from),
      width(this->data_[0]), height(this->data_[1]), depth(this->data_[2])
  {
  }

  ~Size() {}

  Size& operator=(const Size& from)
  {
    SizeBase::operator=(from);
    return *this;
  }

  Size& operator=(const SizeBase<3>& from)
  {
    SizeBase::operator=(from);
    return *this;
  }
};


template<typename PixelType, int Ndim>
class Vector: public VectorBase<PixelType, Ndim>
{
public:
  Vector() :
      VectorBase<PixelType, Ndim>()
  {
  }
  ~Vector()
  {
  }
  inline Vector<PixelType, Ndim> operator*(const PixelType &scalar) const
  {
    Vector<PixelType, Ndim> v;
    for (int i = 0; i < Ndim; i++)
    {
      v[i] = this->data_[i] * scalar;
    }
    return v;
  }

  inline Vector<PixelType, Ndim> operator*(Vector<PixelType, Ndim> &v1) const
  {
    Vector<PixelType, Ndim> v2;
    for (int i = 0; i < Ndim; i++)
    {
      v2[i] = this->data_[i] * v1[i];
    }
    return v2;
  }

  void operator*=(const Vector<PixelType, Ndim> &v2)
  {
    for (int i = 0; i < Ndim; i++)
    {
      this->data_[i] *= v2[i];
    }
  }

  void operator*=(const PixelType &scalar)
  {
    for (int i = 0; i < Ndim; i++)
    {
      this->data_[i] *= scalar;
    }
  }

  Vector operator/(const PixelType scalar) const
  {
    IU_ASSERT(scalar != 0);
    PixelType invFactor = 1 / scalar;
    return operator*(invFactor);
  }
};

}  //namespace iu

typedef iu::Size<3> IuSize;

#endif /* IUVECTOR_H_ */
