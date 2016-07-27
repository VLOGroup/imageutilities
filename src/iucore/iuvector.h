/*
 * iuvector.h
 *
 *  Created on: Jul 22, 2016
 *      Author: kerstin
 */

#ifndef IUVECTOR_H_
#define IUVECTOR_H_

#include "coredefs.h"
#include <type_traits>

namespace iu {
template<typename PixelType, unsigned int Ndim>
class VectorBase
{
IU_ASSERT(Ndim > 0)

public:
  VectorBase()
  {
    for (unsigned int i = 0; i < Ndim; i++)
    {
      data_[i] = 0.0;
    }
  }

  ~VectorBase()
  {
  }

  inline PixelType operator[](unsigned int i) const
  {
    return data_[i];
  }

  inline PixelType &operator[](unsigned int i)
  {
    return data_[i];
  }

  static unsigned int ndim()
  {
    return Ndim;
  }

  void fill(const PixelType& value)
  {
    for (unsigned int i = 0; i < Ndim; i++)
    {
      data_[i] = value;
    }
  }

  template<typename T = std::ostream>
  friend typename std::enable_if<(Ndim==1), T&>::type operator<<(std::ostream & out,
                                  VectorBase<PixelType, Ndim> const& v)
  {
    out << "[" << v[0] << "]";
    return out;
  }

  template<typename T = std::ostream>
  friend typename std::enable_if< (Ndim>1), T&>::type operator<<(std::ostream & out,
                                  VectorBase<PixelType, Ndim> const& v)
  {
    out << "[";
    for (int i = 0; i < static_cast<int>(Ndim - 1); i++)
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

template<typename PixelType, unsigned int Ndim>
bool operator==(const VectorBase<PixelType, Ndim> &v1,
                const VectorBase<PixelType, Ndim> &v2)
{
  for (unsigned int i = 0; i < Ndim; i++)
  {
    if (v1[i] != v2[i])
      return false;
  }
  return true;
}

template<typename PixelType, unsigned int Ndim>
bool operator!=(const VectorBase<PixelType, Ndim> &v1,
                const VectorBase<PixelType, Ndim> &v2)
{
  for (unsigned int i = 0; i < Ndim; i++)
  {
    if (v1[i] != v2[i])
      return true;
  }
  return false;
}

template<unsigned int Ndim>
class SizeBase: public VectorBase<unsigned int, Ndim>
{
public:
  SizeBase() :
      VectorBase<unsigned int, Ndim>()
  {
    this->fill(1);
  }

  SizeBase(unsigned int value) : VectorBase<unsigned int, Ndim>()
  {
    this->fill(value);
  }

  ~SizeBase()
  {
  }

  unsigned int numel() const
  {
    unsigned int num_elements = this->data_[0];
    for (unsigned int i = 1; i < Ndim; i++)
    {
      num_elements *= this->data_[i];
    }
    return num_elements;
  }

  template<typename ScalarType>
  SizeBase operator*(const ScalarType& scalar) const
  {
    SizeBase<Ndim> v;
    for (unsigned int i = 0; i < Ndim; i++)
    {
      v[i] = static_cast<unsigned int>(this->data_[i] * scalar + 0.5f);
    }
    return v;
  }

  template<typename ScalarType>
  void operator*=(const ScalarType &scalar)
  {
    for (unsigned int i = 0; i < Ndim; i++)
    {
      this->data_[i] = static_cast<unsigned int>(this->data_[i] * scalar + 0.5f);
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

  SizeBase(const SizeBase& from)
  {
    for (unsigned int i = 0; i < Ndim; i++)
      this->data_[i] = from[i];
  }

  SizeBase& operator=(const SizeBase& from)
  {
    for (unsigned int i = 0; i < Ndim; i++)
      this->data_[i] = from[i];
    return *this;
  }

  // this is only used in kernel data of device memory
  unsigned int* ptr()
  {
    return this->data_;
  }

  const unsigned int* ptr() const
  {
    return reinterpret_cast<const unsigned int*>(this->data_);
  }
};

template<unsigned int Ndim>
class Size: public SizeBase<Ndim>
{
public:
  Size() : SizeBase<Ndim>() {}
  Size(unsigned int value) : SizeBase<Ndim>(value) {}

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
  unsigned int& width;
  unsigned int& height;
  unsigned int& depth;

  Size() : SizeBase<3>(),
      width(this->data_[0]), height(this->data_[1]), depth(this->data_[2])
  {
  }

  Size(unsigned int value) : SizeBase<3>(value),
      width(this->data_[0]), height(this->data_[1]), depth(this->data_[2])
  {
  }

  Size(unsigned int width, unsigned int height, unsigned int depth=0) : SizeBase<3>(),
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


template<typename PixelType, unsigned int Ndim>
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
    for (unsigned int i = 0; i < Ndim; i++)
    {
      v[i] = this->data_[i] * scalar;
    }
    return v;
  }

  inline Vector<PixelType, Ndim> operator*(Vector<PixelType, Ndim> &v1) const
  {
    Vector<PixelType, Ndim> v2;
    for (unsigned int i = 0; i < Ndim; i++)
    {
      v2[i] = this->data_[i] * v1[i];
    }
    return v2;
  }

  void operator*=(const Vector<PixelType, Ndim> &v2)
  {
    for (unsigned int i = 0; i < Ndim; i++)
    {
      this->data_[i] *= v2[i];
    }
  }

  void operator*=(const PixelType &scalar)
  {
    for (unsigned int i = 0; i < Ndim; i++)
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
