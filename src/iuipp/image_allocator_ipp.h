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
 * Module      : IPP-Connector
 * Class       : ImageAllocatorIpp
 * Language    : C++
 * Description : Image allocation functions for Ipp images.
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUCORE_IMAGE_ALLOCATOR_IPP_H
#define IUCORE_IMAGE_ALLOCATOR_IPP_H

#include <ippi.h>
#include <iucore/coredefs.h>


namespace iuprivate {

//--------------------------------------------------------------------------
template <typename PixelType, unsigned int NumChannels>
class ImageAllocatorIpp
{
};


// PARTIAL SPECIALIZATIONS:

//--------------------------------------------------------------------------
template<>
class ImageAllocatorIpp<Ipp8u, 1>
{
public:
  static Ipp8u* alloc(unsigned int width, unsigned int height, size_t *pitch)
  {
    if ((width == 0) || (height == 0)) throw IuException("width or height is 0", __FILE__,__FUNCTION__, __LINE__);

    Ipp8u *buffer = ippiMalloc_8u_C1(width, height, reinterpret_cast<int *>(pitch));
    if (buffer == 0) throw std::bad_alloc();

    return buffer;
  }

  static void free(Ipp8u *buffer)
  {
    ippiFree(buffer);
  }

  static void copy(const Ipp8u *src, size_t src_pitch, Ipp8u *dst, size_t dst_pitch, IuSize size)
  {
    IppStatus status;
    IppiSize sz = {size.width, size.height};
    status = ippiCopy_8u_C1R(src, static_cast<int>(src_pitch), dst, static_cast<int>(dst_pitch), sz);
    if (status != ippStsNoErr) throw IuException("ippiCopy returned error code", __FILE__, __FUNCTION__, __LINE__);
  }
};

//--------------------------------------------------------------------------
template<>
class ImageAllocatorIpp<Ipp8u, 2>
{
public:
  static Ipp8u* alloc(unsigned int width, unsigned int height, size_t *pitch)
  {
    if ((width == 0) || (height == 0)) throw IuException("width or height is 0", __FILE__,__FUNCTION__, __LINE__);

    Ipp8u *buffer = ippiMalloc_8u_C2(width, height, reinterpret_cast<int *>(pitch));
    if (buffer == 0) throw std::bad_alloc();

    return buffer;
  }

  static void free(Ipp8u *buffer)
  {
    ippiFree(buffer);
  }

  static void copy(const Ipp8u *src, size_t src_pitch, Ipp8u *dst, size_t dst_pitch, IuSize size)
  {
    IppStatus status;
    // ship around 2-channel copy as there is no ippiCopy_8u_C2
    // we simply double the width and use _C1.
    IppiSize size_C1 = {size.width*2, size.height};
    status = ippiCopy_8u_C1R(src, static_cast<int>(src_pitch), dst, static_cast<int>(dst_pitch), size_C1);
    if (status != ippStsNoErr) throw IuException("ippiCopy returned error code", __FILE__, __FUNCTION__, __LINE__);
  }
};

//--------------------------------------------------------------------------
template<>
class ImageAllocatorIpp<Ipp8u, 3>
{
public:
  static Ipp8u *alloc(unsigned int width, unsigned int height, size_t *pitch)
  {
    if ((width == 0) || (height == 0)) throw IuException("width or height is 0", __FILE__,__FUNCTION__, __LINE__);

    Ipp8u *buffer =  ippiMalloc_8u_C3(width, height, reinterpret_cast<int *>(pitch));
    if (buffer == 0) throw std::bad_alloc();

    return buffer;
  }

  static void free(Ipp8u *buffer)
  {
    ippiFree(buffer);
  }

  static void copy(const Ipp8u *src, size_t src_pitch, Ipp8u *dst, size_t dst_pitch, IuSize size)
  {
    IppStatus status;
    IppiSize sz = {size.width, size.height};
    status = ippiCopy_8u_C3R(src, static_cast<int>(src_pitch), dst, static_cast<int>(dst_pitch), sz);
    if (status != ippStsNoErr) throw IuException("ippiCopy returned error code", __FILE__, __FUNCTION__, __LINE__);
  }
};

//--------------------------------------------------------------------------
template<>
class ImageAllocatorIpp<Ipp8u, 4>
{
public:
  static Ipp8u* alloc(unsigned int width, unsigned int height, size_t *pitch)
  {
    if ((width == 0) || (height == 0)) throw IuException("width or height is 0", __FILE__,__FUNCTION__, __LINE__);

    Ipp8u *buffer = ippiMalloc_8u_C4(width, height, reinterpret_cast<int *>(pitch));
    if (buffer == 0) throw std::bad_alloc();

    return buffer;
  }

  static void free(Ipp8u *buffer)
  {
    ippiFree(buffer);
  }

  static void copy(const Ipp8u *src, size_t src_pitch, Ipp8u *dst, size_t dst_pitch, IuSize size)
  {
    IppStatus status;
    IppiSize sz = {size.width, size.height};
    status = ippiCopy_8u_C4R(src, static_cast<int>(src_pitch), dst, static_cast<int>(dst_pitch), sz);
    if (status != ippStsNoErr) throw IuException("ippiCopy returned error code", __FILE__, __FUNCTION__, __LINE__);
  }
};

//--------------------------------------------------------------------------
template<>
class ImageAllocatorIpp<Ipp32f, 1>
{
public:
  static Ipp32f* alloc(unsigned int width, unsigned int height, size_t *pitch)
  {
    if ((width == 0) || (height == 0)) throw IuException("width or height is 0", __FILE__,__FUNCTION__, __LINE__);

    Ipp32f *buffer = ippiMalloc_32f_C1(width, height, reinterpret_cast<int *>(pitch));
    if (buffer == 0) throw std::bad_alloc();

    return buffer;
  }

  static void free(Ipp32f *buffer)
  {
    ippiFree(buffer);
  }

  static void copy(const Ipp32f *src, size_t src_pitch, Ipp32f *dst, size_t dst_pitch, IuSize size)
  {
    IppStatus status;
    IppiSize sz = {size.width, size.height};
    status = ippiCopy_32f_C1R(src, static_cast<int>(src_pitch), dst, static_cast<int>(dst_pitch), sz);
    if (status != ippStsNoErr) throw IuException("ippiCopy returned error code", __FILE__, __FUNCTION__, __LINE__);
  }
};

//--------------------------------------------------------------------------
template<>
class ImageAllocatorIpp<Ipp32f, 2>
{
public:
  static Ipp32f* alloc(unsigned int width, unsigned int height, size_t *pitch)
  {
    if ((width == 0) || (height == 0)) throw IuException("width or height is 0", __FILE__,__FUNCTION__, __LINE__);

    Ipp32f *buffer = ippiMalloc_32f_C2(width, height, reinterpret_cast<int *>(pitch));
    if (buffer == 0) throw std::bad_alloc();

    return buffer;
  }

  static void free(Ipp32f *buffer)
  {
    ippiFree(buffer);
  }

  static void copy(const Ipp32f *src, size_t src_pitch, Ipp32f *dst, size_t dst_pitch, IuSize size)
  {
    IppStatus status;
    // ship around 2-channel copy as there is no ippiCopy_32f_C2
    // we simply double the width and use _C1.
    IppiSize size_C1 = {size.width*2, size.height};
    status = ippiCopy_32f_C1R(src, static_cast<int>(src_pitch), dst, static_cast<int>(dst_pitch), size_C1);
    if (status != ippStsNoErr) throw IuException("ippiCopy returned error code", __FILE__, __FUNCTION__, __LINE__);
  }
};

//--------------------------------------------------------------------------
template<>
class ImageAllocatorIpp<Ipp32f, 3>
{
public:
  static Ipp32f* alloc(unsigned int width, unsigned int height, size_t *pitch)
  {
    if ((width == 0) || (height == 0)) throw IuException("width or height is 0", __FILE__,__FUNCTION__, __LINE__);

    Ipp32f *buffer = ippiMalloc_32f_C3(width, height, reinterpret_cast<int *>(pitch));
    if (buffer == 0) throw std::bad_alloc();

    return buffer;
  }

  static void free(Ipp32f *buffer)
  {
    ippiFree(buffer);
  }

  static void copy(const Ipp32f *src, size_t src_pitch, Ipp32f *dst, size_t dst_pitch, IuSize size)
  {
    IppStatus status;
    IppiSize sz = {size.width, size.height};
    status = ippiCopy_32f_C3R(src, static_cast<int>(src_pitch), dst, static_cast<int>(dst_pitch), sz);
    if (status != ippStsNoErr) throw IuException("ippiCopy returned error code", __FILE__, __FUNCTION__, __LINE__);
  }
};

//--------------------------------------------------------------------------
template<>
class ImageAllocatorIpp<Ipp32f, 4>
{
public:
  static Ipp32f* alloc(unsigned int width, unsigned int height, size_t *pitch)
  {
    if ((width == 0) || (height == 0)) throw IuException("width or height is 0", __FILE__,__FUNCTION__, __LINE__);

    Ipp32f *buffer = ippiMalloc_32f_C4(width, height, reinterpret_cast<int *>(pitch));
    if (buffer == 0) throw std::bad_alloc();

    return buffer;
  }

  static void free(Ipp32f *buffer)
  {
    ippiFree(buffer);
  }

  static void copy(const Ipp32f *src, size_t src_pitch, Ipp32f *dst, size_t dst_pitch, IuSize size)
  {
    IppStatus status;
    IppiSize sz = {size.width, size.height};
    status = ippiCopy_32f_C4R(src, static_cast<int>(src_pitch), dst, static_cast<int>(dst_pitch), sz);
    if (status != ippStsNoErr) throw IuException("ippiCopy returned error code", __FILE__, __FUNCTION__, __LINE__);
  }
};

} // namespace iuprivate

#endif // IUCORE_IMAGE_ALLOCATOR_IPP_H
