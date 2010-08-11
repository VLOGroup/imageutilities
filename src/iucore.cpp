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
 * Module      : Core Module
 * Class       : Wrapper
 * Language    : C
 * Description : Implementation of public interfaces to core module
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#include "iucore.h"
#include "core/copy.h"
#include "core/memory_modification.h"

namespace iu {

/* ***************************************************************************
 * 1D COPY
 * ***************************************************************************/

// 1D copy host -> host;
void copy(const LinearHostMemory_8u* src, LinearHostMemory_8u* dst)
{ iuprivate::copy(src,dst); }
void copy(const LinearHostMemory_32f* src, LinearHostMemory_32f* dst)
{ iuprivate::copy(src,dst); }

// 1D copy device -> device;
void copy(const LinearDeviceMemory_8u* src, LinearDeviceMemory_8u* dst)
{ iuprivate::copy(src,dst); }
void copy(const LinearDeviceMemory_32f* src, LinearDeviceMemory_32f* dst)
{ iuprivate::copy(src,dst); }

// 1D copy host -> device;
void copy(const LinearHostMemory_8u* src, LinearDeviceMemory_8u* dst)
{ iuprivate::copy(src,dst); }
void copy(const LinearHostMemory_32f* src, LinearDeviceMemory_32f* dst)
{ iuprivate::copy(src,dst); }

// 1D copy device -> host;
void copy(const LinearDeviceMemory_8u* src, LinearHostMemory_8u* dst)
{ iuprivate::copy(src,dst); }
void copy(const LinearDeviceMemory_32f* src, LinearHostMemory_32f* dst)
{ iuprivate::copy(src,dst); }

/* ***************************************************************************
 * 2D COPY
 * ***************************************************************************/

// 2D copy host -> host;
void copy(const ImageCpu_8u_C1* src, ImageCpu_8u_C1* dst) { iuprivate::copy(src, dst); }
void copy(const ImageCpu_8u_C2* src, ImageCpu_8u_C2* dst) { iuprivate::copy(src, dst); }
void copy(const ImageCpu_8u_C3* src, ImageCpu_8u_C3* dst) { iuprivate::copy(src, dst); }
void copy(const ImageCpu_8u_C4* src, ImageCpu_8u_C4* dst) { iuprivate::copy(src, dst); }
void copy(const ImageCpu_32f_C1* src, ImageCpu_32f_C1* dst) { iuprivate::copy(src, dst); }
void copy(const ImageCpu_32f_C2* src, ImageCpu_32f_C2* dst) { iuprivate::copy(src, dst); }
void copy(const ImageCpu_32f_C3* src, ImageCpu_32f_C3* dst) { iuprivate::copy(src, dst); }
void copy(const ImageCpu_32f_C4* src, ImageCpu_32f_C4* dst) { iuprivate::copy(src, dst); }

// 2D copy device -> device;
void copy(const ImageNpp_8u_C1* src, ImageNpp_8u_C1* dst) { iuprivate::copy(src, dst); }
void copy(const ImageNpp_8u_C2* src, ImageNpp_8u_C2* dst) { iuprivate::copy(src, dst); }
void copy(const ImageNpp_8u_C3* src, ImageNpp_8u_C3* dst) { iuprivate::copy(src, dst); }
void copy(const ImageNpp_8u_C4* src, ImageNpp_8u_C4* dst) { iuprivate::copy(src, dst); }
void copy(const ImageNpp_32f_C1* src, ImageNpp_32f_C1* dst) { iuprivate::copy(src, dst); }
void copy(const ImageNpp_32f_C2* src, ImageNpp_32f_C2* dst) { iuprivate::copy(src, dst); }
void copy(const ImageNpp_32f_C3* src, ImageNpp_32f_C3* dst) { iuprivate::copy(src, dst); }
void copy(const ImageNpp_32f_C4* src, ImageNpp_32f_C4* dst) { iuprivate::copy(src, dst); }

// 2D copy host -> device;
void copy(const ImageCpu_8u_C1* src, ImageNpp_8u_C1* dst) { iuprivate::copy(src, dst); }
void copy(const ImageCpu_8u_C2* src, ImageNpp_8u_C2* dst) { iuprivate::copy(src, dst); }
void copy(const ImageCpu_8u_C3* src, ImageNpp_8u_C3* dst) { iuprivate::copy(src, dst); }
void copy(const ImageCpu_8u_C4* src, ImageNpp_8u_C4* dst) { iuprivate::copy(src, dst); }
void copy(const ImageCpu_32f_C1* src, ImageNpp_32f_C1* dst) { iuprivate::copy(src, dst); }
void copy(const ImageCpu_32f_C2* src, ImageNpp_32f_C2* dst) { iuprivate::copy(src, dst); }
void copy(const ImageCpu_32f_C3* src, ImageNpp_32f_C3* dst) { iuprivate::copy(src, dst); }
void copy(const ImageCpu_32f_C4* src, ImageNpp_32f_C4* dst) { iuprivate::copy(src, dst); }

// 2D copy device -> host;
void copy(const ImageNpp_8u_C1* src, ImageCpu_8u_C1* dst) { iuprivate::copy(src, dst); }
void copy(const ImageNpp_8u_C2* src, ImageCpu_8u_C2* dst) { iuprivate::copy(src, dst); }
void copy(const ImageNpp_8u_C3* src, ImageCpu_8u_C3* dst) { iuprivate::copy(src, dst); }
void copy(const ImageNpp_8u_C4* src, ImageCpu_8u_C4* dst) { iuprivate::copy(src, dst); }
void copy(const ImageNpp_32f_C1* src, ImageCpu_32f_C1* dst) { iuprivate::copy(src, dst); }
void copy(const ImageNpp_32f_C2* src, ImageCpu_32f_C2* dst) { iuprivate::copy(src, dst); }
void copy(const ImageNpp_32f_C3* src, ImageCpu_32f_C3* dst) { iuprivate::copy(src, dst); }
void copy(const ImageNpp_32f_C4* src, ImageCpu_32f_C4* dst) { iuprivate::copy(src, dst); }


/* ***************************************************************************
     SET
 * ***************************************************************************/

// 1D set value; host; 8-bit
void setValue(Npp8u value, LinearHostMemory_8u* srcdst)
{iuprivate::setValue(value, srcdst);}
void setValue(float value, LinearHostMemory_32f* srcdst)
{iuprivate::setValue(value, srcdst);}
void setValue(Npp8u value, LinearDeviceMemory_8u* srcdst)
{iuprivate::setValue(value, srcdst);}
void setValue(Npp32f value, LinearDeviceMemory_32f* srcdst)
{iuprivate::setValue(value, srcdst);}

void setValue(const Npp8u &value, ImageCpu_8u_C1* srcdst, const IuRect& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const Npp8u &value, ImageCpu_8u_C2* srcdst, const IuRect& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const Npp8u &value, ImageCpu_8u_C3* srcdst, const IuRect& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const Npp8u &value, ImageCpu_8u_C4* srcdst, const IuRect& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const Npp32f &value, ImageCpu_32f_C1* srcdst, const IuRect& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const Npp32f &value, ImageCpu_32f_C2* srcdst, const IuRect& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const Npp32f &value, ImageCpu_32f_C3* srcdst, const IuRect& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const Npp32f &value, ImageCpu_32f_C4* srcdst, const IuRect& roi) {iuprivate::setValue(value, srcdst, roi);}

void setValue(const Npp8u &value, ImageNpp_8u_C1* srcdst, const IuRect& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const Npp8u &value, ImageNpp_8u_C2* srcdst, const IuRect& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const Npp8u &value, ImageNpp_8u_C3* srcdst, const IuRect& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const Npp8u &value, ImageNpp_8u_C4* srcdst, const IuRect& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const Npp32f &value, ImageNpp_32f_C1* srcdst, const IuRect& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const Npp32f &value, ImageNpp_32f_C2* srcdst, const IuRect& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const Npp32f &value, ImageNpp_32f_C3* srcdst, const IuRect& roi) {iuprivate::setValue(value, srcdst, roi);}
void setValue(const Npp32f &value, ImageNpp_32f_C4* srcdst, const IuRect& roi) {iuprivate::setValue(value, srcdst, roi);}


/* ***************************************************************************
     CLAMP
 * ***************************************************************************/

void clamp(const Npp32f& min, const Npp32f& max, iu::ImageNpp_32f_C1 *srcdst, const IuRect &roi)
{ iuprivate::clamp(min, max, srcdst, roi); }


///* ***************************************************************************
//     MEMORY CONVERSIONS
// * ***************************************************************************/

// conversion; device; 32-bit 3-channel -> 32-bit 4-channel
void convert(const ImageNpp_32f_C3* src, const IuRect& src_roi, ImageNpp_32f_C4* dst, const IuRect& dst_roi)
{iuprivate::convert(src, src_roi, dst, dst_roi);}
// conversion; device; 32-bit 4-channel -> 32-bit 3-channel
void convert(const ImageNpp_32f_C4* src, const IuRect& src_roi, ImageNpp_32f_C3* dst, const IuRect& dst_roi)
{iuprivate::convert(src, src_roi, dst, dst_roi);}


} // namespace iu
