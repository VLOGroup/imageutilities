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
 * Description : Public interfaces to core module
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger\icg.tugraz.at
 *
 */

#ifndef IU_CORE_MODULE_H
#define IU_CORE_MODULE_H

#include "iudefs.h"

namespace iu {

/** \defgroup Core The core module.
 *  TODO more detailed docu
 */

/* ***************************************************************************
     COPY
 * ***************************************************************************/

//////////////////////////////////////////////////////////////////////////////
/** \defgroup Copy1D 1D Memory Copy
 *  \ingroup Core
 *  TODO more detailed docu
 *  \{
 */

// copy host -> host;
/** Copy methods for host to host 1D copy methods for 8bit buffers.
 * \param src Source buffer [host]
 * \param dst Destination buffer [host]
 */
IU_DLLAPI void copy(const LinearHostMemory_8u_C1* src, LinearHostMemory_8u_C1* dst);
IU_DLLAPI void copy(const LinearHostMemory_32f_C1* src, LinearHostMemory_32f_C1* dst);

// copy device -> device;
/** Copy methods for host to host 1D copy methods for 8bit buffers.
 * \param src Source buffer [host]
 * \param dst Destination buffer [host]
 */
IU_DLLAPI void copy(const LinearDeviceMemory_8u_C1* src, LinearDeviceMemory_8u_C1* dst);
IU_DLLAPI void copy(const LinearDeviceMemory_8u_C2* src, LinearDeviceMemory_8u_C2* dst);
IU_DLLAPI void copy(const LinearDeviceMemory_8u_C3* src, LinearDeviceMemory_8u_C3* dst);
IU_DLLAPI void copy(const LinearDeviceMemory_8u_C4* src, LinearDeviceMemory_8u_C4* dst);
IU_DLLAPI void copy(const LinearDeviceMemory_32f_C1* src, LinearDeviceMemory_32f_C1* dst);
IU_DLLAPI void copy(const LinearDeviceMemory_32f_C2* src, LinearDeviceMemory_32f_C2* dst);
IU_DLLAPI void copy(const LinearDeviceMemory_32f_C3* src, LinearDeviceMemory_32f_C3* dst);
IU_DLLAPI void copy(const LinearDeviceMemory_32f_C4* src, LinearDeviceMemory_32f_C4* dst);

// copy host -> device;
/** Copy methods for host to host 1D copy methods for 8bit buffers.
 * \param src Source buffer [host]
 * \param dst Destination buffer [device]
 */
IU_DLLAPI void copy(const LinearHostMemory_8u_C1* src, LinearDeviceMemory_8u_C1* dst);
IU_DLLAPI void copy(const LinearHostMemory_8u_C2* src, LinearDeviceMemory_8u_C2* dst);
IU_DLLAPI void copy(const LinearHostMemory_8u_C3* src, LinearDeviceMemory_8u_C3* dst);
IU_DLLAPI void copy(const LinearHostMemory_8u_C4* src, LinearDeviceMemory_8u_C4* dst);
IU_DLLAPI void copy(const LinearHostMemory_32f_C1* src, LinearDeviceMemory_32f_C1* dst);
IU_DLLAPI void copy(const LinearHostMemory_32f_C2* src, LinearDeviceMemory_32f_C2* dst);
IU_DLLAPI void copy(const LinearHostMemory_32f_C3* src, LinearDeviceMemory_32f_C3* dst);
IU_DLLAPI void copy(const LinearHostMemory_32f_C4* src, LinearDeviceMemory_32f_C4* dst);

// copy device -> host;
/** Copy methods for host to host 1D copy methods for 8bit buffers.
 * \param src Source buffer [device]
 * \param dst Destination buffer [host]
 */
IU_DLLAPI void copy(const LinearDeviceMemory_8u_C1* src, LinearHostMemory_8u_C1* dst);
IU_DLLAPI void copy(const LinearDeviceMemory_8u_C2* src, LinearHostMemory_8u_C2* dst);
IU_DLLAPI void copy(const LinearDeviceMemory_8u_C3* src, LinearHostMemory_8u_C3* dst);
IU_DLLAPI void copy(const LinearDeviceMemory_8u_C4* src, LinearHostMemory_8u_C4* dst);
IU_DLLAPI void copy(const LinearDeviceMemory_32f_C1* src, LinearHostMemory_32f_C1* dst);
IU_DLLAPI void copy(const LinearDeviceMemory_32f_C2* src, LinearHostMemory_32f_C2* dst);
IU_DLLAPI void copy(const LinearDeviceMemory_32f_C3* src, LinearHostMemory_32f_C3* dst);
IU_DLLAPI void copy(const LinearDeviceMemory_32f_C4* src, LinearHostMemory_32f_C4* dst);

/** \} */ // end of Copy1D

//////////////////////////////////////////////////////////////////////////////
/** \defgroup Copy2D 2D Memory Copy
 *  \ingroup Core
 *  Copy methods for 2D images of various types.
 *  \{
 */

// 2D; copy host -> host;
/** Copy methods for host to host 2D copy
 * \param src Source image [host].
 * \param src_roi Region of interest in the source image.
 * \param dst Destination image [host]
 * \param dst_roi Region of interest in the dsetination image.
 */
IU_DLLAPI void copy(const ImageCpu_8u_C1* src, ImageCpu_8u_C1* dst);
IU_DLLAPI void copy(const ImageCpu_8u_C2* src, ImageCpu_8u_C2* dst);
IU_DLLAPI void copy(const ImageCpu_8u_C3* src, ImageCpu_8u_C3* dst);
IU_DLLAPI void copy(const ImageCpu_8u_C4* src, ImageCpu_8u_C4* dst);
IU_DLLAPI void copy(const ImageCpu_32f_C1* src, ImageCpu_32f_C1* dst);
IU_DLLAPI void copy(const ImageCpu_32f_C2* src, ImageCpu_32f_C2* dst);
IU_DLLAPI void copy(const ImageCpu_32f_C3* src, ImageCpu_32f_C3* dst);
IU_DLLAPI void copy(const ImageCpu_32f_C4* src, ImageCpu_32f_C4* dst);

// 2D; copy device -> device;
/** Copy methods for device to device 2D copy
 * \param src Source image [device].
 * \param src_roi Region of interest in the source image.
 * \param dst Destination image [device]
 * \param dst_roi Region of interest in the dsetination image.
 */
IU_DLLAPI void copy(const ImageGpu_8u_C1* src, ImageGpu_8u_C1* dst);
IU_DLLAPI void copy(const ImageGpu_8u_C2* src, ImageGpu_8u_C2* dst);
IU_DLLAPI void copy(const ImageGpu_8u_C3* src, ImageGpu_8u_C3* dst);
IU_DLLAPI void copy(const ImageGpu_8u_C4* src, ImageGpu_8u_C4* dst);
IU_DLLAPI void copy(const ImageGpu_32f_C1* src, ImageGpu_32f_C1* dst);
IU_DLLAPI void copy(const ImageGpu_32f_C2* src, ImageGpu_32f_C2* dst);
IU_DLLAPI void copy(const ImageGpu_32f_C3* src, ImageGpu_32f_C3* dst);
IU_DLLAPI void copy(const ImageGpu_32f_C4* src, ImageGpu_32f_C4* dst);

// 2D; copy host -> device;
/** Copy methods for host to device 2D copy
 * \param src Source image [host].
 * \param src_roi Region of interest in the source image.
 * \param dst Destination image [device]
 * \param dst_roi Region of interest in the dsetination image.
 */
IU_DLLAPI void copy(const ImageCpu_8u_C1* src, ImageGpu_8u_C1* dst);
IU_DLLAPI void copy(const ImageCpu_8u_C2* src, ImageGpu_8u_C2* dst);
IU_DLLAPI void copy(const ImageCpu_8u_C3* src, ImageGpu_8u_C3* dst);
IU_DLLAPI void copy(const ImageCpu_8u_C4* src, ImageGpu_8u_C4* dst);
IU_DLLAPI void copy(const ImageCpu_32f_C1* src, ImageGpu_32f_C1* dst);
IU_DLLAPI void copy(const ImageCpu_32f_C2* src, ImageGpu_32f_C2* dst);
IU_DLLAPI void copy(const ImageCpu_32f_C3* src, ImageGpu_32f_C3* dst);
IU_DLLAPI void copy(const ImageCpu_32f_C4* src, ImageGpu_32f_C4* dst);

// 2D; copy device -> host;
/** Copy methods for device to host 2D copy
 * \param src Source image [device].
 * \param src_roi Region of interest in the source image.
 * \param dst Destination image [host]
 * \param dst_roi Region of interest in the dsetination image.
 */
IU_DLLAPI void copy(const ImageGpu_8u_C1* src, ImageCpu_8u_C1* dst);
IU_DLLAPI void copy(const ImageGpu_8u_C2* src, ImageCpu_8u_C2* dst);
IU_DLLAPI void copy(const ImageGpu_8u_C3* src, ImageCpu_8u_C3* dst);
IU_DLLAPI void copy(const ImageGpu_8u_C4* src, ImageCpu_8u_C4* dst);
IU_DLLAPI void copy(const ImageGpu_32f_C1* src, ImageCpu_32f_C1* dst);
IU_DLLAPI void copy(const ImageGpu_32f_C2* src, ImageCpu_32f_C2* dst);
IU_DLLAPI void copy(const ImageGpu_32f_C3* src, ImageCpu_32f_C3* dst);
IU_DLLAPI void copy(const ImageGpu_32f_C4* src, ImageCpu_32f_C4* dst);

/** \} */ // end of Copy2D

/* ***************************************************************************
     SET
 * ***************************************************************************/
//////////////////////////////////////////////////////////////////////////////
/** \defgroup Set1D 1D Memory Set
 * \ingroup Core
 * Set methods for 1D buffers.
 * \{
 */

/** Sets all the pixels in the given buffer to a certain value.
 * \param value The pixel value to be set.
 * \param buffer Pointer to the buffer
 */
IU_DLLAPI void setValue(const unsigned char& value, LinearHostMemory_8u_C1* srcdst);
IU_DLLAPI void setValue(const float& value, LinearHostMemory_32f_C1* srcdst);
IU_DLLAPI void setValue(const unsigned char& value, LinearDeviceMemory_8u_C1* srcdst);
IU_DLLAPI void setValue(const float& value, LinearDeviceMemory_32f_C1* srcdst);

/** \} */ // end of Set1D


//////////////////////////////////////////////////////////////////////////////
/** \defgroup Set2D 2D Memory Set
 * \ingroup Core
 * Set methods for 2D images.
 * \{
 */

//TODO this is not shown because signature does not exist. use Qt documentation system??
/** \fn void setValue(<datatype> value, Image<datatype>* srcdst, const IuRect& roi)
 * \brief Sets all pixel in the region of interest to a certain value.
 * \ingroup Set2D
 * \param value The pixel value to be set.
 * \param image Pointer to the image.
 * \param roi Region of interest which should be set.
 */
// host:
IU_DLLAPI void setValue(const unsigned char& value, ImageCpu_8u_C1* srcdst, const IuRect& roi);
IU_DLLAPI void setValue(const uchar2& value, ImageCpu_8u_C2* srcdst, const IuRect& roi);
IU_DLLAPI void setValue(const uchar3& value, ImageCpu_8u_C3* srcdst, const IuRect& roi);
IU_DLLAPI void setValue(const uchar4& value, ImageCpu_8u_C4* srcdst, const IuRect& roi);
IU_DLLAPI void setValue(const float& value, ImageCpu_32f_C1* srcdst, const IuRect& roi);
IU_DLLAPI void setValue(const float2& value, ImageCpu_32f_C2* srcdst, const IuRect& roi);
IU_DLLAPI void setValue(const float3& value, ImageCpu_32f_C3* srcdst, const IuRect& roi);
IU_DLLAPI void setValue(const float4& value, ImageCpu_32f_C4* srcdst, const IuRect& roi);
// device:
IU_DLLAPI void setValue(const unsigned char& value, ImageGpu_8u_C1* srcdst, const IuRect& roi);
IU_DLLAPI void setValue(const uchar2& value, ImageGpu_8u_C2* srcdst, const IuRect& roi);
IU_DLLAPI void setValue(const uchar3& value, ImageGpu_8u_C3* srcdst, const IuRect& roi);
IU_DLLAPI void setValue(const uchar4& value, ImageGpu_8u_C4* srcdst, const IuRect& roi);
IU_DLLAPI void setValue(const float& value, ImageGpu_32f_C1* srcdst, const IuRect& roi);
IU_DLLAPI void setValue(const float2& value, ImageGpu_32f_C2* srcdst, const IuRect& roi);
IU_DLLAPI void setValue(const float3& value, ImageGpu_32f_C3* srcdst, const IuRect& roi);
IU_DLLAPI void setValue(const float4& value, ImageGpu_32f_C4* srcdst, const IuRect& roi);

/** \} */ // end of Set2D

//////////////////////////////////////////////////////////////////////////////
/** \defgroup Clamp
 * \ingroup Core
 * Clamping methods for 2D images.
 * \{
 */

/** Clamps all values of srcdst to the interval min/max.
 * \param min Minimum value for the clamping.
 * \param max Maximum value for the clamping.
 * \param srcdst Image which pixels are clamped.
 * \param roi Region of interest of processed pixels.
 */
void clamp(const float& min, const float& max, iu::ImageGpu_32f_C1* srcdst, const IuRect& roi);

/** \} */ // end of Clamp

//////////////////////////////////////////////////////////////////////////////
/** \defgroup Conversions
 * \ingroup Core
 * Conversion methods for 2D images.
 * \{
 */

/** Converts an 32-bit 3-channel image to a 32-bit 4-channel image (adds alpha channel with value 1.0 everywhere).
 * \param src 3-channel source image [device].
 * \param src_roi Region of interest in the source image.
 * \param dst 4-channel destination image [device]
 * \param dst_roi Region of interest in the dsetination image.
 */
IU_DLLAPI void convert(const ImageGpu_32f_C3* src, const IuRect& src_roi, ImageGpu_32f_C4* dst, const IuRect& dst_roi);

/** Converts an 32-bit 4-channel image to a 32-bit 3-channel image (simply neglects the alpha channel).
 * \param src 4-channel source image [device].
 * \param src_roi Region of interest in the source image.
 * \param dst 3-channel destination image [device]
 * \param dst_roi Region of interest in the dsetination image.
 */
IU_DLLAPI void convert(const ImageGpu_32f_C4* src, const IuRect& src_roi, ImageGpu_32f_C3* dst, const IuRect& dst_roi);

/** Converts an 32-bit single-channel image to a 8-bit single-channel image.
 * \params src 1-channel source image [host].
 * \params dst 1-channel destination image [host].
 * \params mul_constant The optional scale factor.
 * \params add_constant The optional delta, added to the scaled values.
 */
IU_DLLAPI void convert_32f8u_C1(const iu::ImageCpu_32f_C1* src, iu::ImageCpu_8u_C1* dst,
                                float mul_constant=255.0f, float add_constant=0.0f);

/** Converts an 16-bit single-channel image to a 32-bit single-channel image.
 * \params src 1-channel source image [host].
 * \params dst 1-channel destination image [host].
 * \params mul_constant The optional scale factor.
 * \params add_constant The optional delta, added to the scaled values.
 */
IU_DLLAPI void convert_16u32f_C1(const iu::ImageCpu_16u_C1* src, iu::ImageCpu_32f_C1 *dst,
                                 float mul_constant, float add_constant);

/** \} */ // end of Conversions


/** \} */ // end of Core module

} // namespace iu

#endif // IU_CORE_MODULE_H
