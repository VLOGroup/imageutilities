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



class IuCudaTimer
{
public:
  IuCudaTimer() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }

  ~IuCudaTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start() {
    cudaEventRecord(start_, 0);
  }

  float elapsed() {
    cudaEventRecord(stop_);
    cudaEventSynchronize(stop_);
    float t = 0;
    cudaEventElapsedTime(&t, start_, stop_);
    return t;
  }

private:
  cudaEvent_t start_;
  cudaEvent_t stop_;
};


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
IUCORE_DLLAPI void copy(const LinearHostMemory_8u_C1* src, LinearHostMemory_8u_C1* dst);
IUCORE_DLLAPI void copy(const LinearHostMemory_16u_C1* src, LinearHostMemory_16u_C1* dst);
IUCORE_DLLAPI void copy(const LinearHostMemory_32f_C1* src, LinearHostMemory_32f_C1* dst);

// copy device -> device;
/** Copy methods for host to host 1D copy methods for 8bit buffers.
 * \param src Source buffer [host]
 * \param dst Destination buffer [host]
 */
IUCORE_DLLAPI void copy(const LinearDeviceMemory_8u_C1* src, LinearDeviceMemory_8u_C1* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_8u_C2* src, LinearDeviceMemory_8u_C2* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_8u_C3* src, LinearDeviceMemory_8u_C3* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_8u_C4* src, LinearDeviceMemory_8u_C4* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_16u_C1* src, LinearDeviceMemory_16u_C1* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_16u_C2* src, LinearDeviceMemory_16u_C2* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_16u_C3* src, LinearDeviceMemory_16u_C3* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_16u_C4* src, LinearDeviceMemory_16u_C4* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_32s_C1* src, LinearDeviceMemory_32s_C1* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_32s_C2* src, LinearDeviceMemory_32s_C2* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_32s_C3* src, LinearDeviceMemory_32s_C3* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_32s_C4* src, LinearDeviceMemory_32s_C4* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_32u_C1* src, LinearDeviceMemory_32u_C1* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_32u_C2* src, LinearDeviceMemory_32u_C2* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_32u_C4* src, LinearDeviceMemory_32u_C4* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_32f_C1* src, LinearDeviceMemory_32f_C1* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_32f_C2* src, LinearDeviceMemory_32f_C2* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_32f_C3* src, LinearDeviceMemory_32f_C3* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_32f_C4* src, LinearDeviceMemory_32f_C4* dst);

// copy host -> device;
/** Copy methods for host to host 1D copy methods for 8bit buffers.
 * \param src Source buffer [host]
 * \param dst Destination buffer [device]
 */
IUCORE_DLLAPI void copy(const LinearHostMemory_8u_C1* src, LinearDeviceMemory_8u_C1* dst);
IUCORE_DLLAPI void copy(const LinearHostMemory_8u_C2* src, LinearDeviceMemory_8u_C2* dst);
IUCORE_DLLAPI void copy(const LinearHostMemory_8u_C3* src, LinearDeviceMemory_8u_C3* dst);
IUCORE_DLLAPI void copy(const LinearHostMemory_8u_C4* src, LinearDeviceMemory_8u_C4* dst);
IUCORE_DLLAPI void copy(const LinearHostMemory_16u_C1* src, LinearDeviceMemory_16u_C1* dst);
IUCORE_DLLAPI void copy(const LinearHostMemory_16u_C2* src, LinearDeviceMemory_16u_C2* dst);
IUCORE_DLLAPI void copy(const LinearHostMemory_16u_C3* src, LinearDeviceMemory_16u_C3* dst);
IUCORE_DLLAPI void copy(const LinearHostMemory_16u_C4* src, LinearDeviceMemory_16u_C4* dst);
IUCORE_DLLAPI void copy(const LinearHostMemory_32s_C1* src, LinearDeviceMemory_32s_C1* dst);
IUCORE_DLLAPI void copy(const LinearHostMemory_32s_C2* src, LinearDeviceMemory_32s_C2* dst);
IUCORE_DLLAPI void copy(const LinearHostMemory_32s_C3* src, LinearDeviceMemory_32s_C3* dst);
IUCORE_DLLAPI void copy(const LinearHostMemory_32s_C4* src, LinearDeviceMemory_32s_C4* dst);
IUCORE_DLLAPI void copy(const LinearHostMemory_32u_C1* src, LinearDeviceMemory_32u_C1* dst);
IUCORE_DLLAPI void copy(const LinearHostMemory_32u_C2* src, LinearDeviceMemory_32u_C2* dst);
IUCORE_DLLAPI void copy(const LinearHostMemory_32u_C4* src, LinearDeviceMemory_32u_C4* dst);
IUCORE_DLLAPI void copy(const LinearHostMemory_32f_C1* src, LinearDeviceMemory_32f_C1* dst);
IUCORE_DLLAPI void copy(const LinearHostMemory_32f_C2* src, LinearDeviceMemory_32f_C2* dst);
IUCORE_DLLAPI void copy(const LinearHostMemory_32f_C3* src, LinearDeviceMemory_32f_C3* dst);
IUCORE_DLLAPI void copy(const LinearHostMemory_32f_C4* src, LinearDeviceMemory_32f_C4* dst);

// copy device -> host;
/** Copy methods for host to host 1D copy methods for 8bit buffers.
 * \param src Source buffer [device]
 * \param dst Destination buffer [host]
 */
IUCORE_DLLAPI void copy(const LinearDeviceMemory_8u_C1* src, LinearHostMemory_8u_C1* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_8u_C2* src, LinearHostMemory_8u_C2* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_8u_C3* src, LinearHostMemory_8u_C3* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_8u_C4* src, LinearHostMemory_8u_C4* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_16u_C1* src, LinearHostMemory_16u_C1* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_16u_C2* src, LinearHostMemory_16u_C2* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_16u_C3* src, LinearHostMemory_16u_C3* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_16u_C4* src, LinearHostMemory_16u_C4* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_32s_C1* src, LinearHostMemory_32s_C1* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_32s_C2* src, LinearHostMemory_32s_C2* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_32s_C3* src, LinearHostMemory_32s_C3* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_32s_C4* src, LinearHostMemory_32s_C4* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_32u_C1* src, LinearHostMemory_32u_C1* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_32u_C2* src, LinearHostMemory_32u_C2* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_32u_C4* src, LinearHostMemory_32u_C4* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_32f_C1* src, LinearHostMemory_32f_C1* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_32f_C2* src, LinearHostMemory_32f_C2* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_32f_C3* src, LinearHostMemory_32f_C3* dst);
IUCORE_DLLAPI void copy(const LinearDeviceMemory_32f_C4* src, LinearHostMemory_32f_C4* dst);

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
 * \param dst Destination image [host]
 */
IUCORE_DLLAPI void copy(const ImageCpu_8u_C1* src, ImageCpu_8u_C1* dst);
IUCORE_DLLAPI void copy(const ImageCpu_8u_C2* src, ImageCpu_8u_C2* dst);
IUCORE_DLLAPI void copy(const ImageCpu_8u_C3* src, ImageCpu_8u_C3* dst);
IUCORE_DLLAPI void copy(const ImageCpu_8u_C4* src, ImageCpu_8u_C4* dst);
IUCORE_DLLAPI void copy(const ImageCpu_32s_C1* src, ImageCpu_32s_C1* dst);
IUCORE_DLLAPI void copy(const ImageCpu_32f_C1* src, ImageCpu_32f_C1* dst);
IUCORE_DLLAPI void copy(const ImageCpu_32f_C2* src, ImageCpu_32f_C2* dst);
IUCORE_DLLAPI void copy(const ImageCpu_32f_C3* src, ImageCpu_32f_C3* dst);
IUCORE_DLLAPI void copy(const ImageCpu_32f_C4* src, ImageCpu_32f_C4* dst);

// 2D; copy device -> device;
/** Copy methods for device to device 2D copy
 * \param src Source image [device].
 * \param dst Destination image [device]
 */
IUCORE_DLLAPI void copy(const ImageGpu_8u_C1* src, ImageGpu_8u_C1* dst);
IUCORE_DLLAPI void copy(const ImageGpu_8u_C2* src, ImageGpu_8u_C2* dst);
IUCORE_DLLAPI void copy(const ImageGpu_8u_C3* src, ImageGpu_8u_C3* dst);
IUCORE_DLLAPI void copy(const ImageGpu_8u_C4* src, ImageGpu_8u_C4* dst);
IUCORE_DLLAPI void copy(const ImageGpu_32s_C1* src, ImageGpu_32s_C1* dst);
IUCORE_DLLAPI void copy(const ImageGpu_32f_C1* src, ImageGpu_32f_C1* dst);
IUCORE_DLLAPI void copy(const ImageGpu_32f_C2* src, ImageGpu_32f_C2* dst);
IUCORE_DLLAPI void copy(const ImageGpu_32f_C3* src, ImageGpu_32f_C3* dst);
IUCORE_DLLAPI void copy(const ImageGpu_32f_C4* src, ImageGpu_32f_C4* dst);

// 2D; copy host -> device;
/** Copy methods for host to device 2D copy
 * \param src Source image [host].
 * \param dst Destination image [device]
 */
IUCORE_DLLAPI void copy(const ImageCpu_8u_C1* src, ImageGpu_8u_C1* dst);
IUCORE_DLLAPI void copy(const ImageCpu_8u_C2* src, ImageGpu_8u_C2* dst);
IUCORE_DLLAPI void copy(const ImageCpu_8u_C3* src, ImageGpu_8u_C3* dst);
IUCORE_DLLAPI void copy(const ImageCpu_8u_C4* src, ImageGpu_8u_C4* dst);
IUCORE_DLLAPI void copy(const ImageCpu_32s_C1* src, ImageGpu_32s_C1* dst);
IUCORE_DLLAPI void copy(const ImageCpu_32s_C2* src, ImageGpu_32s_C2* dst);
IUCORE_DLLAPI void copy(const ImageCpu_32s_C4* src, ImageGpu_32s_C4* dst);
IUCORE_DLLAPI void copy(const ImageCpu_32u_C1* src, ImageGpu_32u_C1* dst);
IUCORE_DLLAPI void copy(const ImageCpu_32u_C2* src, ImageGpu_32u_C2* dst);
IUCORE_DLLAPI void copy(const ImageCpu_32u_C4* src, ImageGpu_32u_C4* dst);
IUCORE_DLLAPI void copy(const ImageCpu_32f_C1* src, ImageGpu_32f_C1* dst);
IUCORE_DLLAPI void copy(const ImageCpu_32f_C2* src, ImageGpu_32f_C2* dst);
IUCORE_DLLAPI void copy(const ImageCpu_32f_C3* src, ImageGpu_32f_C3* dst);
IUCORE_DLLAPI void copy(const ImageCpu_32f_C4* src, ImageGpu_32f_C4* dst);

// 2D; copy device -> host;
/** Copy methods for device to host 2D copy
 * \param src Source image [device].
 * \param dst Destination image [host]
 */
IUCORE_DLLAPI void copy(const ImageGpu_8u_C1* src, ImageCpu_8u_C1* dst);
IUCORE_DLLAPI void copy(const ImageGpu_8u_C2* src, ImageCpu_8u_C2* dst);
IUCORE_DLLAPI void copy(const ImageGpu_8u_C3* src, ImageCpu_8u_C3* dst);
IUCORE_DLLAPI void copy(const ImageGpu_8u_C4* src, ImageCpu_8u_C4* dst);
IUCORE_DLLAPI void copy(const ImageGpu_32s_C1* src, ImageCpu_32s_C1* dst);
IUCORE_DLLAPI void copy(const ImageGpu_32s_C2* src, ImageCpu_32s_C2* dst);
IUCORE_DLLAPI void copy(const ImageGpu_32s_C4* src, ImageCpu_32s_C4* dst);
IUCORE_DLLAPI void copy(const ImageGpu_32u_C1* src, ImageCpu_32u_C1* dst);
IUCORE_DLLAPI void copy(const ImageGpu_32u_C2* src, ImageCpu_32u_C2* dst);
IUCORE_DLLAPI void copy(const ImageGpu_32u_C4* src, ImageCpu_32u_C4* dst);
IUCORE_DLLAPI void copy(const ImageGpu_32f_C1* src, ImageCpu_32f_C1* dst);
IUCORE_DLLAPI void copy(const ImageGpu_32f_C2* src, ImageCpu_32f_C2* dst);
IUCORE_DLLAPI void copy(const ImageGpu_32f_C3* src, ImageCpu_32f_C3* dst);
IUCORE_DLLAPI void copy(const ImageGpu_32f_C4* src, ImageCpu_32f_C4* dst);

/** \} */ // end of Copy2D

// 3D; copy host -> host;
/** Copy methods for host to host 3D copy
 * \param src Source volume [host].
 * \param dst Destination volume [host]
 */
IUCORE_DLLAPI void copy(const VolumeCpu_8u_C1* src, VolumeCpu_8u_C1* dst);
IUCORE_DLLAPI void copy(const VolumeCpu_8u_C2* src, VolumeCpu_8u_C2* dst);
IUCORE_DLLAPI void copy(const VolumeCpu_8u_C4* src, VolumeCpu_8u_C4* dst);
IUCORE_DLLAPI void copy(const VolumeCpu_16u_C1* src, VolumeCpu_16u_C1* dst);
IUCORE_DLLAPI void copy(const VolumeCpu_32f_C1* src, VolumeCpu_32f_C1* dst);
IUCORE_DLLAPI void copy(const VolumeCpu_32f_C2* src, VolumeCpu_32f_C2* dst);
IUCORE_DLLAPI void copy(const VolumeCpu_32f_C3* src, VolumeCpu_32f_C3* dst);
IUCORE_DLLAPI void copy(const VolumeCpu_32f_C4* src, VolumeCpu_32f_C4* dst);
IUCORE_DLLAPI void copy(const VolumeCpu_32u_C1* src, VolumeCpu_32u_C1* dst);
IUCORE_DLLAPI void copy(const VolumeCpu_32u_C2* src, VolumeCpu_32u_C2* dst);
IUCORE_DLLAPI void copy(const VolumeCpu_32u_C4* src, VolumeCpu_32u_C4* dst);

IUCORE_DLLAPI void copy(const VolumeCpu_32s_C1* src, VolumeCpu_32s_C1* dst);
IUCORE_DLLAPI void copy(const VolumeCpu_32s_C2* src, VolumeCpu_32s_C2* dst);
IUCORE_DLLAPI void copy(const VolumeCpu_32s_C4* src, VolumeCpu_32s_C4* dst);


// 3D; copy device -> device;
/** Copy methods for device to device 3D copy
 * \param src Source volume [device].
 * \param dst Destination volume [device]
 */
IUCORE_DLLAPI void copy(const VolumeGpu_8u_C1* src, VolumeGpu_8u_C1* dst);
IUCORE_DLLAPI void copy(const VolumeGpu_8u_C2* src, VolumeGpu_8u_C2* dst);
IUCORE_DLLAPI void copy(const VolumeGpu_8u_C4* src, VolumeGpu_8u_C4* dst);
IUCORE_DLLAPI void copy(const VolumeGpu_16u_C1* src, VolumeGpu_16u_C1* dst);
IUCORE_DLLAPI void copy(const VolumeGpu_32f_C1* src, VolumeGpu_32f_C1* dst);
IUCORE_DLLAPI void copy(const VolumeGpu_32f_C2* src, VolumeGpu_32f_C2* dst);
IUCORE_DLLAPI void copy(const VolumeGpu_32f_C4* src, VolumeGpu_32f_C4* dst);
IUCORE_DLLAPI void copy(const VolumeGpu_32u_C1* src, VolumeGpu_32u_C1* dst);
IUCORE_DLLAPI void copy(const VolumeGpu_32u_C2* src, VolumeGpu_32u_C2* dst);
IUCORE_DLLAPI void copy(const VolumeGpu_32u_C4* src, VolumeGpu_32u_C4* dst);

IUCORE_DLLAPI void copy(const VolumeGpu_32s_C1* src, VolumeGpu_32s_C1* dst);
IUCORE_DLLAPI void copy(const VolumeGpu_32s_C2* src, VolumeGpu_32s_C2* dst);
IUCORE_DLLAPI void copy(const VolumeGpu_32s_C4* src, VolumeGpu_32s_C4* dst);


// 3D; copy host -> device;
/** Copy methods for host to device 3D copy
 * \param src Source volume [host].
 * \param dst Destination volume [device]
 */
IUCORE_DLLAPI void copy(const VolumeCpu_8u_C1* src, VolumeGpu_8u_C1* dst);
IUCORE_DLLAPI void copy(const VolumeCpu_8u_C2* src, VolumeGpu_8u_C2* dst);
IUCORE_DLLAPI void copy(const VolumeCpu_8u_C4* src, VolumeGpu_8u_C4* dst);
IUCORE_DLLAPI void copy(const VolumeCpu_16u_C1* src, VolumeGpu_16u_C1* dst);
IUCORE_DLLAPI void copy(const VolumeCpu_32f_C1* src, VolumeGpu_32f_C1* dst);
IUCORE_DLLAPI void copy(const VolumeCpu_32f_C2* src, VolumeGpu_32f_C2* dst);
IUCORE_DLLAPI void copy(const VolumeCpu_32f_C3* src, VolumeGpu_32f_C3* dst);
IUCORE_DLLAPI void copy(const VolumeCpu_32f_C4* src, VolumeGpu_32f_C4* dst);
IUCORE_DLLAPI void copy(const VolumeCpu_32u_C1* src, VolumeGpu_32u_C1* dst);
IUCORE_DLLAPI void copy(const VolumeCpu_32u_C2* src, VolumeGpu_32u_C2* dst);
IUCORE_DLLAPI void copy(const VolumeCpu_32u_C4* src, VolumeGpu_32u_C4* dst);

IUCORE_DLLAPI void copy(const VolumeCpu_32s_C1* src, VolumeGpu_32s_C1* dst);
IUCORE_DLLAPI void copy(const VolumeCpu_32s_C2* src, VolumeGpu_32s_C2* dst);
IUCORE_DLLAPI void copy(const VolumeCpu_32s_C4* src, VolumeGpu_32s_C4* dst);

// 3D; copy device -> host;
/** Copy methods for device to host 3D copy
 * \param src Source volume [device].
 * \param dst Destination volume [host]
 */
IUCORE_DLLAPI void copy(const VolumeGpu_8u_C1* src, VolumeCpu_8u_C1* dst);
IUCORE_DLLAPI void copy(const VolumeGpu_8u_C2* src, VolumeCpu_8u_C2* dst);
IUCORE_DLLAPI void copy(const VolumeGpu_8u_C4* src, VolumeCpu_8u_C4* dst);
IUCORE_DLLAPI void copy(const VolumeGpu_16u_C1* src, VolumeCpu_16u_C1* dst);
IUCORE_DLLAPI void copy(const VolumeGpu_32f_C1* src, VolumeCpu_32f_C1* dst);
IUCORE_DLLAPI void copy(const VolumeGpu_32f_C2* src, VolumeCpu_32f_C2* dst);
IUCORE_DLLAPI void copy(const VolumeGpu_32f_C3* src, VolumeCpu_32f_C3* dst);
IUCORE_DLLAPI void copy(const VolumeGpu_32f_C4* src, VolumeCpu_32f_C4* dst);
IUCORE_DLLAPI void copy(const VolumeGpu_32u_C1* src, VolumeCpu_32u_C1* dst);
IUCORE_DLLAPI void copy(const VolumeGpu_32u_C2* src, VolumeCpu_32u_C2* dst);
IUCORE_DLLAPI void copy(const VolumeGpu_32u_C4* src, VolumeCpu_32u_C4* dst);

IUCORE_DLLAPI void copy(const VolumeGpu_32s_C1* src, VolumeCpu_32s_C1* dst);
IUCORE_DLLAPI void copy(const VolumeGpu_32s_C2* src, VolumeCpu_32s_C2* dst);
IUCORE_DLLAPI void copy(const VolumeGpu_32s_C4* src, VolumeCpu_32s_C4* dst);


// convert to linear (contiguous) memory
IUCORE_DLLAPI void copy(const ImageCpu_8u_C1* src, LinearHostMemory_8u_C1* dst);
IUCORE_DLLAPI void copy(const ImageCpu_16u_C1* src, LinearHostMemory_16u_C1* dst);
IUCORE_DLLAPI void copy(const ImageCpu_32f_C1* src, LinearHostMemory_32f_C1* dst);

IUCORE_DLLAPI void copy(const ImageGpu_32f_C1* src, LinearDeviceMemory_32f_C1* dst);

/** \} */ // end of Copy3D


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
IUCORE_DLLAPI void convert(const ImageGpu_32f_C3* src, ImageGpu_32f_C4* dst);

/** Converts an 32-bit 4-channel image to a 32-bit 3-channel image (simply neglects the alpha channel).
 * \param src 4-channel source image [device].
 * \param src_roi Region of interest in the source image.
 * \param dst 3-channel destination image [device]
 * \param dst_roi Region of interest in the dsetination image.
 */
IUCORE_DLLAPI void convert(const ImageGpu_32f_C4* src, ImageGpu_32f_C3* dst);

/** Converts an 32-bit single-channel image to a 8-bit single-channel image.
 * \param src 1-channel source image [host].
 * \param dst 1-channel destination image [host].
 * \param mul_constant The optional scale factor.
 * \param add_constant The optional delta, added to the scaled values.
 */
IUCORE_DLLAPI void convert_32f8u_C1(const iu::ImageCpu_32f_C1* src, iu::ImageCpu_8u_C1* dst,
                                float mul_constant=255.0f, float add_constant=0.0f);

/** Converts an 32-bit single-channel image to a 8-bit single-channel image.
 * \param src 1-channel source image [device].
 * \param dst 1-channel destination image [device].
 * \param mul_constant The optional scale factor.
 * \param add_constant The optional delta, added to the scaled values.
 */
IUCORE_DLLAPI void convert_32f8u_C1(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_8u_C1* dst,
                                float mul_constant=255.0f, unsigned char add_constant=0);

/** Converts an 32-bit 4-channel image to a 8-bit 4-channel image.
 * \param src 4-channel source image [device].
 * \param dst 4-channel destination image [device].
 * \param mul_constant The optional scale factor.
 * \param add_constant The optional delta, added to the scaled values.
 */
IUCORE_DLLAPI void convert_32f8u_C4(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_8u_C4* dst,
                                float mul_constant=255.0f, unsigned char add_constant=0);

/** Converts an 8-bit single-channel image to a 32-bit single-channel image.
 * \param src 1-channel source image [device].
 * \param dst 1-channel destination image [device].
 * \param mul_constant The optional scale factor.
 * \param add_constant The optional delta, added to the scaled values.
 */
IUCORE_DLLAPI void convert_8u32f_C1(const iu::ImageGpu_8u_C1* src, iu::ImageGpu_32f_C1* dst,
                                float mul_constant=1/255.0f, float add_constant=0.0f);

/** Converts a 32-bit single-channel uint image to a 32-bit single-channel float image.
 * \param src 1-channel source image [device].
 * \param dst 1-channel destination image [device].
 * \param mul_constant The optional scale factor.
 * \param add_constant The optional delta, added to the scaled values.
 */
IUCORE_DLLAPI void convert_32u32f_C1(const iu::ImageGpu_32u_C1* src, iu::ImageGpu_32f_C1* dst,
                                float mul_constant, float add_constant=0.0f);


/** Converts a 32-bit single-channel uint image to a 32-bit single-channel float image.
 * \param src 1-channel source image [host].
 * \param dst 1-channel destination image [host].
 * \param mul_constant The optional scale factor.
 * \param add_constant The optional delta, added to the scaled values.
 */
IUCORE_DLLAPI void convert_32u32f_C1(const iu::ImageCpu_32u_C1* src, iu::ImageCpu_32f_C1* dst,
                                float mul_constant, float add_constant=0.0f);

/** Converts a 32-bit single-channel linear memory to a 32-bit single-channel linear memory.
 * Inplace conversion is possible.
 * \param src 1-channel src linear memory [device].
 * \param dst 1-channel destination linear memory [device].
 */
IUCORE_DLLAPI void convert_32s32f_C1_lin(iu::LinearDeviceMemory_32s_C1* src, iu::LinearDeviceMemory_32f_C1* dst);


/** Converts an 8-bit 3-channel image to a 32-bit 4-channel image.
 * \param src 3-channel source image [device].
 * \param dst 4-channel destination image [device].
 * \param mul_constant The optional scale factor.
 * \param add_constant The optional delta, added to the scaled values.
 */
IUCORE_DLLAPI void convert_8u32f_C3C4(const iu::ImageGpu_8u_C3* src, iu::ImageGpu_32f_C4* dst,
                                float mul_constant=1/255.0f, float add_constant=0.0f);


/** Converts an 16-bit single-channel image to a 32-bit single-channel image.
 * \param src 1-channel source image [host].
 * \param dst 1-channel destination image [host].
 * \param mul_constant The optional scale factor.
 * \param add_constant The optional delta, added to the scaled values.
 */
IUCORE_DLLAPI void convert_16u32f_C1(const iu::ImageCpu_16u_C1* src, iu::ImageCpu_32f_C1 *dst,
                                 float mul_constant, float add_constant);


/** Converts an RGB image to a HSV image.
 * \param src 4-channel source image [device].
 * \param dst 4-channel destination image [device].
 * \param normalize Normalizes all channels to [0, 1]
 */
IUCORE_DLLAPI void convert_RgbHsv(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst, bool normalize=false);

/** Converts a HSV image to an RGB image.
 * \param src 4-channel source image [device].
 * \param dst 4-channel destination image [device].
 * \param normalize Normalizes all channels to [0, 1]
 */
IUCORE_DLLAPI void convert_HsvRgb(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst, bool denormalize=false);

/** Converts a RGB image to a CIELAB image using D65 (=6500K) white point normalization.
 * \param src 4-channel source image [device].
 * \param dst 4-channel destination image [device].
 * \param isNormalized flag indicating whether the values of src are normalized to [0, 1] or not. Not normalized images are assumed to have values in [0, 255].
 */
IUCORE_DLLAPI void convert_RgbLab(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst, bool isNormalized=true);


/** Converts a CIELAB image to a RGB image assuming D65 (=6500K) white point.
 * \param src 4-channel source image [device].
 * \param dst 4-channel destination image [device].
 */
IUCORE_DLLAPI void convert_LabRgb(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst);


/** \} */ // end of Conversions

//IUCORE_DLLAPI double summation(iu::ImageGpu_32f_C1* src);


/** \} */ // end of Core module

} // namespace iu

#endif // IU_CORE_MODULE_H
