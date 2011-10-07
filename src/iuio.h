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
 * Module      : IO Module
 * Class       : Wrapper
 * Language    : C
 * Description : Public interfaces to IO module
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUIO_MODULE_H
#define IUIO_MODULE_H

#include <string>
#include "iudefs.h"

namespace iu {

/** \defgroup IMAGEIO
 *  \brief The image IO module.
 *  TODO more detailed docu
 *  @{
 */

/** Loads an image to host memory from a file.
 * @param filename Name of file to be loaded
 * @returns loaded image in host memory (ImageCpu).
 */
IUIO_DLLAPI iu::ImageCpu_8u_C1* imread_8u_C1(const std::string& filename);
IUIO_DLLAPI iu::ImageCpu_8u_C3* imread_8u_C3(const std::string& filename);
IUIO_DLLAPI iu::ImageCpu_8u_C4* imread_8u_C4(const std::string& filename);
IUIO_DLLAPI iu::ImageCpu_32f_C1* imread_32f_C1(const std::string& filename);
IUIO_DLLAPI iu::ImageCpu_32f_C3* imread_32f_C3(const std::string& filename);
IUIO_DLLAPI iu::ImageCpu_32f_C4* imread_32f_C4(const std::string& filename);

/** Loads an image to device memory from a file.
 * @param filename Name of file to be loaded
 * @returns loaded image in device memory (ImageGpu).
 */
IUIO_DLLAPI iu::ImageGpu_8u_C1* imread_cu8u_C1(const std::string& filename);
IUIO_DLLAPI iu::ImageGpu_8u_C4* imread_cu8u_C4(const std::string& filename);
IUIO_DLLAPI iu::ImageGpu_32f_C1* imread_cu32f_C1(const std::string& filename);
IUIO_DLLAPI iu::ImageGpu_32f_C4* imread_cu32f_C4(const std::string& filename);

/** Saves a host image to a file.
 * @param image Pointer to host image (cpu) that should be written to disk.
 * @param filename Name of file to be saved.
 * @returns Saved status.
 */
IUIO_DLLAPI bool imsave(iu::ImageCpu_8u_C1* image, const std::string& filename, const bool& normalize=false);
IUIO_DLLAPI bool imsave(iu::ImageCpu_8u_C3* image, const std::string& filename, const bool& normalize=false);
IUIO_DLLAPI bool imsave(iu::ImageCpu_8u_C4* image, const std::string& filename, const bool& normalize=false);
IUIO_DLLAPI bool imsave(iu::ImageCpu_32f_C1* image, const std::string& filename, const bool& normalize=false);
IUIO_DLLAPI bool imsave(iu::ImageCpu_32f_C3* image, const std::string& filename, const bool& normalize=false);
IUIO_DLLAPI bool imsave(iu::ImageCpu_32f_C4* image, const std::string& filename, const bool& normalize=false);

/** Saves a device image to a file.
 * @param image Pointer to device image (gpu) that should be written to disk.
 * @param filename Name of file to be saved.
 * @returns Saved status.
 */
IUIO_DLLAPI bool imsave(iu::ImageGpu_8u_C1* image, const std::string& filename, const bool& normalize=false);
IUIO_DLLAPI bool imsave(iu::ImageGpu_8u_C4* image, const std::string& filename, const bool& normalize=false);
IUIO_DLLAPI bool imsave(iu::ImageGpu_32f_C1* image, const std::string& filename, const bool& normalize=false);
IUIO_DLLAPI bool imsave(iu::ImageGpu_32f_C4* image, const std::string& filename, const bool& normalize=false);

/** Shows the host image in a window using OpenCVs imshow
 * @param image Pointer to host image (cpu) that should be shown.
 * @param winname Name of the window.
 */
IUIO_DLLAPI void imshow(iu::ImageCpu_8u_C1* image, const std::string& winname, const bool& normalize=false);
IUIO_DLLAPI void imshow(iu::ImageCpu_8u_C3* image, const std::string& winname, const bool& normalize=false);
IUIO_DLLAPI void imshow(iu::ImageCpu_8u_C4* image, const std::string& winname, const bool& normalize=false);
IUIO_DLLAPI void imshow(iu::ImageCpu_32f_C1* image, const std::string& winname, const bool& normalize=false);
IUIO_DLLAPI void imshow(iu::ImageCpu_32f_C3* image, const std::string& winname, const bool& normalize=false);
IUIO_DLLAPI void imshow(iu::ImageCpu_32f_C4* image, const std::string& winname, const bool& normalize=false);

/** Shows the device image in a host window using OpenCVs imshow
 * @param image Pointer to device image (gpu) that should be shown.
 * @param winname Name of the window.
 */
IUIO_DLLAPI void imshow(iu::ImageGpu_8u_C1* image, const std::string& winname, const bool& normalize=false);
IUIO_DLLAPI void imshow(iu::ImageGpu_8u_C4* image, const std::string& winname, const bool& normalize=false);
IUIO_DLLAPI void imshow(iu::ImageGpu_32f_C1* image, const std::string& winname, const bool& normalize=false);
IUIO_DLLAPI void imshow(iu::ImageGpu_32f_C4* image, const std::string& winname, const bool& normalize=false);


/** @} */ // end of IMAGEIO

} // namespace iu


/** \defgroup VIDEOIO
 *  \brief The video IO module.
 *  TODO more detailed docu
 *  @{
 */

// VideoCapture
#include "iuio/videocapture.h"

/** @} */ // end of IMAGEIO


#endif // IUIO_MODULE_H
