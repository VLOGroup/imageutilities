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

/** \defgroup IO
 *  \brief The IO module.
 *  TODO more detailed docu
 *  @{
 */

/** Loads an image to host memory from a file.
 * @param filename Name of file to be loaded
 * @returns loaded image in host memory (ImageCpu).
 */
IU_DLLAPI iu::ImageCpu_32f_C1* imread_32f_C1(const std::string& filename);

/** Loads an image to device memory from a file.
 * @param filename Name of file to be loaded
 * @returns loaded image in device memory (ImageNpp).
 */
IU_DLLAPI iu::ImageNpp_32f_C1* imread_cu32f_C1(const std::string& filename);

/** Saves a host image to a file.
 * @param image Pointer to host image (cpu) that should be written to disk.
 * @param filename Name of file to be saved.
 * @returns Saved status.
 */
IU_DLLAPI bool imsave(iu::ImageCpu_32f_C1* image, const std::string& filename);
//IU_DLLAPI bool imsave(iu::ImageCpu_32f_C4* image, const std::string& filename);

/** Saves a device image to a file.
 * @param image Pointer to device image (gpu) that should be written to disk.
 * @param filename Name of file to be saved.
 * @returns Saved status.
 */
IU_DLLAPI bool imsave(iu::ImageNpp_32f_C1* image, const std::string& filename);
//IU_DLLAPI bool imsave(iu::ImageNpp_32f_C4* image, const std::string& filename);


/** @} */ // end of IO

} // namespace iu

#endif // IUIO_MODULE_H
