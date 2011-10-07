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
 * Module      : IOPGM Module
 * Class       : Wrapper
 * Language    : C
 * Description : Public interfaces to IOPGM module
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUIOPGM_MODULE_H
#define IUIOPGM_MODULE_H

#include <string>
#include "iudefs.h"

namespace iu {

/** \defgroup IOPGM
 *  \brief The IOPGM module.
 *  TODO more detailed docu
 *  @{
 */

/** Loads an pgm image with 12-bit data in a 16-bit container to host memory from a file.
 * @param filename Name of file to be loaded
 * @returns loaded image in host memory (ImageCpu).
 * @note The memory is directly converted to either 16-bit or 32-bit images.
 */
IUIOPGM_DLLAPI iu::ImageCpu_16u_C1* imread_16u_C1(const std::string& filename);
IUIOPGM_DLLAPI iu::ImageCpu_32f_C1* imread_16u32f_C1(const std::string& filename, int max_val=65536);
IUIOPGM_DLLAPI iu::ImageGpu_32f_C1* imread_cu16u32f_C1(const std::string& filename, int max_val=65536);


/** @} */ // end of IOPGM

} // namespace iu

#endif // IUIOPGM_MODULE_H
