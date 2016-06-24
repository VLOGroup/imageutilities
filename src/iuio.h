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
#include <opencv2/core/core.hpp>

namespace iu {

/** \defgroup IO iuio
 * \brief Provides I/O functionality.

 * Reading/writing images, reading from cameras
 * and video files, interfaces to read/write OpenEXR files
 * \{
 */

/** \defgroup ImageIO Images
  * \ingroup IO
  * \brief Read and write images
  * \{
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


/** \}  */ // end of ImagIO

/** Construct a ImageCpu from an openCV Mat.
  * The ImageCpu is NOT a deep copy, it just uses the data pointer from the Mat, i.e.
  * it wraps the memory from the Mat in an ImageCpu.
  */
template<typename PixelType, class Allocator >
IUIO_DLLAPI void imageCpu_from_Mat(cv::Mat& mat, iu::ImageCpu<PixelType, Allocator> &img)
{
    IuSize mat_sz(mat.cols, mat.rows);

    if (img.data())
        throw IuException("Conversion from cv::Mat to iu::ImageCpu: expected empty ImageCpu", __FILE__, __FUNCTION__, __LINE__);

//    if (mat_sz != img.size())
//        throw IuException("Conversion from cv::Mat to iu::ImageCpu: size mismatch", __FILE__, __FUNCTION__, __LINE__);

//    int type = mat.type();
//    switch (type)
//    {
//    case CV_8UC1:
//        if (img.pixelType() != IU_8U_C1)
//            throw IuException("Conversion from cv::Mat to iu::ImageCpu: type mismatch", __FILE__, __FUNCTION__, __LINE__);
//        break;
//    case CV_8UC2:
//        if (img.pixelType() != IU_8U_C2)
//            throw IuException("Conversion from cv::Mat to iu::ImageCpu: type mismatch", __FILE__, __FUNCTION__, __LINE__);
//        break;
//    case CV_8UC3:
//        if (img.pixelType() != IU_8U_C3)
//            throw IuException("Conversion from cv::Mat to iu::ImageCpu: type mismatch", __FILE__, __FUNCTION__, __LINE__);
//        break;
//    case CV_8UC4:
//        if (img.pixelType() != IU_8U_C4)
//            throw IuException("Conversion from cv::Mat to iu::ImageCpu: type mismatch", __FILE__, __FUNCTION__, __LINE__);
//        break;
//    case CV_32FC1:
//        if (img.pixelType() != IU_32F_C1)
//            throw IuException("Conversion from cv::Mat to iu::ImageCpu: type mismatch", __FILE__, __FUNCTION__, __LINE__);
//        break;
//    case CV_32FC2:
//        if (img.pixelType() != IU_32F_C2)
//            throw IuException("Conversion from cv::Mat to iu::ImageCpu: type mismatch", __FILE__, __FUNCTION__, __LINE__);
//        break;
//    case CV_32FC3:
//        if (img.pixelType() != IU_32F_C3)
//            throw IuException("Conversion from cv::Mat to iu::ImageCpu: type mismatch", __FILE__, __FUNCTION__, __LINE__);
//        break;
//    case CV_32FC4:
//        if (img.pixelType() != IU_32F_C4)
//            throw IuException("Conversion from cv::Mat to iu::ImageCpu: type mismatch", __FILE__, __FUNCTION__, __LINE__);
//        break;
//    default:
//        printf("Conversion from cv::Mat to iu::ImageCpu: Error, type not implemented");
//        return;
//    }

    // The assignment operator will involve a temporary copy and destruction of img, but we don't care since we made
    // sure that the img is empty -> img destructor will not free any data
    img = iu::ImageCpu<PixelType, Allocator>((PixelType*)mat.data, mat_sz.width, mat_sz.height, mat.step, true);
}

/** \} */ // end of IO

} // namespace iu



#endif // IUIO_MODULE_H
