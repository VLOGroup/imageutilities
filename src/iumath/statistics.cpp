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
 * Module      : Math
 * Class       : none
 * Language    : C++
 * Description : Implementation of statistics functions
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#include "statistics.cuh"
#include "statistics.h"

namespace iuprivate {

/*
  MIN/MAX
*/

///////////////////////////////////////////////////////////////////////////////

//// [host] find min/max value of image; 8-bit; 1-channel
//void minMax(const iu::ImageCpu_8u_C1 *src, const IuRect &roi, unsigned char& min, unsigned char& max)
//{
//  IppStatus status;
//  IppiSize roi_size = {roi.width, roi.height};
//  status = ippiMinMax_8u_C1R(src->data(roi.x, roi.y), static_cast<int>(src->pitch()), roi_size, &min, &max);
//  IU_ASSERT(status == ippStsNoErr);
//}

//// [host] find min/max value of image; 8-bit; 3-channel
//void minMax(const iu::ImageCpu_8u_C3 *src, const IuRect &roi, unsigned char min[3], unsigned char max[3])
//{
//  IppStatus status;
//  IppiSize roi_size = {roi.width, roi.height};
//  status = ippiMinMax_8u_C3R(src->data(roi.x, roi.y), static_cast<int>(src->pitch()), roi_size, min, max);
//  IU_ASSERT(status == ippStsNoErr);
//}

//// [host] find min/max value of image; 8-bit; 4-channel
//void minMax(const iu::ImageCpu_8u_C4 *src, const IuRect &roi, unsigned char min[4], unsigned char max[4])
//{
//  IppStatus status;
//  IppiSize roi_size = {roi.width, roi.height};
//  status = ippiMinMax_8u_C4R(src->data(roi.x, roi.y), static_cast<int>(src->pitch()), roi_size, min, max);
//  IU_ASSERT(status == ippStsNoErr);
//}

/////////////////////////////////////////////////////////////////////////////////

//// [host] find min/max value of image; 32-bit; 1-channel
//void minMax(const iu::ImageCpu_32f_C1 *src, const IuRect &roi, float& min, float& max)
//{
//  IppStatus status;
//  IppiSize roi_size = {roi.width, roi.height};
//  status = ippiMinMax_32f_C1R(src->data(roi.x, roi.y), static_cast<int>(src->pitch()), roi_size, &min, &max);
//  IU_ASSERT(status == ippStsNoErr);
//}

//// [host] find min/max value of image; 32-bit; 3-channel
//void minMax(const iu::ImageCpu_32f_C3 *src, const IuRect &roi, float min[3], float max[3])
//{
//  IppStatus status;
//  IppiSize roi_size = {roi.width, roi.height};
//  status = ippiMinMax_32f_C3R(src->data(roi.x, roi.y), static_cast<int>(src->pitch()), roi_size, min, max);
//  IU_ASSERT(status == ippStsNoErr);
//}

//// [host] find min/max value of image; 32-bit; 4-channel
//void minMax(const iu::ImageCpu_32f_C4 *src, const IuRect &roi, float min[4], float max[4])
//{
//  IppStatus status;
//  IppiSize roi_size = {roi.width, roi.height};
//  status = ippiMinMax_32f_C4R(src->data(roi.x, roi.y), static_cast<int>(src->pitch()), roi_size, min, max);
//  IU_ASSERT(status == ippStsNoErr);
//}

///////////////////////////////////////////////////////////////////////////////

// [device] find min/max value of image; 8-bit; 1-channel
void minMax(const iu::ImageGpu_8u_C1 *src, const IuRect &roi, unsigned char& min, unsigned char& max)
{
  IuStatus status;
  printf("wrapper\n");
  status = cuMinMax(src, roi, min, max);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}

// [device] find min/max value of image; 8-bit; 4-channel
void minMax(const iu::ImageGpu_8u_C4 *src, const IuRect &roi, uchar4& min, uchar4& max)
{
  IuStatus status;
  status = cuMinMax(src, roi, min, max);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}

///////////////////////////////////////////////////////////////////////////////

// [device] find min/max value of image; 32-bit; 1-channel
void minMax(const iu::ImageGpu_32f_C1 *src, const IuRect &roi, float& min, float& max)
{
  IuStatus status;
  status = cuMinMax(src, roi, min, max);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}

// [device] find min/max value of image; 32-bit; 2-channel
void minMax(const iu::ImageGpu_32f_C2 *src, const IuRect &roi, float2& min, float2& max)
{
  IuStatus status;
  status = cuMinMax(src, roi, min, max);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}

// [device] find min/max value of image; 32-bit; 4-channel
void minMax(const iu::ImageGpu_32f_C4 *src, const IuRect &roi, float4& min, float4& max)
{
  IuStatus status;
  status = cuMinMax(src, roi, min, max);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}

///////////////////////////////////////////////////////////////////////////////

// [device] find min/max value of volume; 32-bit; 1-channel
void minMax(iu::VolumeGpu_32f_C1 *src, float& min, float& max)
{
  IuStatus status;
  status = cuMinMax(src, min, max);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}


///////////////////////////////////////////////////////////////////////////////

// [device] find min value and its coordinates of image; 32-bit; 1-channel
void min(const iu::ImageGpu_32f_C1* src, const IuRect&roi, float& min, int& x, int& y)
{
  IuStatus status;
  status = cuMin(src, roi, min, x, y);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}

// [device] find max value and its coordinates of image; 32-bit; 1-channel
void max(const iu::ImageGpu_32f_C1* src, const IuRect&roi, float& max, int& x, int& y)
{
  IuStatus status;
  status = cuMax(src, roi, max, x, y);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}


/* ****************************************************************************
  SUM
*/

///////////////////////////////////////////////////////////////////////////////

//// [host] compute sum of image; 8-bit; 1-channel
//void summation(const iu::ImageCpu_8u_C1 *src, const IuRect &roi, Ipp64f& sum)
//{
//  IppStatus status;
//  IppiSize roi_size = {roi.width, roi.height};
//  status = ippiSum_8u_C1R(src->data(roi.x, roi.y), static_cast<int>(src->pitch()), roi_size, &sum);
//  IU_ASSERT(status == ippStsNoErr);
//}

//// [host] compute sum of image; 8-bit; 3-channel
//void summation(const iu::ImageCpu_8u_C3 *src, const IuRect &roi, Ipp64f sum[3])
//{
//  IppStatus status;
//  IppiSize roi_size = {roi.width, roi.height};
//  status = ippiSum_8u_C3R(src->data(roi.x, roi.y), static_cast<int>(src->pitch()), roi_size, sum);
//  IU_ASSERT(status == ippStsNoErr);
//}

//// [host] compute sum of image; 8-bit; 4-channel
//void summation(const iu::ImageCpu_8u_C4 *src, const IuRect &roi, Ipp64f sum[4])
//{
//  IppStatus status;
//  IppiSize roi_size = {roi.width, roi.height};
//  status = ippiSum_8u_C4R(src->data(roi.x, roi.y), static_cast<int>(src->pitch()), roi_size, sum);
//  IU_ASSERT(status == ippStsNoErr);
//}

/////////////////////////////////////////////////////////////////////////////////

//// [host] compute sum of image; 32-bit; 1-channel
//void summation(const iu::ImageCpu_32f_C1 *src, const IuRect &roi, Ipp64f& sum)
//{
//  IppStatus status;
//  IppiSize roi_size = {roi.width, roi.height};
//  status = ippiSum_32f_C1R(src->data(roi.x, roi.y), static_cast<int>(src->pitch()), roi_size, &sum, ippAlgHintNone);
//  IU_ASSERT(status == ippStsNoErr);
//}

//// [host] compute sum of image; 32-bit; 3-channel
//void summation(const iu::ImageCpu_32f_C3 *src, const IuRect &roi, Ipp64f sum[3])
//{
//  IppStatus status;
//  IppiSize roi_size = {roi.width, roi.height};
//  status = ippiSum_32f_C3R(src->data(roi.x, roi.y), static_cast<int>(src->pitch()), roi_size, sum, ippAlgHintNone);
//  IU_ASSERT(status == ippStsNoErr);
//}

//// [host] compute sum of image; 32-bit; 4-channel
//void summation(const iu::ImageCpu_32f_C4 *src, const IuRect &roi, Ipp64f sum[4])
//{
//  IppStatus status;
//  IppiSize roi_size = {roi.width, roi.height};
//  status = ippiSum_32f_C4R(src->data(roi.x, roi.y), static_cast<int>(src->pitch()), roi_size, sum, ippAlgHintNone);
//  IU_ASSERT(status == ippStsNoErr);
//}

///////////////////////////////////////////////////////////////////////////////

// [device] compute sum of image; 8-bit; 1-channel
void summation(const iu::ImageGpu_8u_C1 *src, const IuRect &roi, long& sum)
{
  IuStatus status;
  status = cuSummation(src, roi, sum);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}

//// [device] compute sum of image; 8-bit; 4-channel
//void summation(const iu::ImageGpu_8u_C4 *src, const IuRect &roi, unsigned char sum[4])
//{
//  IuStatus status;
//  status = cusummation(const src, roi, sum);
//  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
//}

///////////////////////////////////////////////////////////////////////////////

// [device] compute sum of image; 32-bit; 1-channel
void summation(const iu::ImageGpu_32f_C1 *src, const IuRect &roi, double &sum)
{
  IuStatus status;
  status = cuSummation(src, roi, sum);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}

// [device] compute sum of volume; 32-bit; 1-channel
void summation(iu::VolumeGpu_32f_C1 *src, const IuCube &roi, double &sum)
{
  IuStatus status;
  sum = 0.0;
  double slice_sum = 0.0;
  for (unsigned int oz=0; oz<roi.depth; ++oz)
  {
    iu::ImageGpu_32f_C1 cur_slice = src->getSlice(oz);
    status = cuSummation(&cur_slice, cur_slice.roi(), slice_sum);
    sum += slice_sum;
  }
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}

//// [device] compute sum of image; 32-bit; 4-channel
//void summation(const iu::ImageGpu_32f_C4 *src, const IuRect &roi, float sum[4])
//{
//  IuStatus status;
//  status = cusummation(const src, roi, sum);
//  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
//}


/* ****************************************************************************
  NORM
*/

// [device] compute L1 norm; |image1-image2|;
void normDiffL1(const iu::ImageGpu_32f_C1* src1, const iu::ImageGpu_32f_C1* src2, const IuRect& roi, double& norm)
{
  IuStatus status;
  status = cuNormDiffL1(src1, src2, roi, norm);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}

// [device] compute L1 norm; |image-value|;
void normDiffL1(const iu::ImageGpu_32f_C1* src, const float& value, const IuRect& roi, double& norm)
{
  IuStatus status;
  status = cuNormDiffL1(src, value, roi, norm);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}

// [device] compute L2 norm; ||image1-image2||;
void normDiffL2(const iu::ImageGpu_32f_C1* src1, const iu::ImageGpu_32f_C1* src2, const IuRect& roi, double& norm)
{
  IuStatus status;
  status = cuNormDiffL2(src1, src2, roi, norm);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}

// [device] compute L2 norm; ||image-value||;
void normDiffL2(const iu::ImageGpu_32f_C1* src, const float& value, const IuRect& roi, double& norm)
{
  IuStatus status;
  status = cuNormDiffL2(src, value, roi, norm);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}

/* ***************************************************************************
   ERROR MEASUREMENTS
*/

// [device] compute MSE;
void mse(const iu::ImageGpu_32f_C1* src, const iu::ImageGpu_32f_C1* reference, const IuRect& roi, double& mse)
{
  IuStatus status;
  status = cuMse(src, reference, roi, mse);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}

// [device] compute SSIM;
void ssim(const iu::ImageGpu_32f_C1* src, const iu::ImageGpu_32f_C1* reference, const IuRect& roi, double& ssim)
{
  IuStatus status;
  status = cuSsim(src, reference, roi, ssim);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}

/* ***************************************************************************
   HISTOGRAMS
*/

void colorHistogram(const iu::ImageGpu_8u_C4* binned_image, const iu::ImageGpu_8u_C1* mask,
                    iu::VolumeGpu_32f_C1* hist, unsigned char mask_val)
{
  IuStatus status;
  status = cuColorHistogram(binned_image, mask, hist, mask_val);
  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
}


} // namespace iuprivate
