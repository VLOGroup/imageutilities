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

#include <iucore/copy.h>
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
//  ippiMinMax_8u_C1R(src->data(roi.x, roi.y), static_cast<int>(src->pitch()), roi_size, &min, &max);
//  IU_ASSERT(status == ippStsNoErr);
//}

//// [host] find min/max value of image; 8-bit; 3-channel
//void minMax(const iu::ImageCpu_8u_C3 *src, const IuRect &roi, unsigned char min[3], unsigned char max[3])
//{
//  IppStatus status;
//  IppiSize roi_size = {roi.width, roi.height};
//  ippiMinMax_8u_C3R(src->data(roi.x, roi.y), static_cast<int>(src->pitch()), roi_size, min, max);
//  IU_ASSERT(status == ippStsNoErr);
//}

//// [host] find min/max value of image; 8-bit; 4-channel
//void minMax(const iu::ImageCpu_8u_C4 *src, const IuRect &roi, unsigned char min[4], unsigned char max[4])
//{
//  IppStatus status;
//  IppiSize roi_size = {roi.width, roi.height};
//  ippiMinMax_8u_C4R(src->data(roi.x, roi.y), static_cast<int>(src->pitch()), roi_size, min, max);
//  IU_ASSERT(status == ippStsNoErr);
//}

/////////////////////////////////////////////////////////////////////////////////

//// [host] find min/max value of image; 32-bit; 1-channel
//void minMax(const iu::ImageCpu_32f_C1 *src, const IuRect &roi, float& min, float& max)
//{
//  IppStatus status;
//  IppiSize roi_size = {roi.width, roi.height};
//  ippiMinMax_32f_C1R(src->data(roi.x, roi.y), static_cast<int>(src->pitch()), roi_size, &min, &max);
//  IU_ASSERT(status == ippStsNoErr);
//}

//// [host] find min/max value of image; 32-bit; 3-channel
//void minMax(const iu::ImageCpu_32f_C3 *src, const IuRect &roi, float min[3], float max[3])
//{
//  IppStatus status;
//  IppiSize roi_size = {roi.width, roi.height};
//  ippiMinMax_32f_C3R(src->data(roi.x, roi.y), static_cast<int>(src->pitch()), roi_size, min, max);
//  IU_ASSERT(status == ippStsNoErr);
//}

//// [host] find min/max value of image; 32-bit; 4-channel
//void minMax(const iu::ImageCpu_32f_C4 *src, const IuRect &roi, float min[4], float max[4])
//{
//  IppStatus status;
//  IppiSize roi_size = {roi.width, roi.height};
//  ippiMinMax_32f_C4R(src->data(roi.x, roi.y), static_cast<int>(src->pitch()), roi_size, min, max);
//  IU_ASSERT(status == ippStsNoErr);
//}

///////////////////////////////////////////////////////////////////////////////

// [device] find min/max value of image; 8-bit; 1-channel
void minMax(const iu::ImageGpu_8u_C1 *src, const IuRect &roi, unsigned char& min, unsigned char& max)
{
  cuMinMax(src, roi, min, max);
}

// [device] find min/max value of image; 8-bit; 4-channel
void minMax(const iu::ImageGpu_8u_C4 *src, const IuRect &roi, uchar4& min, uchar4& max)
{
  cuMinMax(src, roi, min, max);
}

///////////////////////////////////////////////////////////////////////////////

// [device] find min/max value of image; 32-bit; 1-channel
void minMax(const iu::ImageGpu_32f_C1 *src, const IuRect &roi, float& min, float& max)
{
  cuMinMax(src, roi, min, max);
}

// [device] find min/max value of image; 32-bit; 2-channel
void minMax(const iu::ImageGpu_32f_C2 *src, const IuRect &roi, float2& min, float2& max)
{
  cuMinMax(src, roi, min, max);
}

// [device] find min/max value of image; 32-bit; 4-channel
void minMax(const iu::ImageGpu_32f_C4 *src, const IuRect &roi, float4& min, float4& max)
{
  cuMinMax(src, roi, min, max);
}



// [host] find min/max value of image; 8-bit; 1-channel
void minMax(const iu::ImageCpu_8u_C1 *src, const IuRect &roi, unsigned char& min, unsigned char& max)
{
  iu::ImageGpu_8u_C1 temp(src->size());
  iuprivate::copy(src, &temp);
  cuMinMax(&temp, roi, min, max);
}

// [host] find min/max value of image; 8-bit; 4-channel
void minMax(const iu::ImageCpu_8u_C4 *src, const IuRect &roi, uchar4& min, uchar4& max)
{
  iu::ImageGpu_8u_C4 temp(src->size());
  iuprivate::copy(src, &temp);
  cuMinMax(&temp, roi, min, max);
}

///////////////////////////////////////////////////////////////////////////////

// [host] find min/max value of image; 32-bit; 1-channel
void minMax(const iu::ImageCpu_32f_C1 *src, const IuRect &roi, float& min, float& max)
{
  iu::ImageGpu_32f_C1 temp(src->size());
  iuprivate::copy(src, &temp);
  cuMinMax(&temp, roi, min, max);
}

// [host] find min/max value of image; 32-bit; 2-channel
void minMax(const iu::ImageCpu_32f_C2 *src, const IuRect &roi, float2& min, float2& max)
{
  iu::ImageGpu_32f_C2 temp(src->size());
  iuprivate::copy(src, &temp);
  cuMinMax(&temp, roi, min, max);
}

// [host] find min/max value of image; 32-bit; 4-channel
void minMax(const iu::ImageCpu_32f_C4 *src, const IuRect &roi, float4& min, float4& max)
{
  iu::ImageGpu_32f_C4 temp(src->size());
  iuprivate::copy(src, &temp);
  cuMinMax(&temp, roi, min, max);
}


///////////////////////////////////////////////////////////////////////////////

// [device] find min/max value of volume; 32-bit; 1-channel
void minMax(const iu::VolumeGpu_32f_C1 *src, float& min, float& max)
{
  cuMinMax(src, min, max);
}

// [host] find min/max value of volume; 32-bit; 4-channel
void minMax(const iu::VolumeCpu_32f_C4 *src, float4& min, float4& max)
{

  float4 minTemp = make_float4(1e6f,1e6f,1e6f,1e6f);
  float4 maxTemp = make_float4(-1e6f,-1e6f,-1e6f,-1e6f);

  for (unsigned int z = 0; z < src->depth(); z++)
  {
    for (unsigned int y = 0; y < src->height(); y++)
    {
      for (unsigned int x = 0; x < src->width(); x++)
      {
        float4 val = *src->data(x,y,z);
        if (val.x < minTemp.x)
          minTemp.x = val.x;
        if (val.y < minTemp.y)
          minTemp.y = val.y;
        if (val.z < minTemp.z)
          minTemp.z = val.z;
        if (val.w < minTemp.w)
          minTemp.w = val.w;

        if (val.x > maxTemp.x)
          maxTemp.x = val.x;
        if (val.y > maxTemp.y)
          maxTemp.y = val.y;
        if (val.z > maxTemp.z)
          maxTemp.z = val.z;
        if (val.w > maxTemp.w)
          maxTemp.w = val.w;
      }
    }
  }
  min = minTemp;
  max = maxTemp;
}



///////////////////////////////////////////////////////////////////////////////

// [device] find min value and its coordinates of image; 32-bit; 1-channel
void min(const iu::ImageGpu_32f_C1* src, const IuRect&roi, float& min, int& x, int& y)
{
  cuMin(src, roi, min, x, y);
}

// [device] find max value and its coordinates of image; 32-bit; 1-channel
void max(const iu::ImageGpu_32f_C1* src, const IuRect&roi, float& max, int& x, int& y)
{
  cuMax(src, roi, max, x, y);
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
//  ippiSum_8u_C1R(src->data(roi.x, roi.y), static_cast<int>(src->pitch()), roi_size, &sum);
//  IU_ASSERT(status == ippStsNoErr);
//}

//// [host] compute sum of image; 8-bit; 3-channel
//void summation(const iu::ImageCpu_8u_C3 *src, const IuRect &roi, Ipp64f sum[3])
//{
//  IppStatus status;
//  IppiSize roi_size = {roi.width, roi.height};
//  ippiSum_8u_C3R(src->data(roi.x, roi.y), static_cast<int>(src->pitch()), roi_size, sum);
//  IU_ASSERT(status == ippStsNoErr);
//}

//// [host] compute sum of image; 8-bit; 4-channel
//void summation(const iu::ImageCpu_8u_C4 *src, const IuRect &roi, Ipp64f sum[4])
//{
//  IppStatus status;
//  IppiSize roi_size = {roi.width, roi.height};
//  ippiSum_8u_C4R(src->data(roi.x, roi.y), static_cast<int>(src->pitch()), roi_size, sum);
//  IU_ASSERT(status == ippStsNoErr);
//}

/////////////////////////////////////////////////////////////////////////////////

//// [host] compute sum of image; 32-bit; 1-channel
//void summation(const iu::ImageCpu_32f_C1 *src, const IuRect &roi, Ipp64f& sum)
//{
//  IppStatus status;
//  IppiSize roi_size = {roi.width, roi.height};
//  ippiSum_32f_C1R(src->data(roi.x, roi.y), static_cast<int>(src->pitch()), roi_size, &sum, ippAlgHintNone);
//  IU_ASSERT(status == ippStsNoErr);
//}

//// [host] compute sum of image; 32-bit; 3-channel
//void summation(const iu::ImageCpu_32f_C3 *src, const IuRect &roi, Ipp64f sum[3])
//{
//  IppStatus status;
//  IppiSize roi_size = {roi.width, roi.height};
//  ippiSum_32f_C3R(src->data(roi.x, roi.y), static_cast<int>(src->pitch()), roi_size, sum, ippAlgHintNone);
//  IU_ASSERT(status == ippStsNoErr);
//}

//// [host] compute sum of image; 32-bit; 4-channel
//void summation(const iu::ImageCpu_32f_C4 *src, const IuRect &roi, Ipp64f sum[4])
//{
//  IppStatus status;
//  IppiSize roi_size = {roi.width, roi.height};
//  ippiSum_32f_C4R(src->data(roi.x, roi.y), static_cast<int>(src->pitch()), roi_size, sum, ippAlgHintNone);
//  IU_ASSERT(status == ippStsNoErr);
//}

///////////////////////////////////////////////////////////////////////////////

// [device] compute sum of image; 8-bit; 1-channel
void summation(const iu::ImageGpu_8u_C1 *src, const IuRect &roi, long& sum)
{
  cuSummation(src, roi, sum);
}

//// [device] compute sum of image; 8-bit; 4-channel
//void summation(const iu::ImageGpu_8u_C4 *src, const IuRect &roi, unsigned char sum[4])
//{
//  
//  cusummation(const src, roi, sum);
//  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
//}

///////////////////////////////////////////////////////////////////////////////

// [device] compute sum of image; 32-bit; 1-channel
void summation(const iu::ImageGpu_32f_C1 *src, const IuRect &roi, double &sum,
               iu::LinearDeviceMemory_32f_C1 *sum_temp)
{
  cuSummation(src, roi, sum, sum_temp);
}

// [device] compute sum of volume; 32-bit; 1-channel
void summation(iu::VolumeGpu_32f_C1 *src, const IuCube &roi, double &sum)
{
  sum = 0.0;
  double slice_sum = 0.0;
  for (unsigned int oz=0; oz<roi.depth; ++oz)
  {
    const iu::ImageGpu_32f_C1 cur_slice = src->getSlice(oz);
    cuSummation(&cur_slice, cur_slice.roi(), slice_sum);
    sum += slice_sum;
  }
}

//// [device] compute sum of image; 32-bit; 4-channel
//void summation(const iu::ImageGpu_32f_C4 *src, const IuRect &roi, float sum[4])
//{
//  
//  cusummation(const src, roi, sum);
//  if (status != IU_SUCCESS) throw IuException("function returned with an error", __FILE__, __FUNCTION__, __LINE__);
//}


/* ****************************************************************************
  NORM
*/

// [device] compute L1 norm; |image1-image2|;
void normDiffL1(const iu::ImageGpu_32f_C1* src1, const iu::ImageGpu_32f_C1* src2, const IuRect& roi, double& norm)
{
  cuNormDiffL1(src1, src2, roi, norm);
}

// [device] compute L1 norm; |image-value|;
void normDiffL1(const iu::ImageGpu_32f_C1* src, const float& value, const IuRect& roi, double& norm)
{
  
  cuNormDiffL1(src, value, roi, norm);

}

// [device] compute L2 norm; ||image1-image2||;
void normDiffL2(const iu::ImageGpu_32f_C1* src1, const iu::ImageGpu_32f_C1* src2, const IuRect& roi, double& norm)
{
  
  cuNormDiffL2(src1, src2, roi, norm);

}

// [device] compute L2 norm; ||image-value||;
void normDiffL2(const iu::ImageGpu_32f_C1* src, const float& value, const IuRect& roi, double& norm)
{
  
  cuNormDiffL2(src, value, roi, norm);

}

/* ***************************************************************************
   ERROR MEASUREMENTS
*/

// [device] compute MSE;
void mse(const iu::ImageGpu_32f_C1* src, const iu::ImageGpu_32f_C1* reference, const IuRect& roi, double& mse)
{
  
  cuMse(src, reference, roi, mse);

}

// [device] compute SSIM;
void ssim(const iu::ImageGpu_32f_C1* src, const iu::ImageGpu_32f_C1* reference, const IuRect& roi, double& ssim)
{
  
  cuSsim(src, reference, roi, ssim);

}

/* ***************************************************************************
   HISTOGRAMS
*/

void colorHistogram(const iu::ImageGpu_8u_C4* binned_image, const iu::ImageGpu_8u_C1* mask,
                    iu::VolumeGpu_32f_C1* hist, unsigned char mask_val)
{
  
  cuColorHistogram(binned_image, mask, hist, mask_val);

}


} // namespace iuprivate
