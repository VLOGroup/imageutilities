
#include <cstring>
#include "convert.h"


namespace iuprivate {

/* ***************************************************************************
 *  Declaration of CUDA WRAPPERS
 * ***************************************************************************/
extern void cuConvert(const iu::ImageGpu_32f_C3* src,
                          iu::ImageGpu_32f_C4* dst);
extern void cuConvert(const iu::ImageGpu_32f_C4* src,
                          iu::ImageGpu_32f_C3* dst);
extern void cuConvert_8u_32f(const iu::ImageGpu_8u_C1* src,
                                 iu::ImageGpu_32f_C1* dst,
                                 float mul_constant,  float add_constant);
extern void cuConvert_32u_32f(const iu::ImageGpu_32u_C1* src,
                                 iu::ImageGpu_32f_C1* dst,
                                 float mul_constant,  float add_constant);
extern void cuConvert_8u_32f_C3C4(const iu::ImageGpu_8u_C3* src,
                                 iu::ImageGpu_32f_C4* dst,
                                 float mul_constant,  float add_constant);
extern void cuConvert_32f_8u(const iu::ImageGpu_32f_C1* src,
                                 iu::ImageGpu_8u_C1* dst,
                                 float mul_constant, unsigned char add_constant);
extern void cuConvert_32f_8u(const iu::ImageGpu_32f_C4* src,
                                 iu::ImageGpu_8u_C4* dst,
                                 float mul_constant, unsigned char add_constant);

// src not declared as const to allow in-place conversion
extern void cuConvert_32s_32f_lin(iu::LinearDeviceMemory_32s_C1 *src, iu::LinearDeviceMemory_32f_C1 *dest);

extern void cuConvert_rgb_to_hsv(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst, bool normalize);
extern void cuConvert_hsv_to_rgb(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst, bool denormalize);

extern void cuConvert_rgb_to_lab(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst, bool isNormalized);
extern void cuConvert_lab_to_rgb(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst);


//extern double cuSummation(iu::ImageGpu_32f_C1* src);

/* ***************************************************************************/



/* ***************************************************************************
 *  FUNCTION IMPLEMENTATIONS
 * ***************************************************************************/

//-----------------------------------------------------------------------------
// [host] 2D bit depth conversion; 32f_C1 -> 8u_C1;
void convert_32f8u_C1(const iu::ImageCpu_32f_C1* src, iu::ImageCpu_8u_C1 *dst,
                      float mul_constant, float add_constant)
{
  for (unsigned int x=0; x<dst->width(); x++)
  {
    for (unsigned int y=0; y<dst->height(); y++)
    {
      float val = *src->data(x,y);
      *dst->data(x,y) = static_cast<int>(mul_constant*val + add_constant);
    }
  }
}

//-----------------------------------------------------------------------------
// [host] 2D bit depth conversion; 16u_C1 -> 32f_C1;
void convert_16u32f_C1(const iu::ImageCpu_16u_C1* src, iu::ImageCpu_32f_C1 *dst,
                       float mul_constant, float add_constant)
{
  for (unsigned int x=0; x<dst->width(); x++)
  {
    for (unsigned int y=0; y<dst->height(); y++)
    {
      unsigned short val = *src->data(x,y); //((*src->data(x,y) & 0x00ffU) << 8) | ((*src->data(x,y) & 0xff00U) >> 8);
      *dst->data(x,y) = mul_constant*(float)val + add_constant;
    }
  }
}

//-----------------------------------------------------------------------------
// [device] conversion 32f_C1 -> 8u_C1
void convert_32f8u_C1(const iu::ImageGpu_32f_C1* src, iu::ImageGpu_8u_C1 *dst,
                      float mul_constant, unsigned char add_constant)
{
  
  cuConvert_32f_8u(src, dst, mul_constant, add_constant);

}

//-----------------------------------------------------------------------------
// [device] conversion 32f_C4 -> 8u_C4
void convert_32f8u_C4(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_8u_C4 *dst,
                      float mul_constant, unsigned char add_constant)
{
  
  cuConvert_32f_8u(src, dst, mul_constant, add_constant);

}



//-----------------------------------------------------------------------------
// [device] conversion 8u_C1 -> 32f_C1
void convert_8u32f_C1(const iu::ImageGpu_8u_C1* src, iu::ImageGpu_32f_C1 *dst,
                      float mul_constant, float add_constant)
{
  cuConvert_8u_32f(src, dst, mul_constant, add_constant);
}


//-----------------------------------------------------------------------------
// [device] conversion 32u_C1 -> 32f_C1
void convert_32u32f_C1(const iu::ImageGpu_32u_C1* src, iu::ImageGpu_32f_C1 *dst,
                       float mul_constant, float add_constant)
{
    cuConvert_32u_32f(src, dst, mul_constant, add_constant);
}

void convert_32s32f_C1_lin(iu::LinearDeviceMemory_32s_C1 *src, iu::LinearDeviceMemory_32f_C1 *dest)
{
	cuConvert_32s_32f_lin(src, dest);
}

//-----------------------------------------------------------------------------
// [host] conversion 32u_C1 -> 32f_C1
void convert_32u32f_C1(const iu::ImageCpu_32u_C1* src, iu::ImageCpu_32f_C1 *dst,
                       float mul_constant, float add_constant)
{
    for (unsigned int x=0; x<dst->width(); x++)
    {
      for (unsigned int y=0; y<dst->height(); y++)
      {
        unsigned short val = *src->data(x,y); //((*src->data(x,y) & 0x00ffU) << 8) | ((*src->data(x,y) & 0xff00U) >> 8);
        *dst->data(x,y) = mul_constant*(float)val + add_constant;
      }
    }
}

//-----------------------------------------------------------------------------
// [device] conversion 8u_C3 -> 32f_C4
void convert_8u32f_C3C4(const iu::ImageGpu_8u_C3* src, iu::ImageGpu_32f_C4 *dst,
                      float mul_constant, float add_constant)
{
  
  cuConvert_8u_32f_C3C4(src, dst, mul_constant, add_constant);

}

//-----------------------------------------------------------------------------
// [device] conversion 32f_C3 -> 32f_C4
void convert(const iu::ImageGpu_32f_C3* src, iu::ImageGpu_32f_C4* dst)
{
  
  cuConvert(src, dst);

}

//-----------------------------------------------------------------------------
// [device] conversion 32f_C4 -> 32f_C3
void convert(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C3* dst)
{
  
  cuConvert(src, dst);

}

//-----------------------------------------------------------------------------
// [device] conversion RGB -> HSV
void convertRgbHsv(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst, bool normalize)
{
  
  cuConvert_rgb_to_hsv(src, dst, normalize);

}

//-----------------------------------------------------------------------------
// [device] conversion HSV -> RGB
void convertHsvRgb(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst, bool denormalize)
{
  
  cuConvert_hsv_to_rgb(src, dst, denormalize);

}


//-----------------------------------------------------------------------------
// [device] conversion RGB -> CIELAB
void convertRgbLab(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst, bool isNormalized)
{
  
  cuConvert_rgb_to_lab(src, dst, isNormalized);

}


//-----------------------------------------------------------------------------
// [device] conversion CIELAB -> RGB
void convertLabRgb(const iu::ImageGpu_32f_C4* src, iu::ImageGpu_32f_C4* dst)
{
  
  cuConvert_lab_to_rgb(src, dst);

}

//double summation(iu::ImageGpu_32f_C1 *src)
//{
//    return cuSummation(src);
//}


} // namespace iuprivate
