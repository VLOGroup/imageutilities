#include <iostream>
#include <typeinfo>

#include "../config.h"
#include "iucore.h"
#include "iumath.h"
#include "iuio.h"
#include "iuhelpermath.h"

template <typename T>
void printImage(T &image)
{
  for (unsigned int y = 0; y < image.size()[1]; y++)
  {
    for (unsigned int x = 0; x < image.size()[0]; x++)
    {
      std::cout << image.getPixel(x,y) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

template <typename T>
void printVolume(T &volume, unsigned int slice)
{
  for (unsigned int y = 0; y < volume.size()[1]; y++)
  {
    for (unsigned int x = 0; x < volume.size()[0]; x++)
    {
      std::cout << volume.getPixel(x,y,slice) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}


void test_fft2_image_float()
{
  std::cout << "TEST: FFT2 IMAGE FLOAT" << std::endl;
  iu::Size<3> size({5,10});
  float data[size.numel()];

  // Fill data pointer
  for (unsigned int i=0; i < size.numel(); i++)
  {
    data[i] = static_cast<float>(i);
  }

  // Attach data pointer to ImageCpu
  iu::ImageCpu_32f_C1 h_img(data, size[0], size[1], size[0]*sizeof(float), true);
  std::cout << "input image:" << std::endl;
  printImage(h_img);

  // Copy to Gpu
  iu::ImageGpu_32f_C1 d_img(size);
  iu::copy(&h_img, &d_img);

  // Create output image with half width
  iu::Size<3> halfsize = size;
  halfsize.width = halfsize.width/2 + 1;
  iu::ImageGpu_32f_C2 d_output_fourier(halfsize);
  iu::math::fill(d_output_fourier, make_float2(0,0));
  iu::ImageCpu_32f_C2 h_output_fourier(halfsize);
  iu::math::fill(h_output_fourier, make_float2(0,0));

  // Perform fft2 real -> complex
  iu::math::fft::fft2(d_img, d_output_fourier);
  iu::copy(&d_output_fourier, &h_output_fourier);
  std::cout << "fourier transform:" << std::endl;
  printImage(h_output_fourier);

  // Perform ifft2 complex -> real
  iu::math::fft::ifft2(d_output_fourier, d_img);
  iu::copy(&d_img, &h_img);
  std::cout << "output image == input image:" << std::endl;
  printImage(h_img);

  std::cout << "DONE: FFT2 IMAGE FLOAT" << std::endl;
}

void test_fft2_image_double()
{
  std::cout << "TEST: FFT2 IMAGE DOUBLE" << std::endl;
  iu::Size<3> size({5,10});
  double data[size.numel()];

  // Fill data pointer
  for (unsigned int i=0; i < size.numel(); i++)
  {
    data[i] = static_cast<double>(i);
  }

  // Attach data pointer to ImageCpu
  iu::ImageCpu_64f_C1 h_img(data, size[0], size[1], size[0]*sizeof(double), true);
  std::cout << "input image:" << std::endl;
  printImage(h_img);

  // Copy to Gpu
  iu::ImageGpu_64f_C1 d_img(size);
  iu::copy(&h_img, &d_img);

  // Create output image with half width
  iu::Size<3> halfsize = size;
  halfsize.width = halfsize.width/2 + 1;
  iu::ImageGpu_64f_C2 d_output_fourier(halfsize);
  iu::math::fill(d_output_fourier, make_double2(0,0));
  iu::ImageCpu_64f_C2 h_output_fourier(halfsize);
  iu::math::fill(h_output_fourier, make_double2(0,0));

  // Perform fft2 real -> complex
  iu::math::fft::fft2(d_img, d_output_fourier);
  iu::copy(&d_output_fourier, &h_output_fourier);
  std::cout << "fourier transform:" << std::endl;
  printImage(h_output_fourier);

  // Perform ifft2 complex -> real
  iu::math::fft::ifft2(d_output_fourier, d_img);
  iu::copy(&d_img, &h_img);
  std::cout << "output image == input image:" << std::endl;
  printImage(h_img);

  std::cout << "DONE: FFT2 IMAGE DOUBLE" << std::endl;
}

void test_fft2_volume_float()
{
  unsigned int slice = 2;
  std::cout << "TEST: FFT2 VOLUME FLOAT" << std::endl;
  iu::Size<3> size({5,10,4});
  float data[size.numel()];

  // Fill data pointer
  for (unsigned int i=0; i < size.numel(); i++)
  {
    data[i] = static_cast<float>(i);
  }

  // Attach data pointer to VolumeCpu
  iu::VolumeCpu_32f_C1 h_img(data, size[0], size[1], size[2], size[0]*sizeof(float), true);
  std::cout << "input volume:" << std::endl;
  printVolume(h_img, 2);

  // Copy to Gpu
  iu::VolumeGpu_32f_C1 d_img(size);
  iu::copy(&h_img, &d_img);

  // Create output image with half width
  iu::Size<3> halfsize = size;
  halfsize.width = halfsize.width/2 + 1;
  iu::VolumeGpu_32f_C2 d_output_fourier(halfsize);
  iu::math::fill(d_output_fourier, make_float2(0,0));
  iu::VolumeCpu_32f_C2 h_output_fourier(halfsize);
  iu::math::fill(h_output_fourier, make_float2(0,0));

  // Perform fft2 real -> complex
  iu::math::fft::fft2(d_img, d_output_fourier);
  iu::copy(&d_output_fourier, &h_output_fourier);
  std::cout << "fourier transform:" << std::endl;
  printVolume(h_output_fourier, slice);

  // Perform ifft2 complex -> real
  iu::math::fft::ifft2(d_output_fourier, d_img);
  iu::copy(&d_img, &h_img);
  std::cout << "output volume == input volume:" << std::endl;
  printVolume(h_img, slice);

  std::cout << "DONE: FFT2 VOLUME FLOAT" << std::endl;
}

void test_fft2_volume_double()
{
  unsigned int slice = 2;
  std::cout << "TEST: FFT2 VOLUME DOUBLE" << std::endl;
  iu::Size<3> size({5,10,4});
  double data[size.numel()];

  // Fill data pointer
  for (unsigned int i=0; i < size.numel(); i++)
  {
    data[i] = static_cast<double>(i);
  }

  // Attach data pointer to VolumeCpu
  iu::VolumeCpu_64f_C1 h_img(data, size[0], size[1], size[2], size[0]*sizeof(double), true);
  std::cout << "input volume:" << std::endl;
  printVolume(h_img, 2);

  // Copy to Gpu
  iu::VolumeGpu_64f_C1 d_img(size);
  iu::copy(&h_img, &d_img);

  // Create output image with half width
  iu::Size<3> halfsize = size;
  halfsize.width = halfsize.width/2 + 1;
  iu::VolumeGpu_64f_C2 d_output_fourier(halfsize);
  iu::math::fill(d_output_fourier, make_double2(0,0));
  iu::VolumeCpu_64f_C2 h_output_fourier(halfsize);
  iu::math::fill(h_output_fourier, make_double2(0,0));

  // Perform fft2 real -> complex
  iu::math::fft::fft2(d_img, d_output_fourier);
  iu::copy(&d_output_fourier, &h_output_fourier);
  std::cout << "fourier transform:" << std::endl;
  printVolume(h_output_fourier, slice);

  // Perform ifft2 complex -> real
  iu::math::fft::ifft2(d_output_fourier, d_img);
  iu::copy(&d_img, &h_img);
  std::cout << "output volume == input volume:" << std::endl;
  printVolume(h_img, slice);

  std::cout << "DONE: FFT2 VOLUME DOUBLE" << std::endl;
}

template<typename PixelType>
void test_fft2_linmem2()
{
  std::cout << "TEST: FFT2 LINMEM2 " << typeid(PixelType).name() << std::endl;
  iu::Size<2> size({5,10});
  PixelType data[size.numel()];

  // Fill data pointer
  for (unsigned int i=0; i < size.numel(); i++)
  {
    data[i] = static_cast<PixelType>(i);
  }

  // Attach data pointer to LinearHostMemory
  iu::LinearHostMemory<PixelType, 2> h_img(data, size, true);
  std::cout << "input image:" << std::endl;
  printImage(h_img);

  // Copy to Gpu
  iu::LinearDeviceMemory<PixelType, 2>  d_img(size);
  iu::copy(&h_img, &d_img);

  // Create output image with half width
  iu::Size<2> halfsize = size;
  halfsize[0] = halfsize[0]/2 + 1;

  // Define the complex data type
  typedef typename iu::VectorType<PixelType, 2>::type ComplexType;
  iu::LinearDeviceMemory<ComplexType, 2>  d_output_fourier(halfsize);
  iu::math::fill(d_output_fourier, iu::VectorType<PixelType, 2>::make(0));
  iu::LinearHostMemory<ComplexType, 2> h_output_fourier(halfsize);
  iu::math::fill(h_output_fourier, iu::VectorType<PixelType, 2>::make(0));

  // Perform fft2 real -> complex
  iu::math::fft::fft2(d_img, d_output_fourier);
  iu::copy(&d_output_fourier, &h_output_fourier);
  std::cout << "fourier transform:" << std::endl;
  printImage(h_output_fourier);

  // Perform ifft2 complex -> real
  iu::math::fft::ifft2(d_output_fourier, d_img);
  iu::copy(&d_img, &h_img);
  std::cout << "output image == input image:" << std::endl;
  printImage(h_img);

  std::cout << "DONE: FFT2 LINMEM " << typeid(PixelType).name() << std::endl;
}

template<typename PixelType, unsigned int Ndim>
void test_fft2_linmemNd()
{
  unsigned int slice = 6;
  bool scale_sqrt = true;
  std::cout << "TEST: FFT2 LINMEM" << Ndim << " " << typeid(PixelType).name() << std::endl;
  iu::Size<Ndim> size;
  for (unsigned int i = 0; i < Ndim; i++)
    size[i] = 5 + i;
  PixelType data[size.numel()];

  // Fill data pointer
  for (unsigned int i=0; i < size.numel(); i++)
  {
    data[i] = static_cast<PixelType>(i);
  }

  // Attach data pointer to LinearHostMemory
  iu::LinearHostMemory<PixelType, Ndim> h_img(data, size, true);
  std::cout << "input image:" << std::endl;
  printVolume(h_img, slice);

  // Copy to Gpu
  iu::LinearDeviceMemory<PixelType, Ndim>  d_img(size);
  iu::copy(&h_img, &d_img);

  // Create output image with half width
  iu::Size<Ndim> halfsize = size;
  halfsize[0] = halfsize[0]/2 + 1;

  // Define the complex data type
  typedef typename iu::VectorType<PixelType, 2>::type ComplexType;
  iu::LinearDeviceMemory<ComplexType, Ndim>  d_output_fourier(halfsize);
  iu::math::fill(d_output_fourier, iu::VectorType<PixelType, 2>::make(0));
  iu::LinearHostMemory<ComplexType, Ndim> h_output_fourier(halfsize);
  iu::math::fill(h_output_fourier, iu::VectorType<PixelType, 2>::make(0));

  // Perform fft2 real -> complex
  iu::math::fft::fft2(d_img, d_output_fourier, scale_sqrt);
  iu::copy(&d_output_fourier, &h_output_fourier);
  std::cout << "fourier transform:" << std::endl;
  printVolume(h_output_fourier, slice);

  // Perform ifft2 complex -> real
  iu::math::fft::ifft2(d_output_fourier, d_img, scale_sqrt);
  iu::copy(&d_img, &h_img);
  std::cout << "output image == input image:" << std::endl;
  printVolume(h_img, slice);

  std::cout << "DONE: FFT2 LINMEM" << Ndim << " " << typeid(PixelType).name() << std::endl;
}

int main()
{
    // 2D real -> complex fft / complex -> real ifft
    test_fft2_image_float();
    test_fft2_image_double();
    test_fft2_volume_float();
    test_fft2_volume_double();

    test_fft2_linmem2<float>();
    test_fft2_linmem2<double>();

    test_fft2_linmemNd<float, 4>();
    test_fft2_linmemNd<double, 4>();

    // TODO complex -> complex
    // TODO fftshift / ifftshift
    // TODO get rid of pitch in constructor with external datapointer -> make internally
    // TODO make second special constructor based on size
    typedef double PixelType;
    iu::VectorType<PixelType, 2>::type vec = iu::VectorType<PixelType, 2>::make(5);
    std::cout << vec << " " << length(vec) << std::endl;

    std::cout << "DONE :)" << std::endl;
}
