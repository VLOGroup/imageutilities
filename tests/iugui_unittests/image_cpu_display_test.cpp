
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include <QApplication>

#include <GL/glew.h>
#include <cuda_runtime.h>
#include <iu/iucore.h>
#include <iu/iuio.h>
#include <iu/iugui.h>

int main(int argc, char **argv)
{
  if(argc < 2)
  {
    std::cout << "usage: gl_display_test image_filename" << std::endl;
    exit(EXIT_FAILURE);
  }
  printf("start gl test\n");

  QApplication app(argc, argv);

  // First load the image, so we know what the size of the image (imageW and imageH)
  printf("Allocating host and CUDA memory and loading image file...\n");
  std::string in_file = argv[1];
  std::cout << "reading " << in_file << std::endl;

  iu::ImageCpu_8u_C1* image_8u_C1 = iu::imread_8u_C1(in_file);
  iu::ImageGpu_8u_C1* cu_image_8u_C1 = new iu::ImageGpu_8u_C1(image_8u_C1->size());
  iu::ImageCpu_32f_C1* image_32f_C1 = iu::imread_32f_C1(in_file);
  iu::ImageGpu_32f_C1* cu_image_32f_C1 = new iu::ImageGpu_32f_C1(image_32f_C1->size());
  iu::ImageCpu_8u_C4* image_8u_C4 = iu::imread_8u_C4(in_file);
  iu::ImageGpu_8u_C4* cu_image_8u_C4 = new iu::ImageGpu_8u_C4(image_8u_C4->size());
  iu::ImageCpu_32f_C4* image_32f_C4 = iu::imread_32f_C4(in_file);
  iu::ImageGpu_32f_C4* cu_image_32f_C4 = new iu::ImageGpu_32f_C4(image_32f_C4->size());
  iu::copy(image_8u_C1, cu_image_8u_C1);
  iu::copy(image_32f_C1, cu_image_32f_C1);
  iu::copy(image_8u_C4, cu_image_8u_C4);
  iu::copy(image_32f_C4, cu_image_32f_C4);
  //iu::imshow(cu_image_8u, "gpu_image");

  double time = iu::getTime();
  iu::QImageCpuDisplay im_display_8u_C1(image_8u_C1, "cpu image 8u C1");
  im_display_8u_C1.show();
  std::cout << "display create/show/close takes " << (iu::getTime()-time) << "ms" << std::endl;

  time = iu::getTime();
  iu::QImageCpuDisplay im_display_8u_C4(image_8u_C4, "cpu image 8u C4");
  im_display_8u_C4.show();
  std::cout << "display create/show/close takes " << (iu::getTime()-time) << "ms" << std::endl;

  time = iu::getTime();
  iu::QImageCpuDisplay im_display_32f_C1(image_32f_C1, "cpu image 32f C1");
  im_display_32f_C1.show();
  std::cout << "display create/show/close takes " << (iu::getTime()-time) << "ms" << std::endl;

  time = iu::getTime();
  iu::QImageCpuDisplay im_display_32f_C4(image_32f_C4, "cpu image 32f C4");
  im_display_32f_C4.show();
  std::cout << "display create/show/close takes " << (iu::getTime()-time) << "ms" << std::endl;

  app.exec();
  cudaThreadExit();
  return(EXIT_SUCCESS);
}
