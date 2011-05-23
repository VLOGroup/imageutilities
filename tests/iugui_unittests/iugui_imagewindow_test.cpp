// includes, GL
#include <GL/glew.h>

#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

// includes
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cutil_gl_error.h>
#include <cuda_gl_interop.h>



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


////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
void dumm_display( void ) {}

CUTBoolean initGL(int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(10, 10);
    int bla = glutCreateWindow("Cuda GL Interop (VBO)");
    glutDisplayFunc(dumm_display);
//    glutKeyboardFunc(keyboard);
//    glutMotionFunc(motion);

        // initialize necessary OpenGL extensions
    glewInit();
    if (! glewIsSupported("GL_VERSION_2_0 ")) {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return CUTFalse;
    }

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, 10, 10);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)10 / (GLfloat) 10, 0.1, 10.0);

    CUT_CHECK_ERROR_GL();

    // start gui for the main application
    //cudaError_t error = cudaGLSetGLDevice(0);
    //cutilGLDeviceInit(argc, argv);
    int deviceCount;
    cutilSafeCallNoSync(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "CUTIL CUDA error: no devices supporting CUDA.\n");
        exit(-1);
    }
    int dev = 0;
    cudaDeviceProp deviceProp;
    cutilSafeCallNoSync(cudaGetDeviceProperties(&deviceProp, dev));
    if (deviceProp.major < 1) {
        fprintf(stderr, "cutil error: device does not support CUDA.\n");
        exit(-1);
    }
    printf("gpu=%s\n", deviceProp.name);
    cutilSafeCall(cudaGLSetGLDevice(dev));

    glutDestroyWindow(bla);

    return CUTTrue;
}

int main(int argc, char **argv)
{
  //cudaGLSetGLDevice(1);

  if(argc < 2)
  {
    std::cout << "usage: iugui_imagewindow_test image_filename" << std::endl;
    exit(EXIT_FAILURE);
  }
  printf("start ImageWindow test\n");

  // we want to remove this!!!
  initGL(argc,argv);

  QApplication app(argc, argv);

  // First load the image, so we know what the size of the image (imageW and imageH)
  std::string in_file = argv[1];
  std::cout << "reading " << in_file << " into gpu memory." << std::endl;

  iu::ImageGpu_8u_C1* image_8u_C1 = iu::imread_cu8u_C1(in_file);
  iu::ImageGpu_32f_C1* image_32f_C1 = iu::imread_cu32f_C1(in_file);
  iu::ImageGpu_8u_C4* image_8u_C4 = iu::imread_cu8u_C4(in_file);
  iu::ImageGpu_32f_C4* image_32f_C4 = iu::imread_cu32f_C4(in_file);
//  iu::imshow(image_8u_C4, "reference input image");

  double time = iu::getTime();
  iu::ImageWindow win_8u_C1;
  win_8u_C1.setWindowTitle("gpu 8u C1");
  win_8u_C1.setImage(image_8u_C1);
  win_8u_C1.show();
  std::cout << "display create/show takes " << (iu::getTime()-time) << "ms" << std::endl;

//  time = iu::getTime();
//  iu::QGLImageGpuWidget widget_8u_C1(0);
//  widget_8u_C1.setWindowTitle("qgl widget: 8u_C1");
//  widget_8u_C1.setImage(image_8u_C1);
//  widget_8u_C1.show();
//  std::cout << "display create/show takes " << (iu::getTime()-time) << "ms" << std::endl;


//  // update test
//  std::cout << "reading " << in_file2 << std::endl;
//  iu::ImageCpu_8u_C1* image2_8u_C1 = iu::imread_8u_C1(in_file2);
//  im_display_8u_C1.updateImage(image2_8u_C1);

//  // check gpu widget
//  iu::ImageGpu_8u_C1* image_cu8u_C1 = iu::imread_cu8u_C1(in_file);
//  iu::ImageGpu_32f_C1* image_cu32f_C1 = iu::imread_cu32f_C1(in_file);
//  iu::ImageGpu_8u_C4* image_cu8u_C4 = iu::imread_cu8u_C4(in_file);
//  iu::ImageGpu_8u_C4* image2_cu8u_C4 = iu::imread_cu8u_C4(in_file2);
//  iu::ImageGpu_32f_C4* image_cu32f_C4 = iu::imread_cu32f_C4(in_file);



  app.exec();

  delete(image_8u_C1);
  delete(image_8u_C4);
  delete(image_32f_C1);
  delete(image_32f_C4);

  cudaThreadExit();
  return(EXIT_SUCCESS);
}
