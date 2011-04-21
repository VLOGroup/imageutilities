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
 * Module      : GUI
 * Class       : QGLImageGpuWidget
 * Language    : C++
 * Description : Definition of a QGLWidget rendering GPU memory (2D)
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#include <GL/glew.h>
#include <cuda_runtime.h>
#include <iumath.h>
#include "qgl_image_gpu_widget_p.h"
#include "qgl_image_gpu_widget.h"

namespace iuprivate {

extern IuStatus cuCopyImageToPbo(iu::Image* image,
                                 unsigned int num_channels, unsigned int bit_depth,
                                 uchar4 *dst,
                                 float min=0.0f, float max=1.0f);

//-----------------------------------------------------------------------------
QGLImageGpuWidget::QGLImageGpuWidget(QWidget *parent) :
  QGLWidget(parent),
  gl_pbo_(NULL),
  gl_tex_(NULL),
  cuda_pbo_resource_(NULL),
  image_(0),
  num_channels_(0),
  bit_depth_(0),
  normalize_(false),
  min_(0.0f),
  max_(1.0f),
  init_ok_(false),
  zoom_(1.0f)
{
  //updateGL();/ // invoke OpenGL initialization
  this->initializeGL();

  IuStatus status = iu::checkCudaErrorState();
  if (status == IU_NO_ERROR)
    printf("QGLImageGpuWidget::QGLImageGpuWidget: initialized (widget + opengl).\n");
  else
    printf("QGLImageGpuWidget::QGLImageGpuWidget: error while init (widget + opengl).\n");
}

//-----------------------------------------------------------------------------
QGLImageGpuWidget::~QGLImageGpuWidget()
{

}

/* ****************************************************************************
     Input
 * ***************************************************************************/


//-----------------------------------------------------------------------------
void QGLImageGpuWidget::setImage(iu::ImageGpu_8u_C1 *image, bool normalize)
{
   printf("QGLImageGpuWidget::setImage(ImageGpu_8u_C1*)\n");

   if(image == 0)
   {
      fprintf(stderr, "The given input image is null!\n");
   }

   // FIXMEEE
  // TODO cleanup pbo and texture if we have already an image set

   if(image_ != 0)
   {
     if(image->size() == image_->size())
     {
       printf("set new image with same sizings\n");
       image_ = image;
       return;
     }
     else
     {
       printf("currently we do not support setting another image with different size.\n");
     }
   }

  image_ = image;
  num_channels_ = 1;
  bit_depth_ = 8;
  if (!this->init())
  {
    fprintf(stderr, "Failed to initialize OpenGL buffers.\n");
    init_ok_ = false;
  }
  else
    init_ok_ = true;

  if(init_ok_)
    this->resize(image_->width(), image_->height());
}

//-----------------------------------------------------------------------------
void QGLImageGpuWidget::setImage(iu::ImageGpu_8u_C4 *image, bool normalize)
{
   printf("QGLImageGpuWidget::setImage(ImageGpu_8u_C4*)\n");

   if(image == 0)
   {
      fprintf(stderr, "The given input image is null!\n");
   }

   // FIXMEEE
  // TODO cleanup pbo and texture if we have already an image set

   if(image_ != 0)
   {
     if(image->size() == image_->size())
     {
       printf("set new image with same sizings\n");
       image_ = image;
       return;
     }
     else
     {
       printf("currently we do not support setting another image with different size.\n");
     }
   }

  image_ = image;
  num_channels_ = 4;
  bit_depth_ = 8;
  if (!this->init())
  {
    fprintf(stderr, "Failed to initialize OpenGL buffers.\n");
    init_ok_ = false;
  }
  else
    init_ok_ = true;

  if(init_ok_)
    this->resize(image_->width(), image_->height());
}

//-----------------------------------------------------------------------------
void QGLImageGpuWidget::setImage(iu::ImageGpu_32f_C1 *image, bool normalize)
{
   printf("QGLImageGpuWidget::setImage(ImageGpu_32f_C1*)\n");

   if(image == 0)
   {
      fprintf(stderr, "The given input image is null!\n");
   }

   // FIXMEEE
  // TODO cleanup pbo and texture if we have already an image set

   if(image_ != 0)
   {
     if(image->size() == image_->size())
     {
       printf("set new image with same sizings\n");
       image_ = image;
       normalize_ = normalize;
       return;
     }
     else
     {
       printf("currently we do not support setting another image with different size.\n");
     }
   }

  image_ = image;
  num_channels_ = 1;
  bit_depth_ = 32;
  normalize_ = normalize;
  if (!this->init())
  {
    fprintf(stderr, "Failed to initialize OpenGL buffers.\n");
    init_ok_ = false;
  }
  else
    init_ok_ = true;

  if(init_ok_)
    this->resize(image_->width(), image_->height());
}

//-----------------------------------------------------------------------------
void QGLImageGpuWidget::setImage(iu::ImageGpu_32f_C4 *image, bool normalize)
{
   printf("QGLImageGpuWidget::setImage(ImageGpu_32f_C4*)\n");

   if(image == 0)
   {
      fprintf(stderr, "The given input image is null!\n");
   }

   // FIXMEEE
  // TODO cleanup pbo and texture if we have already an image set

   if(image_ != 0)
   {
     if(image->size() == image_->size())
     {
       printf("set new image with same sizings\n");
       image_ = image;
       return;
     }
     else
     {
       printf("currently we do not support setting another image with different size.\n");
     }
   }

  image_ = image;
  num_channels_ = 4;
  bit_depth_ = 32;
  if (!this->init())
  {
    fprintf(stderr, "Failed to initialize OpenGL buffers.\n");
    init_ok_ = false;
  }
  else
    init_ok_ = true;

  if(init_ok_)
    this->resize(image_->width(), image_->height());
}

/* ****************************************************************************
     some interaction
 * ***************************************************************************/

//-----------------------------------------------------------------------------
void QGLImageGpuWidget::setMinMax(float min, float max)
{
  min_ = min;
  max_ = max;
}

//-----------------------------------------------------------------------------
void QGLImageGpuWidget::autoMinMax()
{
  if(bit_depth_ == 8)
  {
    if(num_channels_ == 1)
    {
      iu::ImageGpu_8u_C1* img = reinterpret_cast<iu::ImageGpu_8u_C1*>(image_);
      if(img == 0) return;
      unsigned char cur_min, cur_max;
      iu::minMax(img, img->roi(), cur_min, cur_max);
      min_ = static_cast<float>(cur_min);
      max_ = static_cast<float>(cur_max);
    }
    else
    {
      iu::ImageGpu_8u_C4* img = reinterpret_cast<iu::ImageGpu_8u_C4*>(image_);
      if(img == 0) return;
      uchar4 cur_min, cur_max;
      iu::minMax(img, img->roi(), cur_min, cur_max);
      min_ = static_cast<float>(IUMIN(IUMIN(cur_min.x, cur_min.y), cur_min.z));
      max_ = static_cast<float>(IUMAX(IUMAX(cur_max.x, cur_max.y), cur_max.z));
    }
  }
  else
  {
    if(num_channels_ == 1)
    {
      iu::ImageGpu_32f_C1* img = reinterpret_cast<iu::ImageGpu_32f_C1*>(image_);
      if(img == 0) return;
      iu::minMax(img, img->roi(), min_, max_);
    }
    else
    {
      iu::ImageGpu_32f_C4* img = reinterpret_cast<iu::ImageGpu_32f_C4*>(image_);
      if(img == 0) return;
    }
  }
}

//-----------------------------------------------------------------------------
void QGLImageGpuWidget::setAutoNormalize(bool flag)
{
  normalize_ = flag;
}


/* ****************************************************************************
     GL stuff
 * ***************************************************************************/

//-----------------------------------------------------------------------------
void QGLImageGpuWidget::initializeGL()
{
  printf("QGLImageGpuWidget::initializeGL()\n");

  makeCurrent();

  glewInit();
  printf("  Loading extensions: %s\n", glewGetErrorString(glewInit()));
  if (!glewIsSupported( "GL_VERSION_1_5 GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object" ))
  {
     fprintf(stderr, "QGLImageGpuWidget Error: failed to get minimal GL extensions for QGLImageGpuWidget.\n");
     fprintf(stderr, "The widget requires:\n");
     fprintf(stderr, "  OpenGL version 1.5\n");
     fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
     fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
     fflush(stderr);
     return;
  }

  printf("QGLImageGpuWidget::initializeGL() done\n");
}

//-----------------------------------------------------------------------------
void QGLImageGpuWidget::createTexture()
{
  if(gl_tex_) deleteTexture();

  IuSize sz = image_->size();

  // Enable Texturing
  glEnable(GL_TEXTURE_2D);

  // Generate a texture identifier
  glGenTextures(1,&gl_tex_);

  // Make this the current texture (remember that GL is state-based)
  glBindTexture(GL_TEXTURE_2D, gl_tex_);

  // Allocate the texture memory. The last parameter is NULL since we only
  // want to allocate memory, not initialize it
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, sz.width, sz.height, 0, GL_BGRA,
               GL_UNSIGNED_BYTE, NULL);
#if 0
  // Must set the filter mode, GL_LINEAR enables interpolation when scaling
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
  // Note: GL_TEXTURE_RECTANGLE_ARB may be used instead of
  // GL_TEXTURE_2D for improved performance if linear interpolation is
  // not desired. Replace GL_LINEAR with GL_NEAREST in the
  // glTexParameteri() call
#else
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
#endif
}

//-----------------------------------------------------------------------------
void QGLImageGpuWidget::deleteTexture()
{
  if(gl_tex_)
  {
    glDeleteTextures(1, &gl_tex_);
    gl_tex_ = NULL;
  }
}

//-----------------------------------------------------------------------------
void QGLImageGpuWidget::createPbo()
{
  IuSize sz = image_->size();

  // set up vertex data parameter
  int num_texels = sz.width * sz.height;
  int num_values = num_texels * 4;
  int size_tex_data = sizeof(GLubyte) * num_values;

  // Generate a buffer ID called a PBO (Pixel Buffer Object)
  glGenBuffers(1,&gl_pbo_);
  // Make this the current UNPACK buffer (OpenGL is state-based)
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl_pbo_);
  // Allocate data for the buffer. 4-channel 8-bit image
  glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);

  cudaGraphicsGLRegisterBuffer( &cuda_pbo_resource_, gl_pbo_, cudaGraphicsMapFlagsNone );
}

void QGLImageGpuWidget::deletePbo()
{
  if(gl_pbo_)
  {
    // delete the PBO
    cudaGraphicsUnregisterResource(cuda_pbo_resource_);
    glDeleteBuffers( 1, &gl_pbo_ );
    gl_pbo_=NULL;
    cuda_pbo_resource_ = NULL;
  }
}

//-----------------------------------------------------------------------------
bool QGLImageGpuWidget::init()
{
  printf("QGLImageGpuWidget::init()\n");


  this->createTexture();

  if (iu::checkCudaErrorState() != IU_NO_ERROR)
    fprintf(stderr, "error while initializing texture (gl)\n");
  else
    printf("  Texture created.\n");

  this->createPbo();

  if (iu::checkCudaErrorState() != IU_NO_ERROR)
    fprintf(stderr, "error while initializing pbo (gl)\n");
  else
    printf("  PBO created.\n");

  return !iu::checkCudaErrorState();
}


//-----------------------------------------------------------------------------
void QGLImageGpuWidget::resizeGL(int w, int h)
{
  if(image_ == 0)
    return;

  float zoom_w = float(w)/float(image_->width());
  float zoom_h = float(h)/float(image_->height());
  zoom_ = (zoom_w < zoom_h) ? zoom_w : zoom_h;
}

//-----------------------------------------------------------------------------
void QGLImageGpuWidget::paintGL()
{
  if(image_ == 0)
     return;

//  printf("QGLImageGpuWidget::paintGL()\n");

  // map GL <-> CUDA resource
  uchar4 *d_dst = NULL;
  size_t start;
  cudaGraphicsMapResources(1, &cuda_pbo_resource_, 0);
  cudaGraphicsResourceGetMappedPointer((void**)&d_dst, &start, cuda_pbo_resource_);

  // check for min/max values if normalization is activated
  if(normalize_)
    this->autoMinMax();

  // get image data
  cuCopyImageToPbo(image_, num_channels_, bit_depth_, d_dst, min_, max_);
  cudaThreadSynchronize();

  // unmap GL <-> CUDA resource
  cudaGraphicsUnmapResources(1, &cuda_pbo_resource_, 0);

  // common display code path
  {
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl_pbo_);
    glBindTexture(GL_TEXTURE_2D, gl_tex_ );
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_->width(), image_->height(),
                    GL_RGBA, GL_UNSIGNED_BYTE, NULL );

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity ();
    glOrtho (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    glViewport(0, 0, (int)floor(image_->width()*zoom_), (int)floor(image_->height()*zoom_));

#if 0
    glBegin(GL_TRIANGLES);
    glTexCoord2f(0, 0); glVertex2f(-1, 1);
    glTexCoord2f(2, 0); glVertex2f( 3, 1);
    glTexCoord2f(0, 2); glVertex2f(-1,-3);
    glEnd();
#else
    glBegin(GL_QUADS);
    glTexCoord2f( 0.0, 0.0); glVertex3f(-1.0,  1.0, 0.5);
    glTexCoord2f( 1.0, 0.0); glVertex3f( 1.0,  1.0, 0.5);
    glTexCoord2f( 1.0, 1.0); glVertex3f( 1.0, -1.0, 0.5);
    glTexCoord2f( 0.0, 1.0); glVertex3f(-1.0, -1.0, 0.5);
    glEnd ();
#endif
  }
  glPopMatrix();

//  printf("QGLImageGpuWidget::paintGL() done\n");

}


} // namespace iuprivate


///////////////////////////////////////////////////////////////////////////////

namespace iu {

//-----------------------------------------------------------------------------
QGLImageGpuWidget::QGLImageGpuWidget(QWidget *parent) :
  iuprivate::QGLImageGpuWidget(parent)
{
//  //updateGL();/ // invoke OpenGL initialization
//  this->initializeGL();

//  IuStatus status = iu::checkCudaErrorState();
//  if (status == IU_NO_ERROR)
//    printf("QGLImageGpuWidget::QGLImageGpuWidget: initialized (widget + opengl).\n");
//  else
//    printf("QGLImageGpuWidget::QGLImageGpuWidget: error while init (widget + opengl).\n");
}

//-----------------------------------------------------------------------------
QGLImageGpuWidget::~QGLImageGpuWidget()
{

}

}
