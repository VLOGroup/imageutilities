#include "qt5imagegpuwidget.h"
#include <GL/glu.h>
#include <cuda_gl_interop.h>
//#include <iu/iuio.h>

//#include "tools.h"
//#include "flow_uv.cuh"




// OpenGL error check macro
#define GL_CHECK_ERROR() do { \
    GLenum err = glGetError(); \
    if(err != GL_NO_ERROR)     \
        printf("OpenGL error:\n File: %s\n Function: %s\n Line: %d\n", __FILE__, __FUNCTION__, __LINE__); \
    } while(0)

// CUDA error check macro
#define CUDA_CHECK_ERROR() do { \
    cudaError_t err = cudaGetLastError(); \
    if(err != cudaSuccess)     \
        printf("CUDA error: %s:%s\n File: %s\n Function: %s\n Line: %d\n", cudaGetErrorName(err), cudaGetErrorString(err), __FILE__, __FUNCTION__, __LINE__); \
    } while(0)


namespace iuprivate {
  extern void copy_to_PBO(iu::ImageGpu_8u_C1& img, iu::ImageGpu_8u_C4& pbo);
  extern void copy_to_PBO(iu::ImageGpu_8u_C4& img, iu::ImageGpu_8u_C4& pbo);
  extern void copy_to_PBO(iu::ImageGpu_32f_C1& img, iu::ImageGpu_8u_C4& pbo, float minVal, float maxVal, bool colormap);
}


namespace iu {

Qt5ImageGpuWidget::Qt5ImageGpuWidget(const IuSize sz, QWidget *parent)
    : QOpenGLWidget(parent)
{
    QSurfaceFormat format;
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setMajorVersion(3);
    format.setMinorVersion(3);
    format.setSwapBehavior(QSurfaceFormat::DoubleBuffer);

    setFormat(format);

    texture_ = 0;
    cuda_img_ = 0;
    shader_program_ = NULL;
    vao_ = NULL;
    vbo_ = NULL;
    image_size_ = sz;
}

Qt5ImageGpuWidget::~Qt5ImageGpuWidget()
{

    cudaGraphicsUnregisterResource(cuda_img_);
    glDeleteTextures(1, &texture_);

    delete shader_program_;
    delete vao_;
    delete vbo_;
}


// simple shaders
// vertex shader transforms points & passes texture coords
// fragment shader reads texture
static const char* vertex_shader_source =
        "#version 330\n"
        "uniform mat4 MVP;\n"
        "in vec3 in_vertex;\n"
        "in vec2 in_texUV;\n"
        "out vec2 texUV;\n"
        "void main(void) {\n"
        "  gl_Position = MVP * vec4(in_vertex, 1.0);\n"
        "  texUV = in_texUV;\n"
        "}\n";
static const char* fragment_shader_source =
        "#version 330\n"
        "uniform sampler2D texture;\n"
        "in vec2 texUV;\n"
        "out vec4 color;\n"
        "void main(void) {\n"
        "  color =  texture2D(texture, texUV);\n"
        "}\n";
void Qt5ImageGpuWidget::init_cuda()
{
    if (shader_program_)
    {
        printf("Error: cuda widget already initialized!\n");
        return;
    }

    /** TODO
     *  check if opengl context is properly created (3.x core profile context)
     */

    shader_program_ = new QOpenGLShaderProgram(this);
    if (!shader_program_->addShaderFromSourceCode(QOpenGLShader::Vertex, vertex_shader_source))
        printf("Error: %s\n", shader_program_->log().toStdString().c_str());
    if (!shader_program_->addShaderFromSourceCode(QOpenGLShader::Fragment, fragment_shader_source))
        printf("Error: %s\n", shader_program_->log().toStdString().c_str());
    shader_program_->link();
    shader_program_->bind();

    // VAO for the screen sized quad
    vao_ = new QOpenGLVertexArrayObject(this);
    vao_->create();
    vao_->bind();

    // PBO for texture
    pbo_ = QOpenGLBuffer(QOpenGLBuffer::PixelUnpackBuffer);
    pbo_.setUsagePattern(QOpenGLBuffer::StreamDraw);
    pbo_.create();
    pbo_.bind();
    pbo_.allocate(image_size_.width*image_size_.height*4*sizeof(unsigned char));  // enough memory for uchar4

    // vertex and texture coordinate data for a quad (2 triangles)
    // layout: x,y,z,u,v
    float vertices_and_texUV[6*(3+2)] = {-0.5f, -0.5f, 0, 0, 0,
            image_size_.width-0.5f, -0.5f, 0, 1, 0,
            image_size_.width-0.5f, image_size_.height-0.5f, 0, 1, 1,
            -0.5f, image_size_.height-0.5f, 0, 0, 1,
            image_size_.width-0.5f, image_size_.height-0.5f, 0, 1, 1,
            -0.5f, -0.5f, 0, 0, 0};
    vbo_ = new QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);
    vbo_->setUsagePattern(QOpenGLBuffer::StaticDraw);
    vbo_->create();
    vbo_->bind();
    vbo_->allocate(vertices_and_texUV, sizeof(vertices_and_texUV));

    // set up shaders
    int loc_vertex = shader_program_->attributeLocation("in_vertex");
    int loc_tex = shader_program_->attributeLocation("in_texUV");
    assert(loc_vertex != -1);
    assert(loc_tex != -1);

    int stride = 5*sizeof(float);    // vertex layout is 5 float per element (x,y,z,u,v)
    glVertexAttribPointer(loc_vertex, 3, GL_FLOAT, GL_FALSE, stride, 0);
    glVertexAttribPointer(loc_tex, 2, GL_FLOAT, GL_FALSE, stride, (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(loc_vertex);
    glEnableVertexAttribArray(loc_tex);

    // create texture
    glGenTextures(1, &texture_);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_size_.width, image_size_.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    // texture unit 0
    shader_program_->setUniformValue("texture", 0);

    // orthographic projection over whole screen
    QMatrix4x4 mvp;
    mvp.ortho(-0.5, image_size_.width-0.5, image_size_.height-0.5, -0.5, -1, 1);
    shader_program_->setUniformValue("MVP", mvp);

    GL_CHECK_ERROR();

    // register pbo to cuda
    cudaGraphicsGLRegisterBuffer(&cuda_img_, pbo_.bufferId(), cudaGraphicsRegisterFlagsWriteDiscard);
    CUDA_CHECK_ERROR();

}


void Qt5ImageGpuWidget::initializeGL()
{
    initializeOpenGLFunctions();   // initialize the QOpenGLFunctions_x_x

    glClearColor(1.0, 0.0, 0.0, 1.0);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    glViewport(0, 0, image_size_.width, image_size_.height);

    resize(image_size_.width, image_size_.height);
    setMinimumSize(image_size_.width, image_size_.height);
    setMaximumSize(image_size_.width, image_size_.height);

    init_cuda();
    GL_CHECK_ERROR();
}

void Qt5ImageGpuWidget::resizeGL(int w, int h)
{
    Q_UNUSED(w);
    Q_UNUSED(h);
}


void Qt5ImageGpuWidget::update_image(iu::ImageGpu_8u_C1 *im)
{
    if (!shader_program_)
        return;

    makeCurrent();

    cudaGraphicsMapResources(1, &cuda_img_);   // this call is slow (~1.8ms) in case there
                                // are 2 gpus, one for display and one for computations and
                                // cuda is NOT running on the display gpu. this is possibly
                                // because in that case data needs to be transferred from the
                                // cuda gpu to the display gpu.
                                // the call is fast (~0.2ms) in case there is only one gpu.

    uchar4* device_ptr = NULL;
    size_t mapped_size;

    // get device pointer from opengl pbo
    cudaGraphicsResourceGetMappedPointer((void**)&device_ptr, &mapped_size, cuda_img_);

    // make an imagepu out of it. The pbo is allcated for uchar4, compute the correct pitch here
    iu::ImageGpu_8u_C4 wrapped_pbo(device_ptr, im->width(), im->height(), mapped_size/im->height(), true);
    iuprivate::copy_to_PBO(*im, wrapped_pbo);

    cudaGraphicsUnmapResources(1, &cuda_img_);

    // make texture from pbo
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, im->width(), im->height(), GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    GL_CHECK_ERROR();
    CUDA_CHECK_ERROR();

    doneCurrent();
}

void Qt5ImageGpuWidget::update_image(ImageGpu_8u_C4 *im)
{
    if (!shader_program_)
        return;

    makeCurrent();

    cudaGraphicsMapResources(1, &cuda_img_);   // this call is slow (~1.8ms) in case there
                                // are 2 gpus, one for display and one for computations and
                                // cuda is NOT running on the display gpu. this is possibly
                                // because in that case data needs to be transferred from the
                                // cuda gpu to the display gpu.
                                // the call is fast (~0.2ms) in case there is only one gpu.

    uchar4* device_ptr = NULL;
    size_t mapped_size;
    cudaGraphicsResourceGetMappedPointer((void**)&device_ptr, &mapped_size, cuda_img_);

    iu::ImageGpu_8u_C4 wrapped_pbo(device_ptr, im->width(), im->height(), mapped_size/im->height(), true);
    iuprivate::copy_to_PBO(*im, wrapped_pbo);

    cudaGraphicsUnmapResources(1, &cuda_img_);

    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, im->width(), im->height(), GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    GL_CHECK_ERROR();
    CUDA_CHECK_ERROR();

    doneCurrent();
    update();
}

void Qt5ImageGpuWidget::update_image(ImageGpu_32f_C1 *im, float minVal, float maxVal)
{
    if (!shader_program_)
        return;

    makeCurrent();

    cudaGraphicsMapResources(1, &cuda_img_);   // this call is slow (~1.8ms) in case there
                                // are 2 gpus, one for display and one for computations and
                                // cuda is NOT running on the display gpu. this is possibly
                                // because in that case data needs to be transferred from the
                                // cuda gpu to the display gpu.
                                // the call is fast (~0.2ms) in case there is only one gpu.

    uchar4* device_ptr = NULL;
    size_t mapped_size;
    cudaGraphicsResourceGetMappedPointer((void**)&device_ptr, &mapped_size, cuda_img_);

    iu::ImageGpu_8u_C4 wrapped_pbo(device_ptr, im->width(), im->height(), mapped_size/im->height(), true);
    iuprivate::copy_to_PBO(*im, wrapped_pbo,minVal,maxVal,false);

    cudaGraphicsUnmapResources(1, &cuda_img_);

    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, im->width(), im->height(), GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    GL_CHECK_ERROR();
    CUDA_CHECK_ERROR();

    doneCurrent();
    update();
}

void Qt5ImageGpuWidget::update_image_colormap(ImageGpu_32f_C1 *im, float minVal, float maxVal)
{
    if (!shader_program_)
        return;

    makeCurrent();

    cudaGraphicsMapResources(1, &cuda_img_);   // this call is slow (~1.8ms) in case there
                                // are 2 gpus, one for display and one for computations and
                                // cuda is NOT running on the display gpu. this is possibly
                                // because in that case data needs to be transferred from the
                                // cuda gpu to the display gpu.
                                // the call is fast (~0.2ms) in case there is only one gpu.

    uchar4* device_ptr = NULL;
    size_t mapped_size;
    cudaGraphicsResourceGetMappedPointer((void**)&device_ptr, &mapped_size, cuda_img_);

    iu::ImageGpu_8u_C4 wrapped_pbo(device_ptr, im->width(), im->height(), mapped_size/im->height(), true);
    iuprivate::copy_to_PBO(*im, wrapped_pbo,minVal,maxVal,true);

    cudaGraphicsUnmapResources(1, &cuda_img_);

    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, im->width(), im->height(), GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    GL_CHECK_ERROR();
    CUDA_CHECK_ERROR();

    doneCurrent();
    update();
}


void Qt5ImageGpuWidget::paintGL()
{
    if (!shader_program_)
        return;

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    shader_program_->bind();
    vao_->bind();
    glDrawArrays(GL_TRIANGLES, 0, 6);

    GL_CHECK_ERROR();
    CUDA_CHECK_ERROR();
}

} // namespace




