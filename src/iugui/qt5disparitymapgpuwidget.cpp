#include "qt5disparitymapgpuwidget.h"
#include <GL/glu.h>
#include <cuda_gl_interop.h>

// OpenGL error check macro
#define GL_CHECK_ERROR() do { \
    GLenum err = glGetError(); \
    if(err != GL_NO_ERROR)     \
        printf("OpenGL error: \n File: %s\n Function: %s\n Line: %d\n", __FILE__, __FUNCTION__, __LINE__); \
    } while(0)

// CUDA error check macro
#define CUDA_CHECK_ERROR() do { \
    cudaError_t err = cudaGetLastError(); \
    if(err != cudaSuccess)     \
        printf("CUDA error: %s:%s\n File: %s\n Function: %s\n Line: %d\n", cudaGetErrorName(err), cudaGetErrorString(err), __FILE__, __FUNCTION__, __LINE__); \
    } while(0)


namespace iuprivate {
  extern void copy_to_VBO(iu::ImageGpu_32f_C1& disparitites, iu::ImageGpu_32f_C1& color, float f, float cx, float cy, float B, float point_size, iu::ImageGpu_32f_C4& vbo);
}


namespace iu {

Qt5DisparitymapGpuWidget::Qt5DisparitymapGpuWidget(const iu::Size<2> sz, float translation, float f, float cx, float cy, float B, QWidget *parent)
    : QOpenGLWidget(parent)
{
    QSurfaceFormat format;
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setMajorVersion(3);
    format.setMinorVersion(3);
    format.setSwapBehavior(QSurfaceFormat::DoubleBuffer);

    setFormat(format);

    shader_program_ = NULL;
    vao_ = NULL;
    image_size_ = sz;
    initial_translation_z_ = translation_z_ = translation;
    translation_x_ = 0.f;
    translation_y_ = 0.f;
    point_size_ = 0.01f; // meters
    f_ = f;
    cx_ = cx;
    cy_ = cy;
    B_ = B;
    data_set_ = false;

    r_mat_.setToIdentity();

    setFocusPolicy(Qt::StrongFocus);
}

Qt5DisparitymapGpuWidget::~Qt5DisparitymapGpuWidget()
{

    delete shader_program_;
    delete vao_;
}


// simple shaders
// vertex shader transforms points & passes texture coords
// fragment shader reads texture
static const char* vertex_shader_source =
        "#version 330\n"
        "uniform mat4 MVP;\n"
        "in vec4 in_vertex;\n"
        "out float color_vertex;\n"
        "void main(void) {\n"
        "  gl_Position = MVP * vec4(in_vertex.xyz,1.0);\n"
        "  color_vertex = in_vertex.w;\n"
        "}\n";
static const char* fragment_shader_source =
        "#version 330\n"
        "in float color_vertex;"
        "out vec4 color;\n"
        "void main(void) {\n"
        "  color =  vec4(color_vertex,color_vertex,color_vertex,1.0);\n"
        "}\n";
void Qt5DisparitymapGpuWidget::init_cuda()
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

    // VBO for vertices
    vbo_ = QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);
    vbo_.setUsagePattern(QOpenGLBuffer::StreamDraw);
    vbo_.create();
    vbo_.bind();
    vbo_.allocate(image_size_.width*image_size_.height*2*3*4*sizeof(float));  // enough memory for float4

    // orthographic projection over whole screen
    QMatrix4x4 mv;
    mv.translate(-100,-100,-300);
    //mv.lookAt(QVector3D(0,0,100),QVector3D(0.5f,-0.5,-1),QVector3D(0,1,0));
    QMatrix4x4 pr;
    pr.perspective(120,image_size_.width/image_size_.height,0.f,300.f);
    //mvp.ortho(-0.5, image_size_.width-0.5, image_size_.height-0.5, -0.5, -1, 300);
    shader_program_->setUniformValue("MVP", pr*mv);

    GL_CHECK_ERROR();

    // register pbo to cuda
    cudaGraphicsGLRegisterBuffer(&cuda_positions_, vbo_.bufferId(), cudaGraphicsRegisterFlagsWriteDiscard);
    CUDA_CHECK_ERROR();

}


void Qt5DisparitymapGpuWidget::initializeGL()
{
    initializeOpenGLFunctions();   // initialize the QOpenGLFunctions_x_x

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    glViewport(0, 0, image_size_.width, image_size_.height);

    resize(image_size_.width, image_size_.height);
    setMinimumSize(image_size_.width, image_size_.height);
    //setMaximumSize(image_size_.width, image_size_.height);

    init_cuda();
    GL_CHECK_ERROR();
}

void Qt5DisparitymapGpuWidget::resizeGL(int w, int h)
{
    const qreal retinascale = devicePixelRatio();
    glViewport(0, 0, w*retinascale, h*retinascale);
    //setMaximumSize(w,h);
    resize(w, h);
}

inline QVector3D getArcballVector(const QPoint& pt, int width, int height)
{
    QVector3D P = QVector3D(1.0*pt.x ()/width * 2 - 1.0,
                            1.0*pt.y ()/height * 2 - 1.0,
                            0);
    P.setY (-P.y ());

    float OP_squared = P.lengthSquared ();
    if (OP_squared <= 1)
        P.setZ (sqrt(1 - OP_squared));  // Pythagore
    else
        P.normalize ();  // nearest point
    return P;
}

void Qt5DisparitymapGpuWidget::mouseMoveEvent(QMouseEvent * e)
{
    if(e->buttons ()==Qt::LeftButton)
    {
        if(!rot_pos_.isNull () && rot_pos_!=e->pos ())
        {
            //rotate using an arcBall for freeform rotation
            QVector3D vec1 = getArcballVector (rot_pos_, width (), height ());
            QVector3D vec2 = getArcballVector (e->pos (), width (), height ());

            //use bisector to get half-angle cos and sin from dot and cross product
            // for quaternion initialisation
            QVector3D vec3 = vec1+vec2;
            vec3.normalize ();
            QVector3D rotaxis = QVector3D::crossProduct (vec1, vec3);
            double cos = QVector3D::dotProduct (vec1, vec3);

            QQuaternion quat (cos,rotaxis);
            quat.normalize ();
            //we want to left-multiply rMat with quat but that isn't available
            //so we'll have to do it the long way around
            QMatrix4x4 rot;
            rot.rotate (quat);
            r_mat_=rot*r_mat_;

            //update();
        }
        rot_pos_=e->pos ();
    }
    if(e->buttons()==Qt::RightButton)
    {
        QPoint direction = e->pos()-trans_pos_;
        if(e->modifiers() & Qt::ShiftModifier)
            translation_z_+=direction.y()/10.f;
        else
            translation_y_-=direction.y()/50.f;
        translation_x_+=direction.x()/50.f;
        trans_pos_=e->pos();
    }
}

void Qt5DisparitymapGpuWidget::wheelEvent(QWheelEvent *event)
{
    point_size_+=event->angleDelta().y()/8.f/15.f/1000.f;
    point_size_ = std::max(0.001f,point_size_);
}

void Qt5DisparitymapGpuWidget::keyPressEvent(QKeyEvent *event)
{
    if(event->key() == Qt::Key_R)
    {
        translation_x_=0;
        translation_y_=0;
        translation_z_=initial_translation_z_;
        r_mat_.setToIdentity();
    }
}

void Qt5DisparitymapGpuWidget::mousePressEvent(QMouseEvent *e)
{
    if(e->buttons ()==Qt::LeftButton)
    {
        rot_pos_=e->pos ();
    }
    if(e->buttons()==Qt::RightButton)
    {
        trans_pos_=e->pos();
    }
}

void Qt5DisparitymapGpuWidget::update_vertices(iu::ImageGpu_32f_C1 *depth,iu::ImageGpu_32f_C1 *im)
{
    if (!shader_program_)
        return;

    makeCurrent();

    cudaGraphicsMapResources(1, &cuda_positions_);   // this call is slow (~1.8ms) in case there
                                // are 2 gpus, one for display and one for computations and
                                // cuda is NOT running on the display gpu. this is possibly
                                // because in that case data needs to be transferred from the
                                // cuda gpu to the display gpu.
                                // the call is fast (~0.2ms) in case there is only one gpu.

    float4* device_ptr = NULL;
    size_t mapped_size;

    // get device pointer from opengl pbo
    cudaGraphicsResourceGetMappedPointer((void**)&device_ptr, &mapped_size, cuda_positions_);

    // make an imagepu out of it. The pbo is allcated for uchar4, compute the correct pitch here
    iu::ImageGpu_32f_C4 wrapped_vbo(device_ptr, im->width()*3*2, im->height(), mapped_size/im->height(), true);
    iuprivate::copy_to_VBO(*depth, *im, f_, cx_, cy_, B_, point_size_, wrapped_vbo);

    cudaGraphicsUnmapResources(1, &cuda_positions_);



    GL_CHECK_ERROR();
    CUDA_CHECK_ERROR();

    doneCurrent();
    data_set_ = true;
}



void Qt5DisparitymapGpuWidget::paintGL()
{
    if (!shader_program_ || !data_set_)
        return;



    shader_program_->bind();
    vao_->bind();
    // set up shaders
    int loc_vertex = shader_program_->attributeLocation("in_vertex");
    assert(loc_vertex != -1);

    glBindBuffer(GL_ARRAY_BUFFER,vbo_.bufferId());
    int stride = 4*sizeof(float);    // vertex layout is 3 float per element (x,y,z)
    glVertexAttribPointer(loc_vertex, 4, GL_FLOAT, GL_FALSE, stride, 0);
    glEnableVertexAttribArray(loc_vertex);

    QMatrix4x4 trans;
    trans.translate(translation_x_,translation_y_,translation_z_);
    QMatrix4x4 x_rot;
    x_rot.setToIdentity();
    x_rot.rotate(180,1,0,0);
    QMatrix4x4 pr;
    pr.perspective(2*atan2(image_size_.width/2,f_)/M_PI*180.0f,1.0f,0.3f,100.f);
//    pr.ortho(-image_size_.width/2, image_size_.width/2, -image_size_.height/2, image_size_.height/2, -1, 300);
    shader_program_->setUniformValue("MVP", pr*trans*r_mat_*x_rot);

    // draw the triangles
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, image_size_.width*image_size_.height*3*2);
    glDisableVertexAttribArray(loc_vertex);


    GL_CHECK_ERROR();
    CUDA_CHECK_ERROR();
}

} // namespace




