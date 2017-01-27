#pragma once

#include <QOpenGLWidget>
#include <QOpenGLBuffer>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLFunctions_3_3_Core>
#include "../iucore.h"


namespace iu {
/**
 * @brief @brief Allows to display images directly in an OpenGL Widget.
 *
 * Images are kept on the graphics card, without the need to copy to the host for displaying.
 * @ingroup GUI
 */

class Qt5ImageGpuWidget : public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core
{
    Q_OBJECT
public:
    /**
      * The non-default constructor
      @param sz Output image size
      @param parent Parent widget
      */
    explicit Qt5ImageGpuWidget(const iu::Size<2> sz, QWidget* parent = NULL);
    virtual ~Qt5ImageGpuWidget();

public slots:
    /**
      * Set a new grayscale image
      @param im A grayscale image in uchar format
      */
    void update_image(iu::ImageGpu_8u_C1* im);
    /**
      * Set a new RGBA image
      @param im A RGBA image in uchar format
      */
    void update_image(iu::ImageGpu_8u_C4* im);
    /**
      * Set a new grayscale image
      @param im A grayscale image in float format
      @param minVal Minimum value to display-> black
      @param maxVal Maximum value to display -> white
      */
    void update_image(iu::ImageGpu_32f_C1 *im, float minVal, float maxVal);
    /**
      * Set a new RGBA image
      @param im A RGBA image in float format
      */
    void update_image(iu::ImageGpu_32f_C4* im);
    /**
      * Set a new grayscale image to be displayed in jet colormap
      @param im A grayscale image in float format
      @param minVal Minimum value to display-> black
      @param maxVal Maximum value to display -> white
      */
    void update_image_colormap(iu::ImageGpu_32f_C1 *im, float minVal, float maxVal);

protected:

    void paintGL();
    void initializeGL();
    void resizeGL(int w, int h);

    void init_cuda();

    //iu::ImageGpu_8u_C1& img_;
    GLuint texture_;
    cudaGraphicsResource_t cuda_img_;
    QOpenGLBuffer pbo_;
    iu::Size<2> image_size_;

    QOpenGLShaderProgram* shader_program_;
    QOpenGLVertexArrayObject* vao_;
    QOpenGLBuffer* vbo_;
};

} // namespace iu


