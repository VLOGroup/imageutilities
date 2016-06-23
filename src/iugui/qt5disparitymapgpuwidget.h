#ifndef GLVERTEXWINDOW_H
#define GLVERTEXWINDOW_H

#include <QOpenGLWidget>
#include <QOpenGLBuffer>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLFunctions_3_3_Core>
#include <QMouseEvent>
#include <QWheelEvent>
#include "../iucore.h"


namespace iu {
/**
 * @brief The Qt5DisparityGpuWidget class allows to display disparity or depth maps in 3D with a built-in mouse and keyboard interaction
 *
 *
 */

class Qt5DisparitymapGpuWidget : public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core
{
    Q_OBJECT
public:
    /**
      * The non-default constructor
      @param sz Output image size
      @param translation_z Where to put the camera initially? [0,0,%translation_z]
      @param f Focal length of virtual camera
      @param cx x-coordinate of principal point
      @param cy y-coordinate of principal point
      @param B Baseline in the case of disparity maps. If 0, then the input is expected to be a depth map
      @param parent Parent widget
      */
    explicit Qt5DisparitymapGpuWidget(const IuSize sz, float translation_z,
                                  float f, float cx, float cy, float B=0.f, QWidget* parent = NULL);
    virtual ~Qt5DisparitymapGpuWidget();

public slots:
    /**
      * Set a new disparity/depthmap
      @param depth A depthmap in float format
      @param im A grayscale image in float format, registered to depth
      */
    void update_vertices(iu::ImageGpu_32f_C1 *depth, iu::ImageGpu_32f_C1 *im);

protected:

    void paintGL();
    void initializeGL();
    void resizeGL(int w, int h);

    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void wheelEvent(QWheelEvent* event);
    void keyPressEvent(QKeyEvent* event);

    void init_cuda();

    //iu::ImageGpu_8u_C1& img_;
    //GLuint positions_;
    cudaGraphicsResource_t cuda_positions_;
    QOpenGLBuffer vbo_;
    IuSize image_size_;

    QOpenGLShaderProgram* shader_program_;
    QOpenGLVertexArrayObject* vao_;

    QMatrix4x4 r_mat_;
    QPoint rot_pos_;
    QPoint trans_pos_;
    float translation_z_,translation_x_,translation_y_;
    float initial_translation_z_;
    float point_size_;
    bool data_set_;
    // Baseline 0 = input is a depth map already
    float f_,cx_,cy_,B_;
};

} // namespace iu

#endif // GLWINDOW_H
