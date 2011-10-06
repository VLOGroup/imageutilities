#ifndef DRAW_CU
#define DRAW_CU

#include <iucore/coredefs.h>
#include <iucore/memorydefs.h>
#include <iucutil.h>

namespace iuprivate {

////////////////////////////////////////////////////////////////////////////////
//! Template for drawing lines
//! @param input          pointer to input image
//! @param width          width of the image
//! @param height         height of the image
//! @param offset_x       offset of grid in x direction
//! @param offset_y       offset of grid in y direction
//! @param x_start        x coordinate of line starting point
//! @param y_start        y coordinate of line starting point
//! @param x_end          x coordinate of line end point
//! @param y_end          y coordinate of line end point
//! @param line_width     half line width
//! @param intensity      intenity value with which will be drawn
////////////////////////////////////////////////////////////////////////////////
template<class T>
__global__ void drawLineKernel(T* input,
                               int x_start, int y_start, int x_end, int y_end,
                               int offset_x, int offset_y, int width, int height, int pitch,
                               float line_width, T intensity)
{
  // calculate absolute texture coordinates
  int x = blockIdx.x*blockDim.x + threadIdx.x + offset_x;
  int y = blockIdx.y*blockDim.y + threadIdx.y + offset_y;

  int center = y*pitch+x;

  if ((x<width) && (y<height) && (x>=0) && (y>=0))
  {
    // draw start
    float distance = (((float)x-(float)x_start)*((float)x-(float)x_start)+
                      ((float)y-(float)y_start)*((float)y-(float)y_start));
    if (distance < line_width)
      input[center] = intensity;

    // draw end
    distance = (((float)x-(float)x_end)*((float)x-(float)x_end)+
                ((float)y-(float)y_end)*((float)y-(float)y_end));
    if (distance < line_width)
      input[center] = intensity;

    float x1 = (float)x_start;
    float y1 = (float)y_start;
    float x2 = (float)x_end;
    float y2 = (float)y_end;

    float u = ((x-x1)*(x2-x1) + (y-y1)*(y2-y1))/((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));

    // draw line segment
    if (!((u < 0.00f) || (u > 1.00f)))
    {
      float intersect_x = x1+u*(x2-x1);
      float intersect_y = y1+u*(y2-y1);
      distance = (intersect_x-x)*(intersect_x-x) + (intersect_y-y)*(intersect_y-y);
      if (distance < line_width)
        input[center] = intensity;
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
//! Draw a line to input image
//! @param image       The image to which will be drawn
//! @param x_start     First coordinate x
//! @param y_start     First coordinate y
//! @param x_end       Second coordinate x
//! @param y_end       Second coordinate y
//! @param line_width  Width of the line
//! @param value       Intensity to draw with
////////////////////////////////////////////////////////////////////////////////
void CUDAdrawLine(iu::Image* image, int x_start, int y_start,
                  int x_end, int y_end, int line_width, float value)
{
  // adapt brush size_x
  int brush_size = line_width*line_width;

  // extract some variables
  int width = image->width();
  int height = image->height();

  // Calculate area that was drawn on
  int min_x = IUMIN(IUMAX(IUMIN(x_start, x_end) - brush_size, 0), width);
  int min_y = IUMIN(IUMAX(IUMIN(y_start, y_end) - brush_size, 0), height);
  int max_x = IUMIN(IUMAX(IUMAX(x_start, x_end) + brush_size, 0), width);
  int max_y = IUMIN(IUMAX(IUMAX(y_start, y_end) + brush_size, 0), height);
  int size_x = max_x - min_x;
  int size_y = max_y - min_y;

  if ((size_x == 0) || (size_y == 0))
    return; // outside

  // fragmentation
  unsigned int block_size = 16;
  dim3 dimBlock(block_size, block_size);
  dim3 dimGrid(iu::divUp(size_x, dimBlock.x), iu::divUp(size_y, dimBlock.y));

  // Call drawing algorithm
  switch (image->pixelType())
  {
  case  IU_8U_C1:
  {
    unsigned char val = static_cast<unsigned char>(value);
    iu::ImageGpu_8u_C1* im = dynamic_cast<iu::ImageGpu_8u_C1*>(image);
    drawLineKernel<<<dimGrid, dimBlock>>>(im->data(),
                                          x_start, y_start, x_end, y_end,
                                          min_x, min_y, width, height, image->stride(),
                                          brush_size, val);
    break;
  }
  case  IU_32F_C1:
  {
    iu::ImageGpu_32f_C1* im = dynamic_cast<iu::ImageGpu_32f_C1*>(image);
    drawLineKernel <<< dimGrid, dimBlock >>> (im->data(),
                                              x_start, y_start, x_end, y_end,
                                              min_x, min_y, width, height, image->stride(),
                                              brush_size, value);
    break;
  }
  default:
    throw IuException("Unsupported PixelType.", __FILE__, __FUNCTION__, __LINE__);
  }
  IU_CUDA_CHECK();
}

} // namespace iuprivate

#endif //DRAW_CU
