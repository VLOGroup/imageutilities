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
 * Module      : IO;
 * Class       : none
 * Language    : C++
 * Description : Implementation of image I/O functions for pgm images
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#include <math.h>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>

#include <iucore.h>

#include "imageiopgm.h"

namespace iuprivate {

/* ****************************************************************************

   helper functions first

*/

  /** Skips a comment. (used for parsing input image)
 * @param src Input stream
 */
void skipComment(std::istream &src)
{
  while(src.peek() == '#')
    src.ignore(1000000, '\n');
}


/* ****************************************************************************

   imread

 */

//-----------------------------------------------------------------------------
iu::ImageCpu_16u_C1* imread_16u_C1(const std::string& filename)
{
  std::ifstream src(filename.c_str(), std::ios::binary);

  if(!src)
  {
    std::cerr << "imread_16u_C1: Couldn't open image file \"" << filename << "\"" << std::endl;
    return 0;
  }

  skipComment(src);
  std::string type;
  src >> type >> std::ws;

  int planes;
  if(type == "P5")
  {
    planes = 1;
  }
  else if(type == "P6")
  {
    planes = 3;
    std::cerr << "imread_16u_C1: Color images not supported yet." << std::endl;
    return 0;
  }
  else
  {
    std::cerr << "imread_16u_C1: File \"" << filename
        << "\" is not a binary gray or color image (PPM type P5 or P6)" << std::endl;
    return 0;
  }

  // get dimensions and maxvalue
  int width, height, maxval;
  skipComment(src);
  src >> width >> height >> maxval;

  int bit_depth;
  if(maxval <= 255)
  {
    std::cout << "Loading 8-bit file..." << std::endl;
    return 0;
    bit_depth = 8;
  }
  else if ( (maxval > 255) && (maxval <= 65535) )
  {
//    std::cout << "Loading 16-bit file..." << std::endl;
    bit_depth = 16;
  }
  else
  {
    std::cerr << "CudaPnmImage::CudaPnmImage(): Only 8 or 16 bit per channel supported" << std::endl;
    return 0;
  }

  // read one additonal '\n' !!!
  char tmp[16];
  src.read(tmp,1);

  // Allocate memory
  iu::ImageCpu_16u_C1* image = new iu::ImageCpu_16u_C1(width, height);
  src.read((char*)image->data(), image->width() * image->height() * planes * 2);
  return image;
}

//-----------------------------------------------------------------------------
iu::ImageCpu_32f_C1* imread_16u32f_C1(const std::string& filename, int max_val)
{
  iu::ImageCpu_16u_C1* im_16u_C1 = iuprivate::imread_16u_C1(filename);
  iu::ImageCpu_32f_C1* im_32f_C1 = new iu::ImageCpu_32f_C1(im_16u_C1->size());
  float normalization_factor = 1.0f/static_cast<float>(max_val);
  iu::convert_16u32f_C1(im_16u_C1, im_32f_C1, normalization_factor, 0.0f);
  delete(im_16u_C1);
  return im_32f_C1;
}


//-----------------------------------------------------------------------------
iu::ImageGpu_32f_C1* imread_cu16u32f_C1(const std::string& filename, int max_val)
{
  iu::ImageCpu_32f_C1* image_32f_C1 = iuprivate::imread_16u32f_C1(filename, max_val);
  iu::ImageGpu_32f_C1* image = new iu::ImageGpu_32f_C1(image_32f_C1->size());
  iu::copy(image_32f_C1, image);
  delete(image_32f_C1);
  return image;
}

} // namespace iuprivate
