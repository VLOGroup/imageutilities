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
 * Module      : Core
 * Class       : ImagePyramid
 * Language    : C++
 * Description : Implementation of multiresolution imagepyramid
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#include <math.h>
#include "coredefs.h"
#include "memorydefs.h"
#include "iutransform/reduce.h"
#include "copy.h"
#include "imagepyramid.h"

namespace iu {

//---------------------------------------------------------------------------
ImagePyramid::ImagePyramid() :
  images_(0), pixel_type_(IU_UNKNOWN_PIXEL_TYPE), scale_factors_(0), num_levels_(0),
  max_num_levels_(0), scale_factor_(0.0f), size_bound_(0)
{
}

//---------------------------------------------------------------------------
ImagePyramid::ImagePyramid(unsigned int& max_num_levels, const IuSize& size, const float& scale_factor,
             unsigned int size_bound) :
  images_(0), pixel_type_(IU_UNKNOWN_PIXEL_TYPE), scale_factors_(0), num_levels_(0),
  max_num_levels_(0), scale_factor_(0.0f), size_bound_(0)
{
  this->init(max_num_levels, size, scale_factor, size_bound);
}

//---------------------------------------------------------------------------
ImagePyramid::~ImagePyramid()
{
  this->reset();
}

//---------------------------------------------------------------------------
bool ImagePyramid::init(unsigned int max_num_levels, const IuSize& size,
                        const float& scale_factor, unsigned int size_bound)
{
  if ((scale_factor <= 0) || (scale_factor >=1))
  {
    fprintf(stderr, "ImagePyramid::init: scale_factor must be in the interval (0,1). Init failed.");
    return 0;
  }

  if (images_ != 0)
    this->reset();

  max_num_levels_ = IUMAX(1u, max_num_levels);
  num_levels_ = max_num_levels_;
  size_bound_ = IUMAX(1u, size_bound);

  // calculate the maximum number of levels
  unsigned int shorter_side = (size.width<size.height) ? size.width : size.height;
  float ratio = static_cast<float>(shorter_side)/static_cast<float>(size_bound_);
  // +1 because the original size is level 0
  unsigned int possible_num_levels = static_cast<int>(
        -logf(ratio)/logf(scale_factor)) + 1;
  if(num_levels_ > possible_num_levels)
    num_levels_ = possible_num_levels;

  // init rate for each level
  scale_factors_ = new float[num_levels_];
  for (unsigned int i=0; i<num_levels_; i++)
  {
    scale_factors_[i] = pow(scale_factor, static_cast<float>(i));
    printf("scale_factors[%d]=%f\n", i, scale_factors_[i]);
  }

  return num_levels_;
}

//---------------------------------------------------------------------------
/** Resets the image pyramid. Deletes all the data.
   */
void ImagePyramid::reset()
{
  printf("reset\n");
  if(images_ != 0)
  {
    printf("delete images[i]\n");
    // delete all arrays and hold elements!
    for (unsigned int i=0; i<num_levels_; i++)
    {
      delete(images_[i]);
      images_[i] = 0;
    }
  }

  delete[] images_;
  images_ = 0;
  pixel_type_ = IU_UNKNOWN_PIXEL_TYPE;
  delete[] scale_factors_;
  scale_factors_ = 0;
  num_levels_ = 0;
}

//---------------------------------------------------------------------------
void ImagePyramid::setImage(iu::Image* image,
                            IuInterpolationType interp_type)
{
  if (image == 0)
  {
    fprintf(stderr, "ImagePyramid::setImage: input image is 0.");
    return;
  }

  if ((images_ != 0) && (
        (images_[0]->size() != image->size()) ||
        (images_[0]->pixelType() != image->pixelType()) ))
  {
    this->reset();
    this->init(max_num_levels_, image->size(), scale_factor_, size_bound_);
  }

  pixel_type_ = image->pixelType();
  printf("ImagePyramid::setImage: image pixel_type = %d\n", image->pixelType());
  printf("ImagePyramid::setImage: pyr pixel_type = %d\n", pixel_type_);
  printf("ImagePyramid::setImage: pyr pixel_type = %d\n", this->pixelType());

  switch (pixel_type_)
  {
  case IU_32F_C1:
  {
    printf("32f_C1\n");
    printf("images=%p\n", images_);
    iu::ImageGpu_32f_C1** cur_images = 0;
    printf("cur_images=%p\n", cur_images);
    if (images_ == 0)
    {
      printf("create array\n");
      cur_images = new iu::ImageGpu_32f_C1*[num_levels_];
    }
    else
      cur_images = reinterpret_cast<iu::ImageGpu_32f_C1**>(images_);
    images_ = reinterpret_cast<iu::Image**>(cur_images);
    printf("cur_images=%p\n", cur_images);
    printf("images=%p\n", images_);
    cur_images[0] = new iu::ImageGpu_32f_C1(image->size());
    iuprivate::copy(reinterpret_cast<iu::ImageGpu_32f_C1*>(image), cur_images[0]);
    printf("1\n");

    for (unsigned int i=1; i<num_levels_; i++)
    {
      IuSize sz(static_cast<int>(floor(0.5+static_cast<double>(image->width())*static_cast<double>(scale_factors_[i]))),
                static_cast<int>(floor(0.5+static_cast<double>(image->height())*static_cast<double>(scale_factors_[i]))));
      printf("i=%d: sz=%d/%d\n", i, sz.width, sz.height);

      cur_images[i] = new iu::ImageGpu_32f_C1(sz);
      iuprivate::reduce(reinterpret_cast<iu::ImageGpu_32f_C1*>(cur_images[i-1]),
                        reinterpret_cast<iu::ImageGpu_32f_C1*>(cur_images[i]),
                        interp_type, 1, 0);
    }
    printf("2\n");
    break;
  }
  default:
    fprintf(stderr, "ImagePyramid::setImage: unsupported pixel type (currently only 32F_C1 supported)\n");
    return;
  }
}

} // namespace iu

