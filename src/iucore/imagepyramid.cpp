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
  max_num_levels = this->init(max_num_levels, size, scale_factor, size_bound);
}

//---------------------------------------------------------------------------
ImagePyramid::~ImagePyramid()
{
  this->reset();
}

//---------------------------------------------------------------------------
unsigned int ImagePyramid::init(unsigned int max_num_levels, const IuSize& size,
                                const float& scale_factor, unsigned int size_bound)
{
  if ((scale_factor <= 0) || (scale_factor >=1))
  {
    throw IuException("scale_factor out of range; must be in interval ]0,1[.", __FILE__, __FUNCTION__, __LINE__);
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
  }

  return num_levels_;
}

//---------------------------------------------------------------------------
/** Resets the image pyramid. Deletes all the data.
   */
void ImagePyramid::reset()
{
  if(images_ != 0)
  {
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
unsigned int ImagePyramid::setImage(iu::Image* image,
                                    IuInterpolationType interp_type)
{
  if (image == 0)
  {
    throw IuException("Input image is NULL.", __FILE__, __FUNCTION__, __LINE__);
  }
  if (!image->onDevice())
  {
    throw IuException("Currently only device images supported.", __FILE__, __FUNCTION__, __LINE__);
  }

  if ((images_ != 0) && (
        (images_[0]->size() != image->size()) ||
        (images_[0]->pixelType() != image->pixelType()) ))
  {
    this->reset();
    this->init(max_num_levels_, image->size(), scale_factor_, size_bound_);
  }

  pixel_type_ = image->pixelType();
  switch (pixel_type_)
  {
  case IU_32F_C1:
  {
    // *** needed so that always the same mem is used (if already existent)
    iu::ImageGpu_32f_C1*** cur_images = reinterpret_cast<iu::ImageGpu_32f_C1***>(&images_);
    if (images_ == 0)
    {
      (*cur_images) = new iu::ImageGpu_32f_C1*[num_levels_];
      for (unsigned int i=0; i<num_levels_; i++)
      {
        IuSize sz(static_cast<int>(floor(0.5+static_cast<double>(image->width())*static_cast<double>(scale_factors_[i]))),
                  static_cast<int>(floor(0.5+static_cast<double>(image->height())*static_cast<double>(scale_factors_[i]))));
        (*cur_images)[i] = new iu::ImageGpu_32f_C1(sz);
      }
    }
    iuprivate::copy(reinterpret_cast<iu::ImageGpu_32f_C1*>(image), (*cur_images)[0]);
    for (unsigned int i=1; i<num_levels_; i++)
    {
      iuprivate::reduce((*cur_images)[i-1], (*cur_images)[i], interp_type, 1, 0);
    }
    break;
  }
  default:
    throw IuException("Unsupported pixel type. currently supported: 32f_C1", __FILE__, __FUNCTION__, __LINE__);
  }

  return num_levels_;
}

} // namespace iu

