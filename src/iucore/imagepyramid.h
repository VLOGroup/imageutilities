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
 * Description : Definition of image class for Gpu
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#include <math.h>

#ifndef IUCORE_IMAGEPYRAMID_H
#define IUCORE_IMAGEPYRAMID_H

namespace iu {

/** Pyramidal image data.
 *
 * :TODO: A longer description
 */
template<typename ImageType>
class ImagePyramid
{
public:
  /** Default constrcutor. */
  ImagePyramid() :
    images_(0), scale_factors_(0), num_levels_(0)
  {
  }

  /** Constructor for an image pyramid with defined number of levels, size and scale rate.
   * @param max_num_levels Defines the maximum number of levels. This is changed the shorter side gets below \a size_bound.
   * @param size The size for the finest level (=0).
   * @param rate Multiplicative scale factor.
   */
  ImagePyramid(unsigned int& max_num_levels, const IuSize& size, const float& scale_factor,
               unsigned int size_bound=1)
  {
    // TODO calculate the maximum number of levels
    if(size_bound == 0) size_bound = 1;
    unsigned int shorter_side = (size.width<size.height) ? size.width : size.height;
    float ratio = static_cast<float>(shorter_side) / static_cast<float>(size_bound);
    // +1 because the original size is level 0
    unsigned int possible_num_levels = static_cast<int>(-logf(ratio)/logf(scale_factor)) + 1;
    if(max_num_levels > possible_num_levels)
      max_num_levels = possible_num_levels;

    images_ = new ImageType*[max_num_levels];
    scale_factors_ = new float[max_num_levels];
    num_levels_ = max_num_levels;

    // init rate for each level
    for (unsigned int i=0; i<max_num_levels; i++)
    {
      scale_factors_[i] = pow(scale_factor, static_cast<float>(i));

      // alloc image memory
      IuSize sz(static_cast<int>(round(static_cast<double>(size.width)*static_cast<double>(scale_factors_[i]))),
                static_cast<int>(round(static_cast<double>(size.height)*static_cast<double>(scale_factors_[i]))));
      images_[i] = new ImageType(sz);
    }
  }

  /** Destructor. */
  virtual ~ImagePyramid()
  {
    if (images_ == 0)
      return;

    // delete all arrays and hold elements!
    for (unsigned int i=0; i<num_levels_; i++)
    {
      delete(images_[i]);
      images_[i] = 0;
    }
    delete[] images_;
    images_ = 0;
    delete[] scale_factors_;
    scale_factors_ = 0;
    num_levels_ = 0;
  }

  /** Returns a pointer to the image from the specified level \a i. */
  ImageType* image(unsigned int i) {return images_[i];}

  /** Returns the number of available levels. */
  unsigned int numLevels() {return num_levels_;}

  /** Returns the image size of the i-th level. */
  IuSize size(unsigned int i) {return images_[i]->size();}

  /** Returns the scale factor (multiplicative factor) from the current level to the base level. */
  float scaleFactor(unsigned int i) {return scale_factors_[i];}


private:
  ImageType** images_; /**< Pointer to array of (level+1) layer images (pointers - dynamically allocated). */
  float* scale_factors_;       /**< Pointer to the array of (level+1) ratios of i-th levels to the zero level (rate-i). */
  unsigned int num_levels_;          /**< Number of levels in the pyramid. */

};

} // namespace iu

#endif // IUCORE_IMAGEPYRAMID_H
