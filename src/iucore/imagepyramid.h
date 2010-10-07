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
    images_(0), rates_(0), nlevels_(0)
  {
  }

  /** Constructor for an image pyramid with defined number of levels, size and scale rate. */
  ImagePyramid(int& nlevels, IuSize size, float rate)
  {
    images_ = new ImageType*[nlevels];
    rates_ = new float[nlevels];
    nlevels_ = nlevels;

    // init rate for each level
    float cur_rate = 1;
    for (int i=0; i<nlevels; i++)
    {
      rates_[i] = cur_rate;
      cur_rate *= rate;
    }
  }

  /** Destructor. */
  virtual ~ImagePyramid()
  {
    if(images_ == 0)
      return;

    // delete all arrays and hold elements!
    delete[] images_;
    images_ = 0;
    delete[] rates_;
    rates_ = 0;
    nlevels_ = 0;
  }

  /** Returns the image size of the i-th level. */
  IuSize size(int i) {return images_[i]->size();}

  /** Returns the scale factor (multiplicative factor) from the current level to the base level. */
  float getRate(int i) {return rates_[i];}

private:
  ImageType** images_; /**< Pointer to array of (level+1) layer images (pointers - dynamically allocated). */
  float* rates_;       /**< Pointer to the array of (level+1) ratios of i-th levels to the zero level (rate-i). */
  int nlevels_;          /**< Number of levels in the pyramid. */

};

} // namespace iu

#endif // IUCORE_IMAGEPYRAMID_H
