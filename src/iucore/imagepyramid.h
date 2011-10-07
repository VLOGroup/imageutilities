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
 * Description : Definition of multiresolution imagepyramid
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#include "memorydefs.h"

#ifndef IUCORE_IMAGEPYRAMID_H
#define IUCORE_IMAGEPYRAMID_H

namespace iu {

/** Pyramidal image data.
 *
 * :TODO: A longer description
 */
class IUCORE_DLLAPI ImagePyramid
{
public:

  /** Default constrcutor. */
  ImagePyramid();

  /** Constructor for an image pyramid with defined number of levels, size and scale rate.
   * @param max_num_levels Defines the maximum number of levels. This is changed the shorter side gets below \a size_bound.
   * @param size The size for the finest level (=0).
   * @param rate Multiplicative scale factor.
   * @param size_bound Smaller size of coarsest level.
   */
  ImagePyramid(unsigned int& max_num_levels, const IuSize& size, const float& scale_factor,
               unsigned int size_bound=1);

  /** Destructor. */
  virtual ~ImagePyramid();

  /** Initialization of the basis pyramid elements without setting an image.
   * @param max_num_levels Defines the maximum number of levels. This is changed the shorter side gets below \a size_bound.
   * @param size The size for the finest level (=0).
   * @param rate Multiplicative scale factor.
   * @param size_bound Smaller size of coarsest level.
   * @returns Number of available levels.
   * @throw IuException
   */
  unsigned int init(unsigned int max_num_levels, const IuSize& size, const float& scale_factor,
                    unsigned int size_bound=1);

  /** Resets the image pyramid. Deletes all the data.
   */
  void reset();

  /** Sets the image data of the pyramid.
   * @params[in] image Input image representing the finest scale.
   * @returns the number of initialized pyramid levels.
   * @throw IuException
   */
  unsigned int setImage(iu::Image* image,
                        IuInterpolationType interp_type = IU_INTERPOLATE_LINEAR);
  //---------------------------------------------------------------------------
  // GETTERS / SETTERS

  /** Returns a pointer to the image from the specified level \a i. */
  inline iu::Image* image(unsigned int i) {return images_[i];}

  /** Returns the images pixel type. */
  inline IuPixelType pixelType() {return pixel_type_;}

  /** Returns the number of available levels. */
  inline unsigned int numLevels() {return num_levels_;}

  /** Returns the image size of the i-th level. */
  inline IuSize size(unsigned int i) {return images_[i]->size();}

  /** Returns the scale factor (multiplicative factor) from the current level to the base level. */
  inline float scaleFactor(unsigned int i) {return scale_factors_[i];}

  //---------------------------------------------------------------------------
  // Get specialized image types.
  // ATTENTION: Whenever a wrong function is called you get a 0-pointer!
  inline iu::ImageGpu_32f_C1* imageGpu_32f_C1(unsigned int i)
  { return reinterpret_cast<iu::ImageGpu_32f_C1*>(images_[i]); }


private:
  iu::Image** images_;      /**< Pointer to array of (level+1) layer images (pointers - dynamically allocated). */
  IuPixelType pixel_type_;  /**< The images pixel type. */
  float* scale_factors_;    /**< Pointer to the array of (level+1) ratios of i-th levels to the zero level (rate-i). */
  unsigned int num_levels_; /**< Number of levels in the pyramid. */

  unsigned int max_num_levels_; /**< Maximum number of levels set by the user. This is not necessary equal to num_levels_. */
  float scale_factor_;          /**< Scale factor from one level to the next. */
  unsigned int size_bound_;     /**< User set smaller side of coarsest level. */
};

} // namespace iu

#endif // IUCORE_IMAGEPYRAMID_H
