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
 * Module      : GUI
 * Class       : Overlay
 * Language    : C++
 * Description : Implementation of an Overlay for the QGLImageGpuWidget
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */


#ifndef IUPRIVATE_OVERLAY_H
#define IUPRIVATE_OVERLAY_H

#include <QString>
#include "iudefs.h"

namespace iuprivate {

class OverlayLUT
{
public:
  OverlayLUT(iu::LinearMemory* values, iu::LinearDeviceMemory_8u_C4* colors)
  {
    values_ = values;
    colors_ = colors;
  }

  iu::LinearMemory* values_;
  iu::LinearDeviceMemory_8u_C4* colors_;
};

//-----------------------------------------------------------------------------
/** \brief Overlay: An overlay class for marking different values in gpu memory.
  \ingroup iuprivate
  */
class Overlay
{
public:
  Overlay(QString& name, iu::Image* constraint_image,
          iu::LinearMemory* lut_values, iu::LinearDeviceMemory_8u_C4* lut_colors,
          bool active = true, IuComparisonOperator comp_op = IU_EQUAL) :
    name_(name),
    active_(active),
    comp_op_(comp_op)
  {
    if (constraint_image == NULL || lut_values == NULL || lut_colors == NULL)
    {
      fprintf(stderr, "An input is NULL.\n");
      return;
    }
    if (!constraint_image->onDevice() || !lut_values->onDevice())
    {
      fprintf(stderr, "Currently only device images/buffers are supported.\n");
      return;
    }

    if (lut_values->length() != lut_colors->length())
    {
      fprintf(stderr, "LUT value and color array must be of same length.\n");
      return;
    }

    constraint_image_ = constraint_image;
    lut_ = new OverlayLUT(lut_values, lut_colors);
  }

  ~Overlay()
  {
    delete lut_;
  }

  inline iu::Image* getConstraintImage() {return constraint_image_;}
  inline iu::LinearMemory* getLUTValues() {return lut_->values_;}
  inline iu::LinearDeviceMemory_8u_C4* getLUTColors() {return lut_->colors_;}
  inline QString& getName() {return name_;}
  inline bool& isActive() {return active_;}
  inline bool& toogleActive() {active_ = !active_; return active_;}
  inline IuComparisonOperator& getComparisonOperator() { return comp_op_; }

private:
  OverlayLUT* lut_;
  iu::Image* constraint_image_;
  QString name_;
  bool active_;
  IuComparisonOperator comp_op_;

};


//////////////////////////////////////////////////////////////////////////////
// typedefs
typedef QList<iuprivate::Overlay*> OverlayList;

} // namespace iuprivate



#endif // IUPRIVATE_OVERLAY_H
