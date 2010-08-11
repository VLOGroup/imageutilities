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
 * Module      : Gui Module
 * Class       : Wrapper
 * Language    : C
 * Description : Public interfaces to gui module
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IUGUI_MODULE_H
#define IUGUI_MODULE_H

#include "iudefs.h"

// :TODO: include namespace again if we have a cool idea how to create a public interface here
//namespace iu {

/** \defgroup Gui
 *  \brief The gui module.
 *  TODO more detailed docu
 *  @{
 */

/* ***************************************************************************
     Device (Npp) widgets
 * ***************************************************************************/

//////////////////////////////////////////////////////////////////////////////
/** @defgroup Device (Npp) widgets
 *  @ingroup Gui
 *  TODO more detailed docu
 *  @{
 */


// TODO how to define public interface for a whole class?
// :TODO: #include "gui/ippwidget.h"
#include "gui/nppglwidget.h"

/** @} */ // end of Device (Npp) widgets

/** @} */ // end of Gui

//} // namespace iu

#endif // IUGUI_MODULE_H
