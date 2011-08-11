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

// forward declaration

/** \defgroup Gui
 *  \brief The gui module.
 *  TODO more detailed docu
 *  @{
 */

#include "iugui/imagewindow.h"
//#include "iugui/image_cpu_display.h"
#include "iugui/qglimagegpuwidget.h"

#ifdef USE_QWT
  #include "iugui/plotgapwidget.h"
#endif // USE_QWT




/** @} */ // end of Gui




#endif // IUGUI_MODULE_H
