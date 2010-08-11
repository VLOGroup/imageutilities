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
 * Module      : Global
 * Class       : none
 * Language    : C++
 * Description : Global typedefinitions and macros for ImageUtilities. (e.g. dll export stuff, ...)
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef IU_GLOBALDEFS_H
#define IU_GLOBALDEFS_H

//-----------------------------------------------------------------------------
/* Shared lib macros for windows dlls
*/
#ifdef WIN32
#pragma warning( disable : 4251 ) // disable the warning about exported template code from stl
#pragma warning( disable : 4231 ) // disable the warning about nonstandard extension in e.g. istream

  #ifdef IU_USE_STATIC
    #define IU_DLLAPI
  #else
    #ifdef IU_DLL_EXPORTS
      #define IU_DLLAPI __declspec(dllexport)
    #else
      #define IU_DLLAPI __declspec(dllimport)
    #endif
  #endif
#else
  #define IU_DLLAPI
#endif

#endif // IU_GLOBALDEFS_H
