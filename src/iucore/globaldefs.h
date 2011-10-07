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

// core module
#ifdef IUCORE_USE_STATIC
  #define IUCORE_DLLAPI
#else
  #ifdef IUCORE_DLL_EXPORTS
    #define IUCORE_DLLAPI __declspec(dllexport)
  #else
    #define IUCORE_DLLAPI __declspec(dllimport)
  #endif
#endif

// io module
#ifdef IUIO_USE_STATIC
  #define IUIO_DLLAPI
#else
  #ifdef IUIO_DLL_EXPORTS
    #define IUIO_DLLAPI __declspec(dllexport)
  #else
    #define IUIO_DLLAPI __declspec(dllimport)
  #endif
#endif

// iopgm module
#ifdef IUIOPGM_USE_STATIC
  #define IUIOPGM_DLLAPI
#else
  #ifdef IUIOPGM_DLL_EXPORTS
    #define IUIOPGM_DLLAPI __declspec(dllexport)
  #else
    #define IUIOPGM_DLLAPI __declspec(dllimport)
  #endif
#endif

// gui module
#ifdef IUGUI_USE_STATIC
  #define IUGUI_DLLAPI
#else
  #ifdef IUGUI_DLL_EXPORTS
    #define IUGUI_DLLAPI __declspec(dllexport)
  #else
    #define IUGUI_DLLAPI __declspec(dllimport)
  #endif
#endif

// ipp module
#ifdef IUIPP_USE_STATIC
  #define IUIPP_DLLAPI
#else
  #ifdef IUIPP_DLL_EXPORTS
    #define IUIPP_DLLAPI __declspec(dllexport)
  #else
    #define IUIPP_DLLAPI __declspec(dllimport)
  #endif
#endif

// matlab module
#ifdef IUMATLAB_USE_STATIC
  #define IUMATLAB_DLLAPI
#else
  #ifdef IUMATLAB_DLL_EXPORTS
    #define IUMATLAB_DLLAPI __declspec(dllexport)
  #else
    #define IUMATLAB_DLLAPI __declspec(dllimport)
  #endif
#endif


#else
  #define IUCORE_DLLAPI
  #define IUIO_DLLAPI
  #define IUIOPGM_DLLAPI
  #define IUGUI_DLLAPI
  #define IUIPP_DLLAPI
  #define IUMATLAB_DLLAPI
#endif

#endif // IU_GLOBALDEFS_H
