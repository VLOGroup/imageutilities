#pragma once

//-----------------------------------------------------------------------------
/* Shared lib macros for windows dlls
*/
#ifdef _WIN32
  #pragma warning( disable : 4251 ) // disable the warning about exported template code from stl
  #pragma warning( disable : 4231 ) // disable the warning about nonstandard extension in e.g. istream

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
#else
  #define IUGUI_DLLAPI
#endif

