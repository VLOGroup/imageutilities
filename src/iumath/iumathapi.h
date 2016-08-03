#pragma once

//-----------------------------------------------------------------------------
/* Shared lib macros for windows dlls
*/
#ifdef _WIN32
  #pragma warning( disable : 4251 ) // disable the warning about exported template code from stl
  #pragma warning( disable : 4231 ) // disable the warning about nonstandard extension in e.g. istream

  // math module
  #ifdef IUMATH_USE_STATIC
    #define IUMATH_DLLAPI
  #else
    #ifdef IUMATH_DLL_EXPORTS
      #define IUMATH_DLLAPI __declspec(dllexport)
    #else
      #define IUMATH_DLLAPI __declspec(dllimport)
    #endif
  #endif
#else
  #define IUMATH_DLLAPI
#endif
