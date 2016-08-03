#pragma once

//-----------------------------------------------------------------------------
/* Shared lib macros for windows dlls
*/
#ifdef _WIN32
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
#else
  #define IUCORE_DLLAPI
#endif
