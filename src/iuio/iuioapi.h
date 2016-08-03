#pragma once

//-----------------------------------------------------------------------------
/* Shared lib macros for windows dlls
*/
#ifdef _WIN32
  #pragma warning( disable : 4251 ) // disable the warning about exported template code from stl
  #pragma warning( disable : 4231 ) // disable the warning about nonstandard extension in e.g. istream

  // iuio module
  #ifdef IUIO_USE_STATIC
    #define IUIO_DLLAPI
  #else
    #ifdef IUIO_DLL_EXPORTS
      #define IUIO_DLLAPI __declspec(dllexport)
    #else
      #define IUIO_DLLAPI __declspec(dllimport)
    #endif
  #endif
#else
  #define IUIO_DLLAPI
#endif

