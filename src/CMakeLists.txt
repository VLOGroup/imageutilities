# Copyright (c) ICG. All rights reserved.
#
# Institute for Computer Graphics and Vision
# Graz University of Technology / Austria
#
#
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.  See the above copyright notices for more information.
#
#
# Project     : ImageUtilities
# Module      : CMakeFile
# Language    : CMake
# Description : CMakeFile
#
# Author     : Manuel Werlberger
# EMail      : werlberger@icg.tugraz.at

project(ImageUtilities CXX C)
#set(CMAKE_BUILD_TYPE Debug)
cmake_minimum_required(VERSION 2.8.0)

# cleanup
set(${IU_PUBLIC_HEADERS} "")

# general settings
set( CMAKE_DEBUG_POSTFIX "d" )
set( NPP64 "" )
if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
  set( NPP64 "-x64" )
endif( CMAKE_SIZEOF_VOID_P EQUAL 8 )

FILE(TO_CMAKE_PATH "$ENV{CMAKE_MODULE_PATH}" OT_CMAKE_MODULE_PATH)
FILE(TO_CMAKE_PATH "$ENV{ICG_CMAKE_MODULE_PATH}" OT_ICG_CMAKE_MODULE_PATH)
SET(CMAKE_MODULE_PATH ${OT_CMAKE_MODULE_PATH} ${OT_ICG_CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/../cmake)

##-----------------------------------------------------------------------------
## Current version number:
##-----------------------------------------------------------------------------
set(IMAGEUTILITIES_VERSION "1.1.0")

string(REGEX MATCHALL "[0-9]" IMAGEUTILITIES_VERSION_PARTS "${IMAGEUTILITIES_VERSION}")

list(GET IMAGEUTILITIES_VERSION_PARTS 0 IMAGEUTILITIES_VERSION_MAJOR)
list(GET IMAGEUTILITIES_VERSION_PARTS 1 IMAGEUTILITIES_VERSION_MINOR)
list(GET IMAGEUTILITIES_VERSION_PARTS 2 IMAGEUTILITIES_VERSION_PATCH)
set(IMAGEUTILITIES_SOVERSION "${IMAGEUTILITIES_VERSION_MAJOR}.${IMAGEUTILITIES_VERSION_MINOR}")

if(WIN32)
    # Postfix of DLLs:
    set(IMAGEUTILITIES_DLLVERSION "${IMAGEUTILITIES_VERSION_MAJOR}${IMAGEUTILITIES_VERSION_MINOR}${IMAGEUTILITIES_VERSION_PATCH}")
    set(IMAGEUTILITIES_DEBUG_POSTFIX d)
else()
    # Postfix of so's:
    #set(IMAGEUTILITIES_DLLVERSION "${IMAGEUTILITIES_VERSION_MAJOR}${IMAGEUTILITIES_VERSION_MINOR}${IMAGEUTILITIES_VERSION_PATCH}")
    set(IMAGEUTILITIES_DLLVERSION "")
    set(IMAGEUTILITIES_DEBUG_POSTFIX)
endif()

##-----------------------------------------------------------------------------
## Flags if modules are used.
## The modules are included per default unless dependencies not found or omitted by the user.
## (the core module is always included!)
OPTION(VMLIBRARIES_IU_USE_GUI "Including GUI module." ON)
OPTION(VMLIBRARIES_IU_USE_IO "Including IO module." ON)
OPTION(VMLIBRARIES_IU_USE_VIDEOCAPTURE "Including VideCapture IO module." ON)
OPTION(VMLIBRARIES_IU_USE_PGRCAMERA "Including PointGray IO module." ON)
OPTION(VMLIBRARIES_IU_BUILD_COMBINED_LIBRARY "Building a complete library package. This is just for compatibility reasons." OFF)


##-----------------------------------------------------------------------------
## Compiler Flags
if(UNIX)
  # enable all warnings
  add_definitions(-Wall)
  # and debug flags
  add_definitions(-g)
endif()

IF(WIN32)
  ADD_DEFINITIONS(-DWIN32)
  # for msvc 8 deprecated warning removal
  ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE)
  ADD_DEFINITIONS(-D_CRT_NONSTDC_NO_DEPRECATE)
  ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE_GLOBALS)

  SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT.lib")
endif(WIN32)

##-----------------------------------------------------------------------------
# CUDA
find_package(CUDA 3.1 REQUIRED)
find_package(CUDASDK REQUIRED)
cuda_include_directories(${CUDA_INCLUDE_DIRS} ${CUDA_CUT_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIRS} ${CUDA_CUT_INCLUDE_DIR})
set(IU_IUCORE_LIB_DEPENDENCIES ${IU_IUCORE_LIB_DEPENDENCIES} ${CUDA_LIBRARIES})

##-----------------------------------------------------------------------------
# NPP
find_package(NPP REQUIRED)
cuda_include_directories(${NPP_INCLUDE_DIRS})
include_directories(${NPP_INCLUDE_DIRS})

##-----------------------------------------------------------------------------
## IO MODULE: OpenCV for Image I/O
find_package( OpenCV QUIET )
# only include if GUI module is used and qt4 found.
if(VMLIBRARIES_IU_USE_IO AND OpenCV_INCLUDE_DIRS)

  message("[+] ImageUtilities include IO module")
  include_directories(${OpenCV_INCLUDE_DIRS})

  set( IU_PUBLIC_HEADERS
    ${IU_PUBLIC_HEADERS}
    ${CMAKE_CURRENT_SOURCE_DIR}/iuio.h
    )

  set( IU_IO_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/iuio/imageio.h
    )

  set( IU_IO_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/iuio.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/iuio/imageio.cpp
    )

else()
  message("[-] ImageUtilities omitting IO module")
  set(VMLIBRARIES_IU_USE_IO OFF CACHE PATH "IU: Omit IO module." FORCE)

  # empty libraries
  set(OpenCV_LIBS "")

endif()

##-----------------------------------------------------------------------------
## Qt
find_package(Qt4 COMPONENTS QtCore QtGui QtOpenGL)
# only include if GUI module is used and qt4 found.
if(VMLIBRARIES_IU_USE_GUI AND QT4_FOUND)

  message("[+] ImageUtilities include GUI module")
  include(${QT_USE_FILE})


  ## GLEW
  find_package( GLEW REQUIRED )
  include_directories(${GLEW_INCLUDE_DIR})

  ## OpenGL
  find_package( OpenGL REQUIRED )
  include_directories(${OPENGL_INCLUDE_DIR})


  set( IU_PUBLIC_HEADERS
    ${IU_PUBLIC_HEADERS}
    ${CMAKE_CURRENT_SOURCE_DIR}/iugui.h
    )

  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/iugui/nppglwidget.h ${CMAKE_CURRENT_SOURCE_DIR}/iugui/nppglwidget_p.h
    DESTINATION include/iu/iugui
    COMPONENT Headers)

  set( IU_GUI_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/iugui/nppglwidget.cuh
    )

  set( IU_GUI_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/iugui/nppglwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/iugui/nppglwidget.cu
    )

  set( IU_GUI_MOC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/iugui/nppglwidget_p.h
    ${CMAKE_CURRENT_SOURCE_DIR}/iugui/nppglwidget.h
    )

  # compile moc headers
  QT4_WRAP_CPP (IU_GUI_MOC_SOURCES ${IU_GUI_MOC_HEADERS})

else()
  message("[-] ImageUtilities omitting GUI module")
  set(VMLIBRARIES_IU_USE_GUI OFF CACHE PATH "IU: Omit GUI module." FORCE)

  # empty libraries
  set(QT_LIBRARIES "")
  set(OPENGL_LIBRARIES "")
  set(GLEW_LIBRARIES "")

endif()



##-----------------------------------------------------------------------------
## CAMERA/Video IO MODULE: OpenCV IO with Qt threading
# NOTE: we need qt4 for that (threading, ...)
# only include if all dependencies were found
if(VMLIBRARIES_IU_USE_VIDEOCAPTURE AND QT4_FOUND AND OpenCV_INCLUDE_DIRS)

  message("[+] ImageUtilities include video/camera module (OpenCV/Qt needed)")
  include(${QT_USE_FILE})

  set( IU_PUBLIC_HEADERS
    ${IU_PUBLIC_HEADERS}
    ${CMAKE_CURRENT_SOURCE_DIR}/iuvideocapture.h

    )

  set( IU_VIDEOCAPTURE_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/iuio/videocapturethread.h
    ${CMAKE_CURRENT_SOURCE_DIR}/iuio/videocapture.h
    )

  set( IU_VIDEOCAPTURE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/iuvideocapture.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/iuio/videocapturethread.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/iuio/videocapture.cpp
    )

  set( IU_VIDEOCAPTURE_MOC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/iuio/videocapturethread.h
#    ${CMAKE_CURRENT_SOURCE_DIR}/iuio/videocapture.h
    )

  # compile moc headers
  QT4_WRAP_CPP (IU_VIDEOCAPTURE_MOC_SOURCES ${IU_VIDEOCAPTURE_MOC_HEADERS})

else()
  message("[-] ImageUtilities omitting VIDEOCAPTURE module")
  set(VMLIBRARIES_IU_USE_VIDEOCAPTURE OFF CACHE PATH "IU: Omit videocapture module." FORCE)

  # empty libraries
  set(FLYCAPTURE2_LIBRARIES "")

endif()

##-----------------------------------------------------------------------------
## IO MODULE: PointGray cameras (flycapture2);
# NOTE: we need qt4 for that (threading, ...)
find_package( FlyCapture2 QUIET )
# only include if PGR specific stuff was found
if(VMLIBRARIES_IU_USE_PGRCAMERA AND QT4_FOUND AND FLYCAPTURE2_FOUND)

  message("[+] ImageUtilities include PGRCAMERA module")
  include(${QT_USE_FILE})
  include_directories(${FLYCAPTURE2_INCLUDE_DIR})
  message("FLYCAPTURE_INCLUDE_DIR=${FLYCAPTURE2_INCLUDE_DIR}")

  set( IU_PUBLIC_HEADERS
    ${IU_PUBLIC_HEADERS}
    ${CMAKE_CURRENT_SOURCE_DIR}/iupgrcamera.h
    )

  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/iuio/pgrcamera.h
    DESTINATION include/iu/iuio
    COMPONENT Headers)

  set( IU_PGRCAMERA_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/iuio/pgrcameradata.h
    ${CMAKE_CURRENT_SOURCE_DIR}/iuio/pgrcameraimagegrabber.h
    ${CMAKE_CURRENT_SOURCE_DIR}/iuio/pgrcamerathreaddata.h
    )

  set( IU_PGRCAMERA_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/iuio/pgrcamera.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/iuio/pgrcameraimageconsumer.cpp
    )

  set( IU_PGRCAMERA_MOC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/iuio/pgrcamera.h
    ${CMAKE_CURRENT_SOURCE_DIR}/iuio/pgrcameraimageconsumer.h
    )

  # compile moc headers
  QT4_WRAP_CPP (IU_PGRCAMERA_MOC_SOURCES ${IU_PGRCAMERA_MOC_HEADERS})

else()
  message("[-] ImageUtilities omitting PGRCAMERA module")
  set(VMLIBRARIES_IU_USE_PGRCAMERA OFF CACHE PATH "IU: Omit PGRCAMERA module." FORCE)

  # empty libraries
  set(FLYCAPTURE2_LIBRARIES "")

endif()

##-----------------------------------------------------------------------------
## local settings, headers and sources
include_directories(
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  )
cuda_include_directories(
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  )

# SPLITTING ALL THE HEADERS FOR THE INSTALL DIRECTORY!
SET( IU_PUBLIC_HEADERS
  ${IU_PUBLIC_HEADERS}
  ${CMAKE_CURRENT_SOURCE_DIR}/iudefs.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucutil.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iumath.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iufilter.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucoreippconnector.h
  )

SET( IU_PUBLIC_CORE_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/globaldefs.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/coredefs.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/memorydefs.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/linearhostmemory.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/lineardevicememory.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/image.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/image_allocator_cpu.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/image_allocator_npp.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/image_cpu.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/image_npp.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../../common/include/derivatives.h
  )

# all the additional headers follow here below (we split that two for better overview and the seperated libs)
SET( IU_CORE_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/copy.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/memory_modification.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/memory_modification.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/iutextures.cuh
  )
SET( IU_MATH_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/iumath/arithmetic.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iumath/arithmetic.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/iumath/statistics.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iumath/statistics.cuh
  )

SET( IU_FILTER_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/iufilter/filter.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iufilter/filter.cuh
  )

SET( IU_TRANSFORM_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/iutransform/resample.h
  #${CMAKE_CURRENT_SOURCE_DIR}/iutransform/resample.cuh
  )


SET( IU_CONNECTOR_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/iuconnector/copy_ipp.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iuconnector/image_allocator_ipp.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iuconnector/image_ipp.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iuconnector/memorydefs_ipp.h
  )

SET( IU_HEADERS
  ${IU_PUBLIC_HEADERS}
  ${IU_PUBLIC_CORE_HEADERS}
  ${IU_CORE_HEADERS}
  ${IU_MATH_HEADERS}
  ${IU_FILTER_HEADERS}
  #${IU_TRANSFORM_HEADERS}
  )

SET( IU_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/memory_modification.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/memory_modification.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/memory_modification_kernels.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/iumath.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iumath/arithmetic.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iumath/arithmetic.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/iumath/statistics.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iumath/statistics.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/iufilter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iufilter/filter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iufilter/filter.cu
  #${CMAKE_CURRENT_SOURCE_DIR}/iutransform/transform.cpp
  #${CMAKE_CURRENT_SOURCE_DIR}/iutransform/transform.cu
  #${CMAKE_CURRENT_SOURCE_DIR}/iutransform/transform_kernels.cu
  )

# denote kernel files as header only to not confuse the compiler
set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/memory_modification_kernels.cu
  PROPERTIES HEADER_FILE_ONLY yes
  )

##-----------------------------------------------------------------------------
set( IU_ALL_FILES
  ${IU_HEADERS}
  ${IU_GUI_HEADERS}
  ${IU_IO_HEADERS}
  ${IU_VIDEOCAPTURE_HEADERS}
  ${IU_PGRCAMERA_HEADERS}
  ${IU_CONNECTOR_HEADERS}
  ${IU_SOURCES}
  ${IU_GUI_SOURCES}
  ${IU_GUI_MOC_SOURCES}
  ${IU_IO_SOURCES}
  ${IU_VIDEOCAPTURE_SOURCES}
  ${IU_VIDEOCAPTURE_MOC_SOURCES}
  ${IU_PGRCAMERA_SOURCES}
  ${IU_PGRCAMERA_MOC_SOURCES}
  )

IF(NOT WIN32)
  SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC"  )
  SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_DEBUG} -fPIC"  )
  SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC"  )
  SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_DEBUG} -fPIC"  )
endif()

set( IU_LIBS "") # variable where all libs that are built are saved

##-----------------------------------------------------------------------------
## build core library
cuda_add_library( iucore SHARED ${IU_HEADERS} ${IU_SOURCES})
set_target_properties(iucore PROPERTIES
  DEFINE_SYMBOL IU_DLL_EXPORTS
  VERSION ${IMAGEUTILITIES_VERSION}
  SOVERSION ${IMAGEUTILITIES_SOVERSION}
  PUBLIC_HEADER "${IU_PUBLIC_HEADERS}"
  )
target_link_libraries( iucore ${NPP_LIBRARIES} ${CUDA_LIBRARIES} )
set(IU_LIBS ${IU_LIBS} iucore)

##-----------------------------------------------------------------------------
## build gui library
if(VMLIBRARIES_IU_USE_GUI)
  cuda_add_library( iugui SHARED ${IU_GUI_HEADERS} ${IU_GUI_SOURCES} ${IU_GUI_MOC_SOURCES} )
  set_target_properties(iugui PROPERTIES
    DEFINE_SYMBOL IU_DLL_EXPORTS
    VERSION ${IMAGEUTILITIES_VERSION}
    SOVERSION ${IMAGEUTILITIES_SOVERSION}
    )
  target_link_libraries( iugui
    ${NPP_LIBRARIES} ${CUDA_LIBRARIES}
    ${QT_LIBRARIES}  ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES}
    )
  set(IU_LIBS ${IU_LIBS} iugui)

endif(VMLIBRARIES_IU_USE_GUI)

##-----------------------------------------------------------------------------
## build io library
if(VMLIBRARIES_IU_USE_IO)
  cuda_add_library( iuio SHARED ${IU_IO_HEADERS} ${IU_IO_SOURCES} )
  set_target_properties(iuio PROPERTIES
    DEFINE_SYMBOL IU_DLL_EXPORTS
    VERSION ${IMAGEUTILITIES_VERSION}
    SOVERSION ${IMAGEUTILITIES_SOVERSION}
    )
  target_link_libraries( iuio
    ${NPP_LIBRARIES} ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    )
  set(IU_LIBS ${IU_LIBS} iuio)
endif(VMLIBRARIES_IU_USE_IO)

##-----------------------------------------------------------------------------
## build VIDEOCAPTURE library
if(VMLIBRARIES_IU_USE_VIDEOCAPTURE)
  cuda_add_library( iuvideocapture SHARED ${IU_VIDEOCAPTURE_HEADERS} ${IU_VIDEOCAPTURE_SOURCES} ${IU_VIDEOCAPTURE_MOC_SOURCES})
  set_target_properties(iuvideocapture PROPERTIES
    DEFINE_SYMBOL IU_DLL_EXPORTS
    VERSION ${IMAGEUTILITIES_VERSION}
    SOVERSION ${IMAGEUTILITIES_SOVERSION}
    )
  target_link_libraries( iuvideocapture
    ${NPP_LIBRARIES} ${CUDA_LIBRARIES}
    ${QT_LIBRARIES}  ${OpenCV_LIBS}
    )
  set(IU_LIBS ${IU_LIBS} iuvideocapture)

endif(VMLIBRARIES_IU_USE_VIDEOCAPTURE)


##-----------------------------------------------------------------------------
## build pgrcamera library
if(VMLIBRARIES_IU_USE_PGRCAMERA)
  cuda_add_library( iupgrcamera SHARED ${IU_PGRCAMERA_HEADERS} ${IU_PGRCAMERA_SOURCES} ${IU_PGRCAMERA_MOC_SOURCES})
  set_target_properties(iupgrcamera PROPERTIES
    DEFINE_SYMBOL IU_DLL_EXPORTS
    VERSION ${IMAGEUTILITIES_VERSION}
    SOVERSION ${IMAGEUTILITIES_SOVERSION}
    )
  target_link_libraries( iupgrcamera
    ${NPP_LIBRARIES} ${CUDA_LIBRARIES}
    ${QT_LIBRARIES}  ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES}
    ${FLYCAPTURE2_LIBRARIES}
    )
  set(IU_LIBS ${IU_LIBS} iupgrcamera)

endif(VMLIBRARIES_IU_USE_PGRCAMERA)

##-----------------------------------------------------------------------------
## build combined package (everything included - for compatibility)
if(VMLIBRARIES_IU_BUILD_COMBINED_LIBRARY)
  cuda_add_library( iu SHARED ${IU_ALL_FILES} )
  set_target_properties(iu PROPERTIES
    DEFINE_SYMBOL IU_DLL_EXPORTS
    VERSION ${IMAGEUTILITIES_VERSION}
    SOVERSION ${IMAGEUTILITIES_SOVERSION}
    )

  if( NOT WIN32 )
    cuda_add_library( iu_static STATIC ${IU_ALL_FILES} )
    set_target_properties(iu_static PROPERTIES
      DEFINE_SYMBOL IU_DLL_EXPORTS
      VERSION ${IMAGEUTILITIES_VERSION}
      SOVERSION ${IMAGEUTILITIES_SOVERSION}
      )
  endif()

  if( WIN32 )
    set_target_properties( iu PROPERTIES LINK_FLAGS_RELEASE "/LTCG" )
  endif( WIN32 )

  set( IU_ALL_SHARED_LIBS
    ${NPP_LIBRARIES} ${CUDA_LIBRARIES}
    ${QT_LIBRARIES}  ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${OpenCV_LIBS} ${FLYCAPTURE2_LIBRARIES}
    )
  TARGET_LINK_LIBRARIES(iu ${IU_ALL_SHARED_LIBS} )

  if( NOT WIN32 )
    set( IU_ALL_STATIC_LIBS
      ${NPP_LIBRARIES} ${CUDA_LIBRARIES}
      ${QT_LIBRARIES}  ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${OpenCV_LIBS} ${FLYCAPTURE2_LIBRARIES}
      )
    TARGET_LINK_LIBRARIES(iu_static ${IU_ALL_STATIC_LIBS} )
  endif()
  set(IU_LIBS ${IU_LIBS} iu)

endif(VMLIBRARIES_IU_BUILD_COMBINED_LIBRARY)

##-----------------------------------------------------------------------------
## Install target

## Two different types of installation supported:
## *) local: for developement (e.g. svn checkout -- currently the default)
## *) global: for usage only (care about mixing up the two -> the global one will be preffered!)

OPTION(VMLIBRARIES_IU_INSTALL_LOCAL "Installation of the ImageUtilities binaries in the local vmlibraries tree." ON)

if(VMLIBRARIES_IU_INSTALL_LOCAL )
  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX $ENV{VMLIBRARIES_ROOT} CACHE PATH "ImageUtilities install prefix" FORCE)
  endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
endif(VMLIBRARIES_IU_INSTALL_LOCAL)

if(NOT VMLIBRARIES_IU_INSTALL_LOCAL OR NOT CMAKE_INSTALL_PREFIX)
  if(NOT UNIX)
    set(CMAKE_INSTALL_PREFIX $ENV{ProgramFiles}/VMLibraries CACHE PATH "ImageUtilities install prefix" FORCE)
  else(NOT UNIX)
    set(CMAKE_INSTALL_PREFIX /usr/local CACHE PATH "ImageUtilities install prefix" FORCE)
  endif(NOT UNIX)
endif(NOT VMLIBRARIES_IU_INSTALL_LOCAL OR NOT CMAKE_INSTALL_PREFIX)

# set the cmake-files that should be installed with the package
set( IU_CMAKE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/FindImageUtilities.cmake
  ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/UseImageUtilities.cmake
  )

# install the libraries
install( TARGETS ${IU_LIBS}
  RUNTIME         DESTINATION   bin
  ARCHIVE         DESTINATION   lib
  LIBRARY         DESTINATION   lib
  PUBLIC_HEADER   DESTINATION   include/iu
  )
install(FILES ${IU_PUBLIC_CORE_HEADERS}
  DESTINATION include/iu/iucore
  COMPONENT Headers
  )
install(FILES ${IU_CONNECTOR_HEADERS}
  DESTINATION include/iu/iuconnector
  COMPONENT Headers
  )
install(FILES ${IU_CMAKE_FILES}
  DESTINATION cmake
  )
#install common directory
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../common
  DESTINATION .
  PATTERN ".svn" EXCLUDE
  )


##-----------------------------------------------------------------------------
## CPACK configuration
INCLUDE(InstallRequiredSystemLibraries)

SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "ImageUtilities - Connecting the world of CPU and GPU.")
SET(CPACK_PACKAGE_VENDOR "Manuel Werlberger <werlberger@icg.tugraz.at>")
SET(CPACK_PACKAGE_CONTACT "Manuel Werlberger <werlberger@icg.tugraz.at>")
SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "Manuel Werlberger <werlberger@icg.tugraz.at>")
# SET(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/ReadMe.txt")
# SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/Copyright.txt")
SET(CPACK_PACKAGE_VERSION_MAJOR "${IMAGEUTILITIES_VERSION_MAJOR}")
SET(CPACK_PACKAGE_VERSION_MINOR "${IMAGEUTILITIES_VERSION_MINOR}")
SET(CPACK_PACKAGE_VERSION_PATCH "${IMAGEUTILITIES_VERSION_PATH}")
SET(CPACK_PACKAGE_INSTALL_DIRECTORY "ImageUtilities ${IMAGEUTILITIES_VERSION_MAJOR}.${IMAGEUTILITIES_VERSION_MINOR}")

# IF(WIN32 AND NOT UNIX)
#   # There is a bug in NSI that does not handle full unix paths properly. Make
#   # sure there is at least one set of four (4) backlasshes.
#   SET(CPACK_PACKAGE_ICON "${CMake_SOURCE_DIR}/Utilities/Release\\\\InstallIcon.bmp")
#   SET(CPACK_NSIS_INSTALLED_ICON_NAME "bin\\\\MyExecutable.exe")
#   SET(CPACK_NSIS_DISPLAY_NAME "${CPACK_PACKAGE_INSTALL_DIRECTORY} My Famous Project")
#   SET(CPACK_NSIS_HELP_LINK "http:\\\\\\\\www.my-project-home-page.org")
#   SET(CPACK_NSIS_URL_INFO_ABOUT "http:\\\\\\\\www.my-personal-home-page.com")
#   SET(CPACK_NSIS_CONTACT "me@my-personal-home-page.com")
#   SET(CPACK_NSIS_MODIFY_PATH ON)
# ELSE(WIN32 AND NOT UNIX)
#   SET(CPACK_STRIP_FILES "bin/MyExecutable")
#   SET(CPACK_SOURCE_STRIP_FILES "")
# ENDIF(WIN32 AND NOT UNIX)
# SET(CPACK_PACKAGE_EXECUTABLES "MyExecutable" "My Executable")
INCLUDE(CPack)
