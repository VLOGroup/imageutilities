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
# Author     : Gottfried Graber
# EMail      : graber@icg.tugraz.at

project(imageutilities_lib)
cmake_minimum_required(VERSION 2.8)

# cleanup
set(${IU_PUBLIC_HEADERS} "")
option(IU_BUILD_STATIC "Build static libs too." OFF)

# general settings
#set(CMAKE_DEBUG_POSTFIX "d")
set(CMAKE_BUILD_TYPE Release)

file(TO_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake" OT_IU_CMAKE_MODULE_PATH)
set(CMAKE_MODULE_PATH ${OT_IU_CMAKE_MODULE_PATH})


##-----------------------------------------------------------------------------
## Current version number:
##-----------------------------------------------------------------------------
set(IMAGEUTILITIES_VERSION "2.0.0")

string(REGEX MATCHALL "[0-9]" IMAGEUTILITIES_VERSION_PARTS "${IMAGEUTILITIES_VERSION}")

list(GET IMAGEUTILITIES_VERSION_PARTS 0 IMAGEUTILITIES_VERSION_MAJOR)
list(GET IMAGEUTILITIES_VERSION_PARTS 1 IMAGEUTILITIES_VERSION_MINOR)
list(GET IMAGEUTILITIES_VERSION_PARTS 2 IMAGEUTILITIES_VERSION_PATCH)
set(IMAGEUTILITIES_SOVERSION "${IMAGEUTILITIES_VERSION_MAJOR}.${IMAGEUTILITIES_VERSION_MINOR}${IMAGEUTILITIES_VERSION_PATCH}")
set(IMAGEUTILITIES_DLLVERSION "${IMAGEUTILITIES_VERSION_MAJOR}${IMAGEUTILITIES_VERSION_MINOR}${IMAGEUTILITIES_VERSION_PATCH}")

if(WIN32)
    # Postfix of DLLs:
    #set(IMAGEUTILITIES_DEBUG_POSTFIX d)
else()
    # Postfix of so's:
    #set(IMAGEUTILITIES_DLLVERSION "${IMAGEUTILITIES_VERSION_MAJOR}${IMAGEUTILITIES_VERSION_MINOR}${IMAGEUTILITIES_VERSION_PATCH}")
    #set(IMAGEUTILITIES_DLLVERSION "")
    #set(IMAGEUTILITIES_DEBUG_POSTFIX)
endif()

##-----------------------------------------------------------------------------
## Flags if modules are used.
## The modules are included per default unless dependencies not found or omitted by the user.
## (the core module is always included!)

if(ANDROID)
option(IU_USE_IO "Including IO module." ON)
option(IU_USE_GUI "Including GUI module." OFF)
option(WITH_FLYCAPTURE2 "Use FlyCapture2." OFF)
option(WITH_OPENEXR "Use OpenEXR." OFF)
set(ANDROID_TOOLCHAIN_NAME arm-linux-androideabi-4.6)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
else(ANDROID)
option(IU_USE_IO "Including IO module." ON)
option(IU_USE_GUI "Including GUI module." ON)
option(WITH_FLYCAPTURE2 "Use FlyCapture2." ON)
option(WITH_OPENEXR "Use OpenEXR." ON)
endif(ANDROID)

##-----------------------------------------------------------------------------
## Compiler Flags (to be added on demand)


##-----------------------------------------------------------------------------
# CUDA
if(ANDROID)
    find_package(CUDAAndroid 5.0 REQUIRED)
else(ANDROID)
    find_package(CUDA 5.0 REQUIRED)    # texture objects are introduced in 5.0
endif(ANDROID)

cuda_include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
set(IU_IUCORE_LIB_DEPENDENCIES ${IU_IUCORE_LIB_DEPENDENCIES} ${CUDA_LIBRARIES})

# set compute capability from environment variable (directly useable as nvcc flag)
if(NOT ANDROID)
if("$ENV{COMPUTE_CAPABILITY}" MATCHES "1.1")
  message("A minimum of compute capability 3.0 and CUDA 5.0 is needed!")
elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "1.2")
  message("A minimum of compute capability 3.0 and CUDA 5.0 is needed!")
elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "1.3")
  message("A minimum of compute capability 3.0 and CUDA 5.0 is needed!")
elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "2.0")
  message("A minimum of compute capability 3.0 and CUDA 5.0 is needed!")
elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "2.1")
  message("A minimum of compute capability 3.0 and CUDA 5.0 is needed!")
elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "3.0")
  set(CUDA_NVCC_FLAGS -arch=sm_30)
elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "3.2")
  set(CUDA_NVCC_FLAGS -arch=sm_32)
elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "3.5")
  set(CUDA_NVCC_FLAGS -arch=sm_35)
elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "3.7")
  set(CUDA_NVCC_FLAGS -arch=sm_37)
elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "5.0")
  set(CUDA_NVCC_FLAGS -arch=sm_50)
elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "5.2")
  set(CUDA_NVCC_FLAGS -arch=sm_52)
else()
  set(CUDA_NVCC_FLAGS -arch=sm_30)
endif()
else(NOT ANDROID)
  set(CUDA_NVCC_FLAGS -arch=sm_30 -target-cpu-arch=ARM -target-os-variant=Android)
endif(NOT ANDROID)



##-----------------------------------------------------------------------------
## IO MODULE: OpenCV for Image I/O
if(ANDROID)
    set(OpenCV_DIR $ENV{NVPACK_CUDA_ROOT_DIR}/../OpenCV-2.4.8.2-Tegra-sdk/sdk/native/jni)
endif(ANDROID)
find_package( OpenCV QUIET COMPONENTS opencv_core opencv_highgui)
# only include if IO module is used and OpenCV found.
if(IU_USE_IO AND OpenCV_LIBS)

  message("[+] ImageUtilities include IO module")
  include_directories(${OpenCV_INCLUDE_DIRS})

  set( IU_PUBLIC_HEADERS
    ${IU_PUBLIC_HEADERS}
    ${CMAKE_CURRENT_SOURCE_DIR}/iuio.h
    ${CMAKE_CURRENT_SOURCE_DIR}/iuvideosource.h
    )

  set( IU_IO_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/iuio/imageio.h
    ${CMAKE_CURRENT_SOURCE_DIR}/iuio/videosource.h
    #${CMAKE_CURRENT_SOURCE_DIR}/iuio/videocapture_private.h
    )
  set(IU_PUBLIC_IO_HEADERS
      ${CMAKE_CURRENT_SOURCE_DIR}/iuio/videosource.h
      ${CMAKE_CURRENT_SOURCE_DIR}/iuio/opencvsource.h
   )

  set( IU_IO_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/iuio.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/iuio/imageio.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/iuio/opencvsource.cpp
    )
  message("Found OpenCV Version: " ${OpenCV_VERSION_MAJOR} "." ${OpenCV_VERSION_MINOR})
  # check for OpenCV greaterequal 2.4
  if(${OpenCV_VERSION_MAJOR} MATCHES "3")
    message("Enabling fixes for OpenCV >= 3")
    add_definitions(-DOPENCV_3)
    find_package( OpenCV QUIET COMPONENTS opencv_videoio opencv_core opencv_highgui)
  endif(${OpenCV_VERSION_MAJOR} MATCHES "3")


  ######################
  # Flycapture2 for pointgrey cameras
  ######################
  if(WITH_FLYCAPTURE2)
    find_package(FlyCapture2 REQUIRED)
    if(FLYCAPTURE2_FOUND)
      message("  [+] build io module with Flycapture2/Pointgrey support")
      add_definitions(-DIUIO_PGR)
      include_directories(${FLYCAPTURE2_INCLUDE_DIR})

      set( IU_IO_HEADERS ${IU_IO_HEADERS}
        ${CMAKE_CURRENT_SOURCE_DIR}/iuio/pgrsource.h
       )
      set(IU_PUBLIC_IO_HEADERS ${IU_PUBLIC_IO_HEADERS}
        ${CMAKE_CURRENT_SOURCE_DIR}/iuio/pgrsource.h
        )
      set( IU_IO_SOURCES ${IU_IO_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/iuio/pgrsource.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/iuio/pgrcameradata.h
       )
      set(ADDITIONAL_IUIO_DEPENDENCIES
         ${FLYCAPTURE2_LIBRARIES}
        )
    
    endif(FLYCAPTURE2_FOUND)
  else(WITH_FLYCAPTURE2)
      message("  [-] build io module without Flycapture2/Pointgrey support")
      set(ADDITIONAL_IUIO_DEPENDENCIES "")
  endif(WITH_FLYCAPTURE2)

  ######################
  # OpenEXR
  ######################
  if(WITH_OPENEXR)
    find_package(OpenEXR REQUIRED)
    if(OPENEXR_FOUND)
        message("  [+] build io module with OpenEXR support")
        include_directories(${OPENEXR_INCLUDE_DIRS})
        add_definitions(-DIUIO_EXR)

        find_package(Eigen3 QUIET)
        if(EIGEN3_FOUND)
            add_definitions(-DIUIO_EIGEN3)
        endif(EIGEN3_FOUND)


        set( IU_IO_HEADERS ${IU_IO_HEADERS}
          ${CMAKE_CURRENT_SOURCE_DIR}/iuio/openexrio.h
         )
        set(IU_PUBLIC_IO_HEADERS ${IU_PUBLIC_IO_HEADERS}
          ${CMAKE_CURRENT_SOURCE_DIR}/iuio/openexrio.h
          )
        set( IU_IO_SOURCES ${IU_IO_SOURCES}
          ${CMAKE_CURRENT_SOURCE_DIR}/iuio/openexrio.cpp
         )

        set(ADDITIONAL_IUIO_DEPENDENCIES ${ADDITIONAL_IUIO_DEPENDENCIES}
            ${OPENEXR_LIBRARIES}
          )

    
    endif(OPENEXR_FOUND)
  else(WITH_OPENEXR)
        message("  [-] build io module without OpenEXR support")
  endif(WITH_OPENEXR)


else()
  message("[-] ImageUtilities omitting IO module")
  set(IU_USE_IO OFF CACHE PATH "IU: Omit IO module." FORCE)

  # empty libraries
  set(OpenCV_LIBS "")

endif()



##-----------------------------------------------------------------------------
## GUI MODULE: Qt5 for ImagepuWidget
find_package(Qt5Core 5.4)
find_package(Qt5Widgets 5.4)
find_package(Qt5OpenGL 5.4)
find_package(OpenGL)
if(IU_USE_GUI AND Qt5Widgets_FOUND AND Qt5OpenGL_FOUND)

  message("[+] ImageUtilities include GUI module")
  set(CMAKE_AUTOMOC ON)

  set( IU_PUBLIC_HEADERS
    ${IU_PUBLIC_HEADERS}
    ${CMAKE_CURRENT_SOURCE_DIR}/iugui.h
   )
  set(IU_PUBLIC_GUI_HEADERS
      ${CMAKE_CURRENT_SOURCE_DIR}/iugui/qt5imagegpuwidget.h
      ${CMAKE_CURRENT_SOURCE_DIR}/iugui/qt5disparitymapgpuwidget.h
   )
  set( IU_GUI_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/iugui.h
    ${CMAKE_CURRENT_SOURCE_DIR}/iugui/qt5imagegpuwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/iugui/qt5disparitymapgpuwidget.h
  )
  set( IU_GUI_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/iugui/qt5imagegpuwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/iugui/qt5disparitymapgpuwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/iugui/qt5imagegpuwidget.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/iugui/qt5disparitymapgpuwidget.cu
  )


else(IU_USE_GUI AND Qt5Widgets_FOUND AND Qt5OpenGL_FOUND)
  message("[-] ImageUtilities omitting GUI module")
  set(IU_USE_GUI OFF CACHE PATH "IU: Omit GUI module." FORCE)

endif(IU_USE_GUI AND Qt5Widgets_FOUND AND Qt5OpenGL_FOUND)


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

# SPLITTING ALL THE HEADERS FOR THE INSTALL DIRECTORIES (PUBLIC) AND HEADERS
# THAT ARE NOT INSTALLED (public not as keyword)!
SET( IU_PUBLIC_HEADERS
  ${IU_PUBLIC_HEADERS}
  ${CMAKE_CURRENT_SOURCE_DIR}/iudefs.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucutil.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore.h
  ${CMAKE_CURRENT_SOURCE_DIR}/helper_math.h
#   ${CMAKE_CURRENT_SOURCE_DIR}/iumath.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iufilter.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iutransform.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iupython.h
  )
# 
SET( IU_PUBLIC_CORE_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/globaldefs.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/coredefs.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/memorydefs.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/linearmemory.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/linearhostmemory.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/lineardevicememory.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/tensor_cpu.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/tensor_gpu.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/image.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/image_allocator_cpu.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/image_cpu.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/image_allocator_gpu.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/image_gpu.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/volume.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/volume_allocator_cpu.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/volume_cpu.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/volume_allocator_gpu.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/volume_gpu.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/copy.h
#   ${CMAKE_CURRENT_SOURCE_DIR}/iucore/imagepyramid.h
  )

SET( IU_PUBLIC_COMMON_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/common/derivative_kernels.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/common/bind_textures.cuh
)

# 
# # all the additional headers follow here below (we split that two for better overview and the seperated libs)
SET( IU_CORE_HEADERS
#  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/setvalue.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/convert.h
  )

SET( IU_HEADERS
  ${IU_PUBLIC_HEADERS}
  ${IU_PUBLIC_CORE_HEADERS}
  ${IU_CORE_HEADERS}
  )
# 
SET( IU_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore.cpp
#   ${CMAKE_CURRENT_SOURCE_DIR}/iucore/imagepyramid.cpp
#  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/setvalue.cpp
#  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/setvalue.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/convert.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/convert.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/copy.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/copy.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/iufilter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iufilter/filter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iufilter/filter.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/iufilter/filterbspline_kernels.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/iutransform.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iutransform/transform.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/iutransform/reduce.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iutransform/reduce.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/iutransform/prolongate.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iutransform/prolongate.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/iutransform/remap.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iutransform/remap.cu
  )
# 
# # denote kernel files as header only to not confuse the compiler
# set_source_files_properties(
#   # ${CMAKE_CURRENT_SOURCE_DIR}/iutransform/transform.cu
#   # ${CMAKE_CURRENT_SOURCE_DIR}/iutransform/remap.cu
#   # ${CMAKE_CURRENT_SOURCE_DIR}/iutransform/prolongate.cu
#   # ${CMAKE_CURRENT_SOURCE_DIR}/iutransform/reduce.cu
#   ${CMAKE_CURRENT_SOURCE_DIR}/iufilter/filterbspline_kernels.cu
#   PROPERTIES HEADER_FILE_ONLY yes
#   )
# 
# ##-----------------------------------------------------------------------------
# set( IU_ALL_FILES
#   ${IU_HEADERS}
#   ${IU_SOURCES}
#   ${IU_SPARSE_SOURCES}
#   ${IU_SPARSE_HEADERS}
#   ${IU_MATLAB_HEADERS}
#   ${IU_MATLAB_SOURCES}
#   #${IU_GUI_MOC_HEADERS}
#   ${IU_GUI_HEADERS}
#   ${IU_GUI_SOURCES}
#   ${IU_GUI_MOC_SOURCES}
#   ${IU_IO_HEADERS}
#   ${IU_IO_SOURCES}
# )

if(IU_BUILD_STATIC)
  message("[+] building static library")
else(IU_BUILD_STATIC)
  message("[-] do not build static library")
endif(IU_BUILD_STATIC)

set(IU_LIBS "") # variable where all libs that are built are saved
set(IU_LIBS_STATIC "") # variable where all libs that are built are saved
message("Compiler Flags NVCC: " ${CUDA_NVCC_FLAGS})



##-----------------------------------------------------------------------------
## build core library
set(IU_CORE_LIB iucore)
cuda_add_library( ${IU_CORE_LIB} SHARED ${IU_HEADERS} ${IU_SOURCES})
set_target_properties(${IU_CORE_LIB} PROPERTIES
  DEFINE_SYMBOL IUCORE_DLL_EXPORTS
  VERSION ${IMAGEUTILITIES_VERSION}
  SOVERSION ${IMAGEUTILITIES_SOVERSION}
  PUBLIC_HEADER "${IU_PUBLIC_HEADERS}"
  )
target_link_libraries( ${IU_CORE_LIB} ${CUDA_LIBRARIES})
set(IU_LIBS ${IU_LIBS} ${IU_CORE_LIB})

if(IU_BUILD_STATIC)
  cuda_add_library( ${IU_CORE_LIB}_static STATIC ${IU_HEADERS} ${IU_SOURCES})
  target_link_libraries( ${IU_CORE_LIB}_static ${CUDA_LIBRARIES})
  set(IU_LIBS_STATIC ${IU_LIBS_STATIC} ${IU_CORE_LIB}_static)
endif(IU_BUILD_STATIC)



message("additional iuio libs: ${ADDITIONAL_IUIO_DEPENDENCIES}")
##-----------------------------------------------------------------------------
## build io library
set(IU_IO_LIB iuio)
if(IU_USE_IO)
  cuda_add_library( ${IU_IO_LIB} SHARED ${IU_IO_HEADERS} ${IU_IO_SOURCES} )
  set_target_properties(${IU_IO_LIB} PROPERTIES
    DEFINE_SYMBOL IUIO_DLL_EXPORTS
    VERSION ${IMAGEUTILITIES_VERSION}
    SOVERSION ${IMAGEUTILITIES_SOVERSION}
    )
  target_link_libraries( ${IU_IO_LIB} iucore
    ${CUDA_LIBRARIES} ${OpenCV_LIBS} ${ADDITIONAL_IUIO_DEPENDENCIES}
    )
  set(IU_LIBS ${IU_LIBS} ${IU_IO_LIB})
  

  if(IU_BUILD_STATIC)
    cuda_add_library( ${IU_IO_LIB}_static STATIC ${IU_IO_HEADERS} ${IU_IO_SOURCES} )
    target_link_libraries( ${IU_IO_LIB}_static
    ${CUDA_LIBRARIES} ${OpenCV_LIBS} ${ADDITIONAL_IUIO_DEPENDENCIES}
    )
    set(IU_LIBS_STATIC ${IU_LIBS_STATIC} ${IU_IO_LIB}_static)
  endif(IU_BUILD_STATIC)
endif(IU_USE_IO)


##-----------------------------------------------------------------------------
## build gui library
set(IU_GUI_LIB iugui)
if(IU_USE_GUI)
  cuda_add_library( ${IU_GUI_LIB} SHARED ${IU_GUI_HEADERS} ${IU_GUI_SOURCES})
  set_target_properties(${IU_GUI_LIB} PROPERTIES
    DEFINE_SYMBOL IUGUI_DLL_EXPORTS
    VERSION ${IMAGEUTILITIES_VERSION}
    SOVERSION ${IMAGEUTILITIES_SOVERSION}
    )
  target_link_libraries( ${IU_GUI_LIB}
    Qt5::Widgets Qt5::OpenGL ${CUDA_LIBRARIES} ${OPENGL_LIBRARIES}
    )
  set(IU_LIBS ${IU_LIBS} ${IU_GUI_LIB})
  

  if(IU_BUILD_STATIC)
    cuda_add_library( ${IU_GUI_LIB}_static STATIC ${IU_GUI_HEADERS} ${IU_GUI_SOURCES})
    target_link_libraries( ${IU_GUI_LIB}_static
    Qt5::Widgets Qt5::OpenGL ${CUDA_LIBRARIES} ${OPENGL_LIBRARIES}
    )
    set(IU_LIBS_STATIC ${IU_LIBS_STATIC} ${IU_GUI_LIB}_static)
  endif(IU_BUILD_STATIC)
endif(IU_USE_GUI)

###-----------------------------------------------------------------------------
### build matlab-based part of library
#set(IU_MATLAB_LIB iumatlab)
#if(IU_USE_MATLAB)
#  cuda_add_library( ${IU_MATLAB_LIB} SHARED ${IU_MATLAB_HEADERS} ${IU_MATLAB_SOURCES})
#  set_target_properties(${IU_MATLAB_LIB} PROPERTIES
#    DEFINE_SYMBOL IUMATLAB_DLL_EXPORTS
#    VERSION ${IMAGEUTILITIES_VERSION}
#    SOVERSION ${IMAGEUTILITIES_SOVERSION}
#    )
#  target_link_libraries( ${IU_MATLAB_LIB} ${MATLAB_LIBRARIES} ${CUDA_SPARSE_LIBRARY})
#  set(IU_LIBS ${IU_LIBS} ${IU_MATLAB_LIB})

#endif(IU_USE_MATLAB)

##-----------------------------------------------------------------------------
## Install target

## Two different types of installation supported:
## *) local: for developement (e.g. svn checkout -- currently the default)
## *) global: for usage only (care about mixing up the two -> the global one will be preffered!)

OPTION(IU_INSTALL_LOCAL "Installation of the ImageUtilities binaries in the local imageutilities tree." ON)



if(IU_INSTALL_LOCAL )
  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX $ENV{IMAGEUTILITIES_ROOT} CACHE PATH "ImageUtilities install prefix" FORCE)
  endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
endif(IU_INSTALL_LOCAL)

if(NOT IU_INSTALL_LOCAL OR NOT CMAKE_INSTALL_PREFIX)
  if(NOT UNIX)
    set(CMAKE_INSTALL_PREFIX $ENV{ProgramFiles}/VMLibraries CACHE PATH "ImageUtilities install prefix" FORCE)
  else(NOT UNIX)
    set(CMAKE_INSTALL_PREFIX /usr/local CACHE PATH "ImageUtilities install prefix" FORCE)
  endif(NOT UNIX)
endif(NOT IU_INSTALL_LOCAL OR NOT CMAKE_INSTALL_PREFIX)

# set the cmake-files that should be installed with the package
set( IU_CMAKE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/FindFlyCapture2.cmake
  )
if(ANDROID)
set( IU_CMAKE_FILES
  ${IU_CMAKE_FILES}
  )
endif(ANDROID)

# set default install path
#set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/.. CACHE PATH "ImageUtilities install prefix" FORCE)

message("install: ${CMAKE_INSTALL_PREFIX}")

# install the libraries
if(ANDROID)
install( TARGETS ${IU_LIBS} ${IU_LIBS_STATIC}
  RUNTIME         DESTINATION   binARM
  ARCHIVE         DESTINATION   libARM
  LIBRARY         DESTINATION   libARM
  PUBLIC_HEADER   DESTINATION   include/iu
  )
else(ANDROID)
install( TARGETS ${IU_LIBS} ${IU_LIBS_STATIC}
  RUNTIME         DESTINATION   bin
  ARCHIVE         DESTINATION   lib
  LIBRARY         DESTINATION   lib
  PUBLIC_HEADER   DESTINATION   include/iu
  )
endif(ANDROID)
# install header files
# set(IMAGE_HELPER ${CMAKE_CURRENT_SOURCE_DIR}/helper_math.h)
# install(FILES ${IMAGE_HELPER}
# 	DESTINATION include
# 	COMPONENT Headers
# 	)

install(FILES ${IU_PUBLIC_CORE_HEADERS}
  DESTINATION include/iu/iucore
  COMPONENT Headers
  )
install(FILES ${IU_PUBLIC_COMMON_HEADERS}
  DESTINATION include/iu/common
  COMPONENT Headers
  )
install(FILES ${IU_PUBLIC_IO_HEADERS}
  DESTINATION include/iu/iuio
  COMPONENT Headers
  )
install(FILES ${IU_PUBLIC_GUI_HEADERS}
  DESTINATION include/iu/iugui
  COMPONENT Headers
  )
install(FILES ${IU_PUBLIC_HEADERS}
  DESTINATION include/iu
  COMPONENT Headers
  )


# install(FILES ${IU_IPP_INSTALL_HEADERS}
#   DESTINATION include/iu/iuipp
#   COMPONENT Headers
#   )
# install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/iuio/videocapture.h
#   DESTINATION include/iu/iuio
#   COMPONENT Headers
# )
install(FILES ${IU_CMAKE_FILES}
  DESTINATION cmake
  )

if(ANDROID)
  message("Compiler Flags NVCC: " ${nvcc_flags})
  message("Compiler Flags GCC:  " ${CMAKE_C_FLAGS})

endif(ANDROID)
##-----------------------------------------------------------------------------
## CPACK configuration
#INCLUDE(InstallRequiredSystemLibraries)

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

##-----------------------------------------------------------------------------
## TESTS
add_subdirectory(tests)

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
#INCLUDE(CPack)
