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

project(imageutilities_lib CXX C)
#set(CMAKE_BUILD_TYPE Debug)
cmake_minimum_required(VERSION 2.8.0)

# cleanup
set(${IU_PUBLIC_HEADERS} "")

# general settings
set( CMAKE_DEBUG_POSTFIX "d" )

FILE(TO_CMAKE_PATH "$ENV{CMAKE_MODULE_PATH}" OT_CMAKE_MODULE_PATH)
FILE(TO_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake" OT_IU_CMAKE_MODULE_PATH)
SET(CMAKE_MODULE_PATH ${OT_CMAKE_MODULE_PATH} ${OT_IU_CMAKE_MODULE_PATH})

##-----------------------------------------------------------------------------
## Current version number:
##-----------------------------------------------------------------------------
set(IMAGEUTILITIES_VERSION "1.3.0")

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
OPTION(VMLIBRARIES_IU_USE_SPARSE "Including Sparse Matrix module." ON)
OPTION(VMLIBRARIES_IU_USE_IPP "Including IPP module." ON)
OPTION(VMLIBRARIES_IU_USE_MATLAB "Including Matlab module." ON)
OPTION(VMLIBRARIES_IU_USE_GUI "Including GUI module." ON)
OPTION(VMLIBRARIES_IU_USE_IO "Including IO module." ON)
OPTION(VMLIBRARIES_IU_USE_IOPGM "Including IOPGM module." ON)
OPTION(VMLIBRARIES_IU_USE_VIDEOCAPTURE "Including VideCapture IO module." OFF)
OPTION(VMLIBRARIES_IU_USE_PGRCAMERA "Including PointGray IO module." OFF)
OPTION(VMLIBRARIES_IU_BUILD_COMBINED_LIBRARY "Building a complete library package. This is just for compatibility reasons." OFF)


##-----------------------------------------------------------------------------
## Compiler Flags
if(UNIX)
  # enable all warnings
  add_definitions(-Wall)
  # and debug flags
  add_definitions(-g)

  add_definitions(-fPIC)
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
# CUDA + SDK
find_package(CUDA 3.1 REQUIRED)
find_package(CUDASDK REQUIRED)
cuda_include_directories(${CUDA_INCLUDE_DIRS} ${CUDA_CUT_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIRS} ${CUDA_CUT_INCLUDE_DIR})
set(IU_IUCORE_LIB_DEPENDENCIES ${IU_IUCORE_LIB_DEPENDENCIES} ${CUDA_LIBRARIES})


if("$ENV{COMPUTE_CAPABILITY}" MATCHES "1.1")
  message(STATUS "IU: using SM 11")
  set(CUDA_NVCC_FLAGS -arch=sm_11)
elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "1.2")
  message(STATUS "IU: using SM 12")
  set(CUDA_NVCC_FLAGS -arch=sm_12)
elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "1.3")
  message(STATUS "IU: using SM 13")
  set(CUDA_NVCC_FLAGS -arch=sm_13)
elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "2.0")
  message(STATUS "IU: using SM 20")
  set(CUDA_NVCC_FLAGS -arch=sm_20)
elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "2.1")
  message(STATUS "IU: using SM 21")
  set(CUDA_NVCC_FLAGS -arch=sm_21)
else()
  message(STATUS "IU: using SM 11")
  set(CUDA_NVCC_FLAGS -arch=sm_11)
endif()

# Checking cuda version
# set defines due to some missing functions in cuda 3.1
if(CUDA_VERSION_STRING STREQUAL "3.1")
  # using CUDA 3.1
  message(STATUS "IU: using CUDA 3.1")
  add_definitions(-DCUDA_VERSION_31)
elseif(CUDA_VERSION_STRING STREQUAL "3.2")
  # using CUDA 3.2
  message(STATUS "IU: using CUDA 3.2")
  add_definitions(-DCUDA_VERSION_32)
elseif(CUDA_VERSION_STRING STREQUAL "4.0")
  # using CUDA 4.0
  message(STATUS "IU: using CUDA 4.0")
  add_definitions(-DCUDA_VERSION_40)
elseif()
  message(STATUS "unknown CUDA version. some things might not be tested.")
endif()

##-----------------------------------------------------------------------------
## CUDA SPARSE: Sparse Matrix operations using CUDA sparse library
## (currently included in iucore lib)
find_package(CUDASparse QUIET)
if(VMLIBRARIES_IU_USE_SPARSE AND CUDASparse_FOUND)

  message("[+] ImageUtilities include SPARSE module")

  set( IU_SPARSE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/iusparse/sparsesum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/iusparse/sparsesum.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/iusparse.cpp
    )

  SET( IU_SPARSE_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/iusparse/sparsesum.cuh
  )

  set( IU_PUBLIC_HEADERS
    ${IU_PUBLIC_HEADERS}
    ${CMAKE_CURRENT_SOURCE_DIR}/iusparse.h
    )

  SET( IU_PUBLIC_SPARSE_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/iusparse/sparsematrixdefs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/iusparse/sparsematrix_cpu.h
    ${CMAKE_CURRENT_SOURCE_DIR}/iusparse/sparsematrix_gpu.h
    ${CMAKE_CURRENT_SOURCE_DIR}/iusparse/sparsesum.h
    ${CMAKE_CURRENT_SOURCE_DIR}/iusparse/sparsemultiplication.h
  )

  # install specialized headers
  install(FILES ${IU_PUBLIC_SPARSE_HEADERS}
    DESTINATION include/iu/iusparse
    COMPONENT Headers
    )

else()
  message("[-] ImageUtilities omitting SPARSE module")
  set(VMLIBRARIES_IU_USE_SPARSE OFF CACHE PATH "IU: Omit SPARSE module." FORCE)
  set(CUDA_SPARSE_LIBRARY "")
  set(IU_SPARSE_SOURCES "")
  set(IU_SPARSE_HEADERS "")
endif()

##-----------------------------------------------------------------------------
## Ipp
find_package(IPP QUIET)
if(VMLIBRARIES_IU_USE_IPP AND IPP_INCLUDE_DIR)
  message("[+] ImageUtilities include IPP module")

  include_directories(${IPP_INCLUDE_DIR})

  set( IU_IPP_INSTALL_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/iuipp/image_allocator_ipp.h
    ${CMAKE_CURRENT_SOURCE_DIR}/iuipp/image_ipp.h
    ${CMAKE_CURRENT_SOURCE_DIR}/iuipp/memorydefs_ipp.h
    ${CMAKE_CURRENT_SOURCE_DIR}/iuipp/copy_ipp.h
  )

  set( IU_IPP_HEADERS
    ${IU_IPP_INSTALL_HEADERS}
    ${CMAKE_CURRENT_SOURCE_DIR}/iuipp/image_ipp_to_cpu_connector.h
    ${CMAKE_CURRENT_SOURCE_DIR}/iuipp/image_cpu_to_ipp_connector.h
  )

  set( IU_PUBLIC_HEADERS
    ${IU_PUBLIC_HEADERS}
    ${CMAKE_CURRENT_SOURCE_DIR}/iuippdefs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/iuipp.h
    )

  set( IU_IPP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/iuipp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/iuipp/image_ipp_to_cpu_connector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/iuipp/image_cpu_to_ipp_connector.cpp
  )

else()
  message("[-] ImageUtilities omitting Ipp module.")
  set(VMLIBRARIES_IU_USE_IPP OFF CACHE PATH "IU: Omit IPP module." FORCE)

  # empty libraries
  set(IPP_LIBRARIES "")

endif()

##-----------------------------------------------------------------------------
## MATLAB (CONNECTOR) MODULE(S): Matlab connector module(s)
find_package(Matlab_ICG QUIET)

if(VMLIBRARIES_IU_USE_MATLAB AND MATLAB_FOUND)

  message("[+] ImageUtilities include MATLAB module")

  include_directories(${MATLAB_INCLUDE_DIR})


  set( IU_PUBLIC_HEADERS
    ${IU_PUBLIC_HEADERS}
    ${CMAKE_CURRENT_SOURCE_DIR}/iumatlabconnector.h
    )

  set( IU_PUBLIC_MATLAB_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/iumatlab/matlabconnector.h
    )

  set( IU_MATLAB_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/iumatlabconnector.cpp
    )


  # Matlab sparse connector (only if sparse module is included)
  if(VMLIBRARIES_IU_USE_SPARSE)
    set( IU_PUBLIC_HEADERS
      ${IU_PUBLIC_HEADERS}
      ${CMAKE_CURRENT_SOURCE_DIR}/iumatlabsparseconnector.h
      )
    set( IU_PUBLIC_MATLAB_HEADERS
      ${IU_PUBLIC_MATLAB_HEADERS}
      ${CMAKE_CURRENT_SOURCE_DIR}/iumatlab/matlabsparseconnector.h
      )

    set( IU_MATLAB_SOURCES
      ${IU_MATLAB_SOURCES}
      ${CMAKE_CURRENT_SOURCE_DIR}/iumatlabsparseconnector.cpp
      )
  endif(VMLIBRARIES_IU_USE_SPARSE)

  # install specialized headers
  install(FILES ${IU_PUBLIC_MATLAB_HEADERS}
    DESTINATION include/iu/iumatlab
    COMPONENT Headers
    )

  set( IU_MATLAB_HEADERS
    ${IU_MATLAB_HEADERS}
    ${IU_PUBLIC_MATLAB_HEADERS}
    )


else(VMLIBRARIES_IU_USE_MATLAB AND MATLAB_FOUND)
  message("[-] ImageUtilities omitting MATLAB module.")
  set(VMLIBRARIES_IU_USE_MATLAB OFF CACHE PATH "IU: Omit MATLAB module." FORCE)

  # empty libraries
  set(MATLAB_LIBRARIES "")

endif(VMLIBRARIES_IU_USE_MATLAB AND MATLAB_FOUND)


##-----------------------------------------------------------------------------
## Qt
find_package(Qt4 COMPONENTS QtCore QtGui QtOpenGL)
# only include if GUI module is used and qt4 found.
if(VMLIBRARIES_IU_USE_GUI AND QT4_FOUND)

  message("[+] ImageUtilities include GUI module")
  include(${QT_USE_FILE})

## Qwt
find_package(Qwt)
IF (QWT_FOUND)
  include_directories(${QWT_INCLUDE_DIR})
  add_definitions(-DUSE_QWT)
  set( IU_QWT_SOURCES
       ${CMAKE_CURRENT_SOURCE_DIR}/iugui/plotgapwidget.cpp
     )
  set( IU_QWT_MOC_HEADERS
       ${CMAKE_CURRENT_SOURCE_DIR}/iugui/plotgapwidget.h
     )
ENDIF (QWT_FOUND)

  ## GLEW
  find_package( GLEW )
  if(GLEW_INCLUDE_DIR)
    include_directories(${GLEW_INCLUDE_DIR})
  else(GLEW_INCLUDE_DIR)
    message("[-] ImageUtilities omitting GUI module (due to missing GLEW module")
    set(VMLIBRARIES_IU_USE_GUI OFF CACHE PATH "IU: GLEW module not found. Needed for GUI module to build." FORCE)
  endif(GLEW_INCLUDE_DIR)

  ## OpenGL
  find_package( OpenGL )
  if(OPENGL_INCLUDE_DIR)
    include_directories(${OPENGL_INCLUDE_DIR})
  else(OPENGL_INCLUDE_DIR)
    message("[-] ImageUtilities omitting GUI module (due to missing OpenGL module")
    set(VMLIBRARIES_IU_USE_GUI OFF CACHE PATH "IU: GLEW module not found. Needed for GUI module to build." FORCE)
  endif(OPENGL_INCLUDE_DIR)


  set( IU_PUBLIC_HEADERS
    ${IU_PUBLIC_HEADERS}
    ${CMAKE_CURRENT_SOURCE_DIR}/iugui.h
    )

  set( IU_GUI_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/iugui/qglimagegpuwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/iugui/overlay.h
    ${CMAKE_CURRENT_SOURCE_DIR}/iugui/imagewindow.h
    )

  set( IU_GUI_SOURCES
    #${CMAKE_CURRENT_SOURCE_DIR}/iugui/image_cpu_display.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/iugui/qgl_image_gpu_widget.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/iugui/qgl_image_gpu_widget.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/iugui/qglimagegpuwidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/iugui/qglimagegpuwidget.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/iugui/imagewindow.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/iugui/QScienceSpinBox.cpp
    ${IU_QWT_SOURCES}
    )

  set( IU_GUI_MOC_HEADERS
    #${CMAKE_CURRENT_SOURCE_DIR}/iugui/image_cpu_display.h
    #${CMAKE_CURRENT_SOURCE_DIR}/iugui/image_cpu_display_p.h
    #${CMAKE_CURRENT_SOURCE_DIR}/iugui/qgl_image_gpu_widget_p.h
    ${CMAKE_CURRENT_SOURCE_DIR}/iugui/qglimagegpuwidget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/iugui/imagewindow.h
    ${CMAKE_CURRENT_SOURCE_DIR}/iugui/QScienceSpinBox.h
    ${IU_QWT_MOC_HEADERS}
    )

  SET(IU_GUI_RC_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/iugui/iu_gui_images.qrc
    )

  # install specialized headers
  install(FILES ${IU_GUI_MOC_HEADERS} ${IU_GUI_HEADERS}
    DESTINATION include/iu/iugui
    COMPONENT Headers)

  # compile moc headers
  QT4_WRAP_CPP (IU_GUI_MOC_SOURCES ${IU_GUI_MOC_HEADERS})

  # add resources
  qt4_add_resources(QTRC_SOURCES ${IU_GUI_RC_SOURCES})

  message("QTRC_SOURCES = ${QTRC_SOURCES}")

else()
  message("[-] ImageUtilities omitting GUI module")
  set(VMLIBRARIES_IU_USE_GUI OFF CACHE PATH "IU: Omit GUI module." FORCE)

  # empty libraries
  set(QT_LIBRARIES "")
  set(OPENGL_LIBRARIES "")
  set(GLEW_LIBRARIES "")

endif()


##-----------------------------------------------------------------------------
## IO MODULE: OpenCV for Image I/O
find_package( OpenCV QUIET )
# only include if IO module is used and OpenCV found.
if(VMLIBRARIES_IU_USE_IO AND OpenCV_LIBS)

  message("[+] ImageUtilities include IO module")
  include_directories(${OpenCV_INCLUDE_DIRS})

  set( IU_PUBLIC_HEADERS
    ${IU_PUBLIC_HEADERS}
    ${CMAKE_CURRENT_SOURCE_DIR}/iuio.h
    )

  set( IU_IO_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/iuio/imageio.h
    ${CMAKE_CURRENT_SOURCE_DIR}/iuio/videocapture.h
    ${CMAKE_CURRENT_SOURCE_DIR}/iuio/videocapture_private.h
    )

  set( IU_IO_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/iuio.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/iuio/imageio.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/iuio/videocapture.cpp
    )

else()
  message("[-] ImageUtilities omitting IO module")
  set(VMLIBRARIES_IU_USE_IO OFF CACHE PATH "IU: Omit IO module." FORCE)

  # empty libraries
  set(OpenCV_LIBS "")

endif()

##-----------------------------------------------------------------------------
## IOPGM MODULE: Reads various kinds of pgm images. (no external dependencies)
if(VMLIBRARIES_IU_USE_IOPGM)

  message("[+] ImageUtilities include IOPGM module")

  set( IU_PUBLIC_HEADERS
    ${IU_PUBLIC_HEADERS}
    ${CMAKE_CURRENT_SOURCE_DIR}/iuiopgm.h
    )

  set( IU_IOPGM_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/iuio/imageiopgm.h
    )

  set( IU_IOPGM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/iuiopgm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/iuio/imageiopgm.cpp
    )

else()
  message("[-] ImageUtilities omitting IOPGM module")
  set(VMLIBRARIES_IU_USE_IOPGM OFF CACHE PATH "IU: Omit IOPGM module." FORCE)

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

# SPLITTING ALL THE HEADERS FOR THE INSTALL DIRECTORIES (PUBLIC) AND HEADERS
# THAT ARE NOT INSTALLED (public not as keyword)!
SET( IU_PUBLIC_HEADERS
  ${IU_PUBLIC_HEADERS}
  ${CMAKE_CURRENT_SOURCE_DIR}/iudefs.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucutil.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iumath.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iufilter.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iutransform.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iuinteraction.h
  )

SET( IU_PUBLIC_CORE_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/globaldefs.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/coredefs.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/memorydefs.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/linearmemory.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/linearhostmemory.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/lineardevicememory.h
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
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/imagepyramid.h
  )

# all the additional headers follow here below (we split that two for better overview and the seperated libs)
SET( IU_CORE_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/copy.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/setvalue.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/convert.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/clamp.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/iutextures.cuh
  )

SET( IU_PUBLIC_COMMON_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/common/vectormath_kernels.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/common/derivative_kernels.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/common/bsplinetexture_kernels.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/common/bind_textures.cuh
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
  ${CMAKE_CURRENT_SOURCE_DIR}/iutransform/reduce.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iutransform/prolongate.h
  ${CMAKE_CURRENT_SOURCE_DIR}/iutransform/remap.h
  )

SET( IU_INTERACTION_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/iuinteraction/draw.h
  )


SET( IU_HEADERS
  ${IU_PUBLIC_HEADERS}
  ${IU_PUBLIC_CORE_HEADERS}
  ${IU_PUBLIC_COMMON_HEADERS}
  ${IU_CORE_HEADERS}
  ${IU_MATH_HEADERS}
  ${IU_FILTER_HEADERS}
  ${IU_TRANSFORM_HEADERS}
  ${IU_INTERACTION_HEADERS}
  ${IU_SPARSE_HEADERS}
  ${IU_PUBLIC_SPARSE_HEADERS}
  )

SET( IU_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/imagepyramid.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/setvalue.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/setvalue.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/convert.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/convert.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/clamp.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iucore/clamp.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/iumath.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iumath/arithmetic.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iumath/arithmetic.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/iumath/statistics.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iumath/statistics.cu
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
  ${CMAKE_CURRENT_SOURCE_DIR}/iuinteraction.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iuinteraction/draw.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/iuinteraction/draw.cu
  ${IU_SPARSE_SOURCES}
  )

# denote kernel files as header only to not confuse the compiler
set_source_files_properties(
  # ${CMAKE_CURRENT_SOURCE_DIR}/iutransform/transform.cu
  # ${CMAKE_CURRENT_SOURCE_DIR}/iutransform/remap.cu
  # ${CMAKE_CURRENT_SOURCE_DIR}/iutransform/prolongate.cu
  # ${CMAKE_CURRENT_SOURCE_DIR}/iutransform/reduce.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/iufilter/filterbspline_kernels.cu
  PROPERTIES HEADER_FILE_ONLY yes
  )

##-----------------------------------------------------------------------------
set( IU_ALL_FILES
  ${IU_HEADERS}
  ${IU_SOURCES}
  ${IU_IPP_HEADERS}
  ${IU_SPARSE_SOURCES}
  ${IU_SPARSE_HEADERS}
  ${IU_MATLAB_HEADERS}
  ${IU_MATLAB_SOURCES}
  #${IU_GUI_MOC_HEADERS}
  ${IU_GUI_HEADERS}
  ${IU_GUI_SOURCES}
  ${IU_GUI_MOC_SOURCES}
  ${IU_IO_HEADERS}
  ${IU_IO_SOURCES}
  ${IU_VIDEOCAPTURE_HEADERS}
  ${IU_VIDEOCAPTURE_SOURCES}
  ${IU_VIDEOCAPTURE_MOC_SOURCES}
  ${IU_PGRCAMERA_HEADERS}
  ${IU_PGRCAMERA_SOURCES}
  ${IU_PGRCAMERA_MOC_SOURCES}
  )

# IF(NOT WIN32)
#   SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC"  )
#   SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_DEBUG} -fPIC"  )
#   SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC"  )
#   SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_DEBUG} -fPIC"  )
# endif()

set( IU_LIBS "") # variable where all libs that are built are saved

##-----------------------------------------------------------------------------
## build core library
set(IU_CORE_LIB iucore)
cuda_add_library( ${IU_CORE_LIB} SHARED ${IU_HEADERS} ${IU_SOURCES})
set_target_properties(${IU_CORE_LIB} PROPERTIES
  DEFINE_SYMBOL IU_DLL_EXPORTS
  VERSION ${IMAGEUTILITIES_VERSION}
  SOVERSION ${IMAGEUTILITIES_SOVERSION}
  PUBLIC_HEADER "${IU_PUBLIC_HEADERS}"
  )
target_link_libraries( ${IU_CORE_LIB} ${CUDA_LIBRARIES} ${CUDA_SPARSE_LIBRARY})
set(IU_LIBS ${IU_LIBS} ${IU_CORE_LIB})

##-----------------------------------------------------------------------------
## build ipp-based part of library
set(IU_IPP_LIB iuipp)
if(VMLIBRARIES_IU_USE_IPP)
  cuda_add_library( ${IU_IPP_LIB} SHARED ${IU_IPP_HEADERS} ${IU_IPP_SOURCES})
  set_target_properties(${IU_IPP_LIB} PROPERTIES
    DEFINE_SYMBOL IU_DLL_EXPORTS
    VERSION ${IMAGEUTILITIES_VERSION}
    SOVERSION ${IMAGEUTILITIES_SOVERSION}
    )
  target_link_libraries( ${IU_IPP_LIB} ${IPP_LIBRARIES} )
  set(IU_LIBS ${IU_LIBS} ${IU_IPP_LIB})

endif(VMLIBRARIES_IU_USE_IPP)

##-----------------------------------------------------------------------------
## build matlab-based part of library
set(IU_MATLAB_LIB iumatlab)
if(VMLIBRARIES_IU_USE_MATLAB)
  cuda_add_library( ${IU_MATLAB_LIB} SHARED ${IU_MATLAB_HEADERS} ${IU_MATLAB_SOURCES})
  set_target_properties(${IU_MATLAB_LIB} PROPERTIES
    DEFINE_SYMBOL IU_DLL_EXPORTS
    VERSION ${IMAGEUTILITIES_VERSION}
    SOVERSION ${IMAGEUTILITIES_SOVERSION}
    )
  target_link_libraries( ${IU_MATLAB_LIB} ${MATLAB_LIBRARIES} ${CUDA_SPARSE_LIBRARY})
  set(IU_LIBS ${IU_LIBS} ${IU_MATLAB_LIB})

endif(VMLIBRARIES_IU_USE_MATLAB)


##-----------------------------------------------------------------------------
## build io library
set(IU_IO_LIB iuio)
if(VMLIBRARIES_IU_USE_IO)
  cuda_add_library( ${IU_IO_LIB} SHARED ${IU_IO_HEADERS} ${IU_IO_SOURCES} )
  set_target_properties(${IU_IO_LIB} PROPERTIES
    DEFINE_SYMBOL IU_DLL_EXPORTS
    VERSION ${IMAGEUTILITIES_VERSION}
    SOVERSION ${IMAGEUTILITIES_SOVERSION}
    )
  target_link_libraries( ${IU_IO_LIB}
    ${CUDA_LIBRARIES} ${OpenCV_LIBS}
    )
  set(IU_LIBS ${IU_LIBS} ${IU_IO_LIB})
endif(VMLIBRARIES_IU_USE_IO)

##-----------------------------------------------------------------------------
## build gui library
set(IU_GUI_LIB iugui)
if(VMLIBRARIES_IU_USE_GUI)
  cuda_add_library( ${IU_GUI_LIB} SHARED ${IU_GUI_HEADERS} ${IU_GUI_SOURCES} ${IU_GUI_MOC_SOURCES} ${QTRC_SOURCES})
  set_target_properties(${IU_GUI_LIB} PROPERTIES
    DEFINE_SYMBOL IU_DLL_EXPORTS
    VERSION ${IMAGEUTILITIES_VERSION}
    SOVERSION ${IMAGEUTILITIES_SOVERSION}
    )
  target_link_libraries( ${IU_GUI_LIB}
    ${IU_CORE_LIB} ${IU_IO_LIB} ${CUDA_LIBRARIES}
    ${QT_LIBRARIES} ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${QWT_LIBRARIES}
    )
  set(IU_LIBS ${IU_LIBS} ${IU_GUI_LIB})

endif(VMLIBRARIES_IU_USE_GUI)


##-----------------------------------------------------------------------------
## build iopgm library
set(IU_IOPGM_LIB iuiopgm)
if(VMLIBRARIES_IU_USE_IOPGM)
  cuda_add_library( ${IU_IOPGM_LIB} SHARED ${IU_IOPGM_HEADERS} ${IU_IOPGM_SOURCES} )
  set_target_properties(${IU_IOPGM_LIB} PROPERTIES
    DEFINE_SYMBOL IU_DLL_EXPORTS
    VERSION ${IMAGEUTILITIES_VERSION}
    SOVERSION ${IMAGEUTILITIES_SOVERSION}
    )
  target_link_libraries( ${IU_IOPGM_LIB}
    ${CUDA_LIBRARIES}
    ${IU_CORE_LIB}
    )
  set(IU_LIBS ${IU_LIBS} ${IU_IOPGM_LIB})
endif(VMLIBRARIES_IU_USE_IOPGM)

##-----------------------------------------------------------------------------
## build VIDEOCAPTURE library
set(IU_VIDEOCAPTURE_LIB iuvideocapture)
if(VMLIBRARIES_IU_USE_VIDEOCAPTURE)
  cuda_add_library( ${IU_VIDEOCAPTURE_LIB} SHARED ${IU_VIDEOCAPTURE_HEADERS} ${IU_VIDEOCAPTURE_SOURCES} ${IU_VIDEOCAPTURE_MOC_SOURCES})
  set_target_properties(${IU_VIDEOCAPTURE_LIB} PROPERTIES
    DEFINE_SYMBOL IU_DLL_EXPORTS
    VERSION ${IMAGEUTILITIES_VERSION}
    SOVERSION ${IMAGEUTILITIES_SOVERSION}
    )
  target_link_libraries( ${IU_VIDEOCAPTURE_LIB}
    ${CUDA_LIBRARIES}
    ${QT_LIBRARIES}  ${OpenCV_LIBS}
    )
  set(IU_LIBS ${IU_LIBS} ${IU_VIDEOCAPTURE_LIB})
endif(VMLIBRARIES_IU_USE_VIDEOCAPTURE)


##-----------------------------------------------------------------------------
## build pgrcamera library
set(IU_PGRCAMERA_LIB iupgrcamera)
if(VMLIBRARIES_IU_USE_PGRCAMERA)
  cuda_add_library( ${IU_PGRCAMERA_LIB} SHARED ${IU_PGRCAMERA_HEADERS} ${IU_PGRCAMERA_SOURCES} ${IU_PGRCAMERA_MOC_SOURCES})
  set_target_properties(${IU_PGRCAMERA_LIB} PROPERTIES
    DEFINE_SYMBOL IU_DLL_EXPORTS
    VERSION ${IMAGEUTILITIES_VERSION}
    SOVERSION ${IMAGEUTILITIES_SOVERSION}
    )
  target_link_libraries( ${IU_PGRCAMERA_LIB}
    ${CUDA_LIBRARIES}
    ${QT_LIBRARIES}
    ${FLYCAPTURE2_LIBRARIES}
    )
  set(IU_LIBS ${IU_LIBS} ${IU_PGRCAMERA_LIB})

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
    ${CUDA_LIBRARIES}
    ${QT_LIBRARIES} ${OpenCV_LIBS} ${FLYCAPTURE2_LIBRARIES}
    )
  TARGET_LINK_LIBRARIES(iu ${IU_ALL_SHARED_LIBS} )

  if( NOT WIN32 )
    set( IU_ALL_STATIC_LIBS
      ${CUDA_LIBRARIES}
      ${QT_LIBRARIES}  ${OpenCV_LIBS} ${FLYCAPTURE2_LIBRARIES}
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
  ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/FindCUDASDK.cmake
  ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/FindCUDASparse.cmake
  ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/FindIPP.cmake
  ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/FindMatlab_ICG.cmake
  ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/FindGLEW.cmake
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
# install header files
install(FILES ${IU_PUBLIC_CORE_HEADERS}
  DESTINATION include/iu/iucore
  COMPONENT Headers
  )
install(FILES ${IU_PUBLIC_COMMON_HEADERS}
  DESTINATION include/iu/common
  COMPONENT Headers
  )
install(FILES ${IU_IPP_INSTALL_HEADERS}
  DESTINATION include/iu/iuipp
  COMPONENT Headers
  )
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/iuio/videocapture.h
  DESTINATION include/iu/iuio
  COMPONENT Headers
)
install(FILES ${IU_CMAKE_FILES}
  DESTINATION cmake
  )


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
