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
# Module      : Testing
# Language    : CMake
# Description : CMakeFile for testing the ImageUtilities library
#
# Author     : Manuel Werlberger
# EMail      : werlberger@icg.tugraz.at

project(ImageUtilitiesTests CXX C)
#set(CMAKE_BUILD_TYPE Debug)
cmake_minimum_required(VERSION 2.6)

##-----------------------------------------------------------------------------
## Compiler Flags
if(UNIX)
  # enable warnings
  add_definitions( -Wall)
  # and debug flags
  add_definitions( -g )
endif()

set( NPP64 "" )
if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
  set( NPP64 "-x64" )
endif( CMAKE_SIZEOF_VOID_P EQUAL 8 )

IF(WIN32)
  ADD_DEFINITIONS(-DWIN32)
  # for msvc 8 deprecated warning removal
  ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE)
  ADD_DEFINITIONS(-D_CRT_NONSTDC_NO_DEPRECATE)
  ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE_GLOBALS)

  SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT.lib")
endif(WIN32)

##-----------------------------------------------------------------------------
## ImageUtilties - Modules
add_definitions(-DIU_USE_GUI_MODULE)


##-----------------------------------------------------------------------------
## CUDA
if(NOT CUDA_BUILD_EMULATION)
  set(CUDA_BUILD_EMULATION OFF CACHE BOOL "disable emulation mode")
  #set(CUDA_BUILD_EMULATION ON CACHE BOOL "enable emulation mode")
endif()
find_package(CUDA REQUIRED COMPONENTS sdk)
set(CUDA_BUILD_CUBIN OFF)
# set(CUDA_PROPAGATE_HOST_FLAGS OFF)
# set(CUDA_NVCC_FLAGS_DEBUG)
# set(CUDA_NVCC_FLAGS -arch sm_13)#-g;-G)
cuda_include_directories(${CUDA_INCLUDE_DIRS} ${CUDA_CUT_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIRS} ${CUDA_CUT_INCLUDE_DIR})


##-----------------------------------------------------------------------------
## NPP
SET(NPPDIR $ENV{NPPROOT} CACHE PATH "Choose right basepath for NPP")
#SET(NPPDIR "/usr/local/cuda/NVIDIA_NPP" CACHE PATH "Choose right basepath for NPP")
message(STATUS "NPPDIR=${NPPDIR}")
cuda_include_directories("${NPPDIR}/common/npp/include")
include_directories("${NPPDIR}/common/npp/include")

link_directories("${NPPDIR}/common/lib")
if(UNIX)
  set(NPP_LIBRARIES npp64 UtilNPP64)
else(UNIX)
    set( NPP_LIBRARIES libnpp-mt${NPP64} )
endif(UNIX)

##-----------------------------------------------------------------------------
## Qt
find_package(Qt4 COMPONENTS QtCore QtGui QtOpenGL REQUIRED )
include(${QT_USE_FILE})

## Glew
find_package( GLEW REQUIRED )
include_directories(${GLEW_INCLUDE_DIR})

## OpenGL
find_package( OpenGL REQUIRED )
include_directories(${OPENGL_INCLUDE_DIR})


##-----------------------------------------------------------------------------
## OpenCV
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

##-----------------------------------------------------------------------------
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../
  ${CMAKE_CURRENT_SOURCE_DIR}/../src/
  )

cuda_include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../
  ${CMAKE_CURRENT_SOURCE_DIR}/../src/
  )

##-----------------------------------------------------------------------------
## build options
set( IU_LIBRARIES optimized iu debug iud)
set( IU_STATIC_LIBRARIES optimized iu_static debug iu_staticd)
set(ALL_LIBRARIES
  ${NPP_LIBRARIES} ${CUDA_LIBRARIES}
  ${OpenCV_LIBS}
  ${QT_LIBRARIES} ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES}
  )

message(STATUS "CMAKE_CURRENT_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}")
link_directories(
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/../release
    ${CMAKE_CURRENT_BINARY_DIR}/../debug
    )

# cuda_add_executable( iu_unittests iu_unittests.cpp )
# TARGET_LINK_LIBRARIES(iu_unittests ${ALL_LIBRARIES} ${IU_LIBRARIES})

cuda_add_executable( iu_linearbuffer_unittest iu_linearbuffer_unittest.cpp )
TARGET_LINK_LIBRARIES(iu_linearbuffer_unittest ${ALL_LIBRARIES} ${IU_LIBRARIES})

cuda_add_executable( iu_image_cpu_unittest iu_image_cpu_unittest.cpp )
TARGET_LINK_LIBRARIES(iu_image_cpu_unittest ${ALL_LIBRARIES} ${IU_LIBRARIES})

cuda_add_executable( iu_image_npp_unittest iu_image_npp_unittest.cpp )
TARGET_LINK_LIBRARIES(iu_image_npp_unittest ${ALL_LIBRARIES} ${IU_LIBRARIES})

cuda_add_executable( iu_arithmetic_npp_unittest iu_arithmetic_npp_unittest.cpp )
TARGET_LINK_LIBRARIES(iu_arithmetic_npp_unittest ${ALL_LIBRARIES} ${IU_LIBRARIES})

cuda_add_executable( iu_statistics_npp_unittest iu_statistics_npp_unittest.cpp )
TARGET_LINK_LIBRARIES(iu_statistics_npp_unittest ${ALL_LIBRARIES} ${IU_LIBRARIES})

# cuda_add_executable( iu_math_arithmetic_unittest iu_math_arithmetic_unittest.cpp)
# TARGET_LINK_LIBRARIES(iu_math_arithmetic_unittest ${ALL_LIBRARIES} ${IU_LIBRARIES})

# cuda_add_executable( iu_nppwidget_unittest iu_nppwidget_unittest.cpp )
# TARGET_LINK_LIBRARIES(iu_nppwidget_unittest ${ALL_LIBRARIES} ${IU_LIBRARIES})

cuda_add_executable( iu_imageio_unittest iu_imageio_unittest.cpp )
TARGET_LINK_LIBRARIES(iu_imageio_unittest ${ALL_LIBRARIES} ${IU_LIBRARIES})

cuda_add_executable( iu_camera_unittest iu_camera_unittest.cpp )
TARGET_LINK_LIBRARIES(iu_camera_unittest ${ALL_LIBRARIES} ${IU_LIBRARIES})

cuda_add_executable( iu_filter_unittest iu_filter_unittest.cpp )
TARGET_LINK_LIBRARIES(iu_filter_unittest ${ALL_LIBRARIES} ${IU_LIBRARIES})

# # build everyting again using the static lib from iu
# cuda_add_executable( iu_static_unittests iu_unittests.cpp )
# TARGET_LINK_LIBRARIES(iu_static_unittests ${ALL_LIBRARIES} ${IU_STATIC_LIBRARIES})

# cuda_add_executable( iu_static_linearbuffer_unittest iu_linearbuffer_unittest.cpp )
# TARGET_LINK_LIBRARIES(iu_static_linearbuffer_unittest ${ALL_LIBRARIES} ${IU_STATIC_LIBRARIES})

# cuda_add_executable( iu_static_math_arithmetic_unittest iu_math_arithmetic_unittest.cpp)
# TARGET_LINK_LIBRARIES(iu_static_math_arithmetic_unittest ${ALL_LIBRARIES} ${IU_STATIC_LIBRARIES})

# cuda_add_executable( iu_static_nppwidget_unittest iu_nppwidget_unittest.cpp )
# TARGET_LINK_LIBRARIES(iu_static_nppwidget_unittest ${ALL_LIBRARIES} ${IU_STATIC_LIBRARIES})

# cuda_add_executable( iu_static_imageio_unittest iu_imageio_unittest.cpp )
# TARGET_LINK_LIBRARIES(iu_static_imageio_unittest ${ALL_LIBRARIES} ${IU_STATIC_LIBRARIES})

# cuda_add_executable( iu_static_filter_unittest iu_filter_unittest.cpp )
# TARGET_LINK_LIBRARIES(iu_static_filter_unittest ${ALL_LIBRARIES} ${IU_STATIC_LIBRARIES})

# run tests:
add_test(iu_linearbuffer_unittest iu_linearbuffer_unittest)
add_test(iu_image_cpu_unittest iu_image_cpu_unittest)
add_test(iu_image_npp_unittest iu_image_npp_unittest)
add_test(iu_arithmetic_npp_unittest iu_arithmetic_npp_unittest)
add_test(iu_statistics_npp_unittest iu_statistics_npp_unittest)
