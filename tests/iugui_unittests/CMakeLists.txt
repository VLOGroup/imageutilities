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
cmake_minimum_required(VERSION 2.8)

FILE(TO_CMAKE_PATH "$ENV{CMAKE_MODULE_PATH}" OT_CMAKE_MODULE_PATH)
FILE(TO_CMAKE_PATH "$ENV{ICG_CMAKE_MODULE_PATH}" OT_ICG_CMAKE_MODULE_PATH)
SET(CMAKE_MODULE_PATH ${OT_CMAKE_MODULE_PATH} ${OT_ICG_CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake)

message("CMAKE_MODULE_PATH = ${CMAKE_MODULE_PATH}")

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

# ##-----------------------------------------------------------------------------
# ## CUDA
# find_package(CUDA)
# find_package(CUDA_SDK)
# cuda_include_directories(${CUDA_INCLUDE_DIRS} ${CUDA_CUT_INCLUDE_DIR})
# include_directories(${CUDA_INCLUDE_DIRS} ${CUDA_CUT_INCLUDE_DIR})

# ##-----------------------------------------------------------------------------
# ## NPP
# SET(NPPDIR $ENV{NPPROOT} CACHE PATH "Choose right basepath for NPP")
# #SET(NPPDIR "/usr/local/cuda/NVIDIA_NPP" CACHE PATH "Choose right basepath for NPP")
# message(STATUS "NPPDIR=${NPPDIR}")
# cuda_include_directories("${NPPDIR}/common/npp/include")
# include_directories("${NPPDIR}/common/npp/include")

# link_directories("${NPPDIR}/common/lib")
# if(UNIX)
#   set(NPP_LIBRARIES npp64 UtilNPP64)
# else(UNIX)
#     set( NPP_LIBRARIES libnpp-mt${NPP64} )
# endif(UNIX)


# ##-----------------------------------------------------------------------------
# include_directories(
#   ${CMAKE_CURRENT_SOURCE_DIR}
#   ${CMAKE_CURRENT_SOURCE_DIR}/../
#   ${CMAKE_CURRENT_SOURCE_DIR}/../src/
#   )

# cuda_include_directories(
#   ${CMAKE_CURRENT_SOURCE_DIR}
#   ${CMAKE_CURRENT_SOURCE_DIR}/../
#   ${CMAKE_CURRENT_SOURCE_DIR}/../src/
#   )

## find iu and set the according libs
find_package(ImageUtilities COMPONENTS iucore iuio iugui)
include(${IU_USE_FILE})

message("IU_LIBRARIES=${IU_LIBRARIES}")

cuda_add_executable( iugui_nppwidget_unittest iugui_nppwidget_unittest.cpp )
TARGET_LINK_LIBRARIES(iugui_nppwidget_unittest ${IU_LIBRARIES})

# run tests:
add_test(iugio_nppwidget_unittest iugio_nppwidget_unittest)
