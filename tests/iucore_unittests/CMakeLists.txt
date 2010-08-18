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

## find iu and set the according libs
find_package(ImageUtilities)
include(${IU_USE_FILE})

message("IU_LIBRARIES=${IU_LIBRARIES}")

cuda_add_executable( iu_linearbuffer_unittest iu_linearbuffer_unittest.cpp )
TARGET_LINK_LIBRARIES(iu_linearbuffer_unittest ${IU_LIBRARIES})

cuda_add_executable( iu_image_cpu_unittest iu_image_cpu_unittest.cpp )
TARGET_LINK_LIBRARIES(iu_image_cpu_unittest ${IU_LIBRARIES})

cuda_add_executable( iu_image_npp_unittest iu_image_npp_unittest.cpp )
TARGET_LINK_LIBRARIES(iu_image_npp_unittest ${IU_LIBRARIES})

# install targets
set(IU_UNITTEST_TARGETS iu_linearbuffer_unittest iu_image_cpu_unittest iu_image_npp_unittest)
message(STATUS "install targets=${IU_UNITTEST_TARGETS}")
install(TARGETS ${IU_UNITTEST_TARGETS} RUNTIME DESTINATION bin)

# run tests:
add_test(iu_linearbuffer_unittest iu_linearbuffer_unittest)
add_test(iu_image_cpu_unittest iu_image_cpu_unittest)
add_test(iu_image_npp_unittest iu_image_npp_unittest)
