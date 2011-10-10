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
# Module      : overall CMake
# Language    : CMake
# Description : overall CMakeFile for building the ImageUtilities library
#
# Author     : Manuel Werlberger
# EMail      : werlberger@icg.tugraz.at

PROJECT(imageutilities_tests CXX C)

cmake_minimum_required(VERSION 2.8.0)
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

FILE(TO_CMAKE_PATH "$ENV{VMLIBRARIES_ROOT}/imageutilities/cmake" OT_VMLIBRARIES_CMAKE_MODULE_PATH)
SET(CMAKE_MODULE_PATH ${OT_VMLIBRARIES_CMAKE_MODULE_PATH})

# install settings
OPTION(VMLIBRARIES_IU_UNITTESTS_INSTALL_LOCAL "Installation of the ImageUtilities unittest binaries in the local vmlibraries tree." ON)
if(VMLIBRARIES_IU_UNITTESTS_INSTALL_LOCAL )
  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX $ENV{VMLIBRARIES_ROOT} CACHE PATH "ImageUtilities install prefix" FORCE)
  endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
endif(VMLIBRARIES_IU_UNITTESTS_INSTALL_LOCAL)
if(NOT VMLIBRARIES_IU_UNITTESTS_INSTALL_LOCAL OR NOT CMAKE_INSTALL_PREFIX)
  if(NOT UNIX)
    set(CMAKE_INSTALL_PREFIX $ENV{ProgramFiles}/VMLibraries CACHE PATH "ImageUtilities install prefix" FORCE)
  else(NOT UNIX)
    set(CMAKE_INSTALL_PREFIX /usr/local CACHE PATH "ImageUtilities install prefix" FORCE)
  endif(NOT UNIX)
endif(NOT VMLIBRARIES_IU_UNITTESTS_INSTALL_LOCAL OR NOT CMAKE_INSTALL_PREFIX)

##-----------------------------------------------------------------------------
## Compiler Flags
if(UNIX)
  # enable warnings
  add_definitions( -Wall)
  # and debug flags
  add_definitions( -g )
endif()

IF(WIN32)
  ADD_DEFINITIONS(-DWIN32)
  # for msvc 8 deprecated warning removal
  ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE)
  ADD_DEFINITIONS(-D_CRT_NONSTDC_NO_DEPRECATE)
  ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE_GLOBALS)

  SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT.lib")
endif(WIN32)

## test if ipp is available
#find_package(IPP QUIET)

## combine different tests:
include(CTest)
enable_testing()

message(STATUS "iucore unittests:")
add_subdirectory(iucore_unittests)

# if(IPP_FOUND)
#   message(STATUS "iuipp unittests:")
#   add_subdirectory(iuipp_unittests)
# endif(IPP_INCLUDE_DIR)

message(STATUS "iumath unittests:")
add_subdirectory(iumath_unittests)

message(STATUS "iusparse unittests:")
add_subdirectory(iusparse_unittests)

#message(STATUS "iugui unittests:")
#add_subdirectory(iugui_unittests)

message(STATUS "iuio unittests:")
add_subdirectory(iuio_unittests)

message(STATUS "iutransform unittests:")
add_subdirectory(iutransform_unittests)

## install the tests into the bin directory if you want to

## Two different types of installation supported:
## *) local: for developement (e.g. svn checkout -- currently the default)
## *) global: for usage only (care about mixing up the two -> the global one will be preffered!)
