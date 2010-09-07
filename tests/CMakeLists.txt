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

PROJECT(ImageUtilities_Tests CXX C)

cmake_minimum_required(VERSION 2.8.0)
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

FILE(TO_CMAKE_PATH "$ENV{CMAKE_MODULE_PATH}" OT_CMAKE_MODULE_PATH)
FILE(TO_CMAKE_PATH "$ENV{ICG_CMAKE_MODULE_PATH}" OT_ICG_CMAKE_MODULE_PATH)
FILE(TO_CMAKE_PATH "$ENV{VMLIBRARIES_ROOT}/cmake" OT_VMLIBRARIES_CMAKE_MODULE_PATH)
SET(CMAKE_MODULE_PATH ${OT_CMAKE_MODULE_PATH} ${OT_ICG_CMAKE_MODULE_PATH} ${OT_VMLIBRARIES_CMAKE_MODULE_PATH})

# install settings
OPTION(VMLIBRARIES_IU_UNITTESTS_INSTALL_LOCAL "Installation of the ImageUtilities unittest binaries in the local vmlibraries tree." ON)
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

## combine different tests:
include(CTest)
enable_testing()
add_subdirectory(iucore_unittests)
add_subdirectory(iumath_unittests)
#add_subdirectory(iugui_unittests)
add_subdirectory(iuio_unittests)

## install the tests into the bin directory if you want to

## Two different types of installation supported:
## *) local: for developement (e.g. svn checkout -- currently the default)
## *) global: for usage only (care about mixing up the two -> the global one will be preffered!)
