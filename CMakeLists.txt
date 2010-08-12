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

PROJECT(ImageUtilities CXX C)

cmake_minimum_required(VERSION 2.8.0)
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

FILE(TO_CMAKE_PATH "$ENV{CMAKE_MODULE_PATH}" OT_CMAKE_MODULE_PATH)
FILE(TO_CMAKE_PATH "$ENV{ICG_CMAKE_MODULE_PATH}" OT_ICG_CMAKE_MODULE_PATH)
SET(CMAKE_MODULE_PATH ${OT_CMAKE_MODULE_PATH} ${OT_ICG_CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/../cmake)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# testing:
include(CTest)
enable_testing()

add_subdirectory(src)
add_subdirectory(tests)

add_custom_target(doc COMMAND cd ${CMAKE_SOURCE_DIR} && doxygen ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
