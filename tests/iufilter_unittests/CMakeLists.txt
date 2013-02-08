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

## find iu and set the according libs
find_package(ImageUtilities COMPONENTS iucore iuio QUIET)
include(${IU_USE_FILE})
set(CUDA_NVCC_FLAGS ${IU_NVCC_FLAGS})

set(IU_UNITTEST_TARGETS "")

cuda_add_executable( iufilter_bilat_unittest iufilter_bilat_unittest.cpp )
TARGET_LINK_LIBRARIES(iufilter_bilat_unittest ${IU_LIBRARIES})
set(IU_UNITTEST_TARGETS ${IU_UNITTEST_TARGETS} iufilter_bilat_unittest)
#add_test(iufilter_bilat_unittest iufilter_bilat_unittest)

# install targets
message(STATUS "install targets=${IU_UNITTEST_TARGETS}")
install(TARGETS ${IU_UNITTEST_TARGETS} RUNTIME DESTINATION bin)

# run tests:
