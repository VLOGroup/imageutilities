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
find_package(ImageUtilities REQUIRED COMPONENTS iucore iuipp)
include(${IU_USE_FILE})

set(IUIPP_UNITTEST_TARGETS "")

# cuda_add_executable( iu_linearbuffer_unittest iu_linearbuffer_unittest.cpp )
# TARGET_LINK_LIBRARIES(iu_linearbuffer_unittest ${IU_LIBRARIES})
# add_test(iu_linearbuffer_unittest iu_linearbuffer_unittest)
# set(IUIPP_UNITTEST_TARGETS ${IUIPP_UNITTEST_TARGETS} iu_linearbuffer_unittest)

cuda_add_executable( iu_image_ipp_unittest iu_image_ipp_unittest.cpp )
TARGET_LINK_LIBRARIES(iu_image_ipp_unittest ${IU_LIBRARIES})
add_test(iu_image_ipp_unittest iu_image_ipp_unittest)
set(IUIPP_UNITTEST_TARGETS ${IUIPP_UNITTEST_TARGETS} iu_image_ipp_unittest)

# install targets
message(STATUS "install targets=${IUIPP_UNITTEST_TARGETS}")
install(TARGETS ${IUIPP_UNITTEST_TARGETS} RUNTIME DESTINATION bin)
