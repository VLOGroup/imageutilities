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
find_package(ImageUtilities COMPONENTS iucore iuio iugui)
include(${IU_USE_FILE})
set(CUDA_NVCC_FLAGS ${IU_NVCC_FLAGS})

## GLUT
find_package(GLUT)
include_directories(${GLUT_INCLUDE_DIR})
#cuda_include_directories(${GLUT_INCLUDE_DIR})


message("IU_LIBRARIES=${IU_LIBRARIES}")

set(IU_UNITTEST_TARGETS "")

#cuda_add_executable( iu_image_cpu_display_test image_cpu_display_test.cpp )
#TARGET_LINK_LIBRARIES(iu_image_cpu_display_test ${IU_LIBRARIES})

cuda_add_executable( iugui_imagewindow_test iugui_imagewindow_test.cpp )
TARGET_LINK_LIBRARIES(iugui_imagewindow_test ${IU_LIBRARIES} ${GLUT_LIBRARIES})

set(IU_UNITTEST_TARGETS ${IU_UNITTEST_TARGETS} iugui_imagewindow_test)

# install targets
message(STATUS "install targets=${IU_UNITTEST_TARGETS}")
install(TARGETS ${IU_UNITTEST_TARGETS} RUNTIME DESTINATION bin)
