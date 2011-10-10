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
find_package(ImageUtilities COMPONENTS iucore iuio iuiopgm) # iuvideocapture)
include(${IU_USE_FILE})
set(CUDA_NVCC_FLAGS ${IU_NVCC_FLAGS})

set(IU_UNITTEST_TARGETS "")

cuda_add_executable( iu_imageio_unittest iu_imageio_unittest.cpp )
TARGET_LINK_LIBRARIES(iu_imageio_unittest ${IU_LIBRARIES})
set(IU_UNITTEST_TARGETS ${IU_UNITTEST_TARGETS} iu_imageio_unittest)
#add_test(iu_imageio_unittest iu_imageio_unittest)

cuda_add_executable( iu_imageio16_unittest iu_imageio16_unittest.cpp )
TARGET_LINK_LIBRARIES(iu_imageio16_unittest ${IU_LIBRARIES})
set(IU_UNITTEST_TARGETS ${IU_UNITTEST_TARGETS} iu_imageio16_unittest)
#add_test(iu_imageio_unittest iu_imageio_unittest)

cuda_add_executable( iu_videocapture_unittest iu_videocapture_unittest.cpp )
TARGET_LINK_LIBRARIES(iu_videocapture_unittest ${IU_LIBRARIES})
set(IU_UNITTEST_TARGETS ${IU_UNITTEST_TARGETS} iu_videocapture_unittest)
#add_test(iu_videocapture_unittest iu_videocapture_unittest)

cuda_add_executable( iu_cv_videocapture_test iu_cv_videocapture_test.cpp )
TARGET_LINK_LIBRARIES(iu_cv_videocapture_test ${IU_LIBRARIES})
set(IU_UNITTEST_TARGETS ${IU_UNITTEST_TARGETS} iu_cv_videocapture_test)
#add_test(iu_cv_videocapture_test iu_cv_videocapture_test)

# install targets
message(STATUS "install targets=${IU_UNITTEST_TARGETS}")
install(TARGETS ${IU_UNITTEST_TARGETS} RUNTIME DESTINATION bin)

# run tests:
