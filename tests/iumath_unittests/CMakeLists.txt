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
find_package(ImageUtilities COMPONENTS iucore)
include(${IU_USE_FILE})

message("IU_LIBRARIES=${IU_LIBRARIES}")

cuda_add_executable( iu_arithmetic_npp_unittest iu_arithmetic_npp_unittest.cpp )
TARGET_LINK_LIBRARIES(iu_arithmetic_npp_unittest ${IU_LIBRARIES})

cuda_add_executable( iu_statistics_npp_unittest iu_statistics_npp_unittest.cpp )
TARGET_LINK_LIBRARIES(iu_statistics_npp_unittest ${IU_LIBRARIES})

# install targets
set(IU_UNITTEST_TARGETS iu_arithmetic_npp_unittest iu_statistics_npp_unittest)
message(STATUS "install targets=${IU_UNITTEST_TARGETS}")
install(TARGETS ${IU_UNITTEST_TARGETS} RUNTIME DESTINATION bin)

# run tests:
add_test(iu_statistics_npp_unittest iu_statistics_npp_unittest)
add_test(iu_statistics_npp_unittest iu_statistics_npp_unittest)
