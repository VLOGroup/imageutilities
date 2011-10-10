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
find_package(ImageUtilities REQUIRED COMPONENTS iucore)
include(${IU_USE_FILE})
set(CUDA_NVCC_FLAGS ${IU_NVCC_FLAGS})

set(IU_UNITTEST_TARGETS "")

cuda_add_executable( iu_linearbuffer_unittest iu_linearbuffer_unittest.cpp )
TARGET_LINK_LIBRARIES(iu_linearbuffer_unittest ${IU_LIBRARIES})
add_test(iu_linearbuffer_unittest iu_linearbuffer_unittest)
set(IU_UNITTEST_TARGETS ${IU_UNITTEST_TARGETS} iu_linearbuffer_unittest)

cuda_add_executable( iu_image_cpu_unittest iu_image_cpu_unittest.cpp )
TARGET_LINK_LIBRARIES(iu_image_cpu_unittest ${IU_LIBRARIES})
add_test(iu_image_cpu_unittest iu_image_cpu_unittest)
set(IU_UNITTEST_TARGETS ${IU_UNITTEST_TARGETS} iu_image_cpu_unittest)

cuda_add_executable( iu_image_gpu_unittest iu_image_gpu_unittest.cpp )
TARGET_LINK_LIBRARIES(iu_image_gpu_unittest ${IU_LIBRARIES})
add_test(iu_image_gpu_unittest iu_image_gpu_unittest)
set(IU_UNITTEST_TARGETS ${IU_UNITTEST_TARGETS} iu_image_gpu_unittest)

cuda_add_executable( iu_volume_gpu_unittest iu_volume_gpu_unittest.cpp )
TARGET_LINK_LIBRARIES(iu_volume_gpu_unittest ${IU_LIBRARIES})
add_test(iu_volume_gpu_unittest iu_volume_gpu_unittest)
set(IU_UNITTEST_TARGETS ${IU_UNITTEST_TARGETS} iu_volume_gpu_unittest)

cuda_add_executable( iu_malloc_gpu_unittest iu_malloc_gpu_unittest.cpp )
TARGET_LINK_LIBRARIES(iu_malloc_gpu_unittest ${IU_LIBRARIES})
add_test(iu_malloc_gpu_unittest iu_malloc_gpu_unittest)
set(IU_UNITTEST_TARGETS ${IU_UNITTEST_TARGETS} iu_malloc_gpu_unittest)

cuda_add_executable( iu_exception_unittest iu_exception_unittest.cpp )
TARGET_LINK_LIBRARIES(iu_exception_unittest ${IU_LIBRARIES})
add_test(iu_exception_unittest iu_exception_unittest)
set(IU_UNITTEST_TARGETS ${IU_UNITTEST_TARGETS} iu_exception_unittest)

cuda_add_executable( iu_minmax_unittest iu_minmax_unittest.cpp iu_minmax_unittest.cu)
TARGET_LINK_LIBRARIES(iu_minmax_unittest ${IU_LIBRARIES})
add_test(iu_minmax_unittest iu_minmax_unittest)
set(IU_UNITTEST_TARGETS ${IU_UNITTEST_TARGETS} iu_minmax_unittest)

# install targets
message(STATUS "install targets=${IU_UNITTEST_TARGETS}")
install(TARGETS ${IU_UNITTEST_TARGETS} RUNTIME DESTINATION bin)
