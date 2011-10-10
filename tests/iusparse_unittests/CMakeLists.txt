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
# Author     :
# EMail      :

project(ImageUtilitiesTests CXX C)
#set(CMAKE_BUILD_TYPE Debug)
cmake_minimum_required(VERSION 2.8)

set(CUDA_NVCC_FLAGS "-arch=compute_20;-code=sm_20,sm_21;-use_fast_math;")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-keep;--ptxas-options=-v")

## find iu and set the according libs
find_package(ImageUtilities COMPONENTS iucore iuio)
include(${IU_USE_FILE})
set(CUDA_NVCC_FLAGS ${IU_NVCC_FLAGS})

message("IU_LIBRARIES=${IU_LIBRARIES}")

set(IU_UNITTEST_TARGETS "")

cuda_add_executable( iu_sparse_gpu_unittest iu_sparse_gpu_unittest.cpp )
TARGET_LINK_LIBRARIES(iu_sparse_gpu_unittest ${IU_LIBRARIES})
add_test(iu_sparse_gpu_unittest iu_sparse_gpu_unittest)
set(IU_UNITTEST_TARGETS iu_sparse_gpu_unittest)


cuda_add_executable( iu_sparse_compare iu_sparse_compare.cpp iu_sparse_compare.cu)
TARGET_LINK_LIBRARIES(iu_sparse_compare ${IU_LIBRARIES})
add_test(iu_sparse_compare iu_sparse_compare)
set(IU_UNITTEST_TARGETS ${IU_UNITTEST_TARGETS} iu_sparse_compare)

# install targets
message(STATUS "install targets=${IU_UNITTEST_TARGETS}")
install(TARGETS ${IU_UNITTEST_TARGETS} RUNTIME DESTINATION bin)
