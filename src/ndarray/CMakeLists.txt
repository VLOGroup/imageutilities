CMAKE_MINIMUM_REQUIRED ( VERSION 3.2.0 )

PROJECT( ndarray )

set (VERSION_MAJOR 1)
set (VERSION_MINOR 0)

SET (SRC_DIR .)
SET(MY_LIB_PATH "../../../lib")
SET(MY_BIN_PATH "../../../bin/tests")
include_directories(${SRC_DIR})

##--------compiler settings------------------------

include(${CMAKE_CURRENT_LIST_DIR}/cmake/flags_functions.cmake.txt)
add_flags(CMAKE_CXX_FLAGS "-g -ggdb --std=c++11 -pg -fPIC -fmax-errors=3 -Wall -Wfloat-conversion -Wc++11-compat -Wdiv-by-zero -Wfloat-equal -Wdisabled-optimization -Wno-unused-but-set-variable -Wno-unused-result -Wno-unused-variable") # -Q --help=warnings,joined,separate") #-Q --help=warnings #-Wfatal-errors
add_flags(CUDA_NVCC_FLAGS "-G -g -pg -expt-extended-lambda -ftemplate-backtrace-limit=2 -Xcompiler -fPIC -lineinfo -Xcompiler -Wno-unused-but-set-variable -pg -lineinfo")

message("COMPILER FLAGS for ndarray:")
message(STATUS "CMAKE_CXX_FLAGS= ${CMAKE_CXX_FLAGS}")
message(STATUS "CUDA_NVCC_FLAGS= ${CUDA_NVCC_FLAGS}")
message(STATUS "CUDA_PROPAGATE_HOST_FLAGS= ${CUDA_PROPAGATE_HOST_FLAGS}")

##-----------build tools---------------------------

include(${CMAKE_CURRENT_LIST_DIR}/cmake/functions.cmake.txt)
                                      
##-----------------------------------------------------------------------------

SET(SRC
${SRC_DIR}/error.cpp
${SRC_DIR}/ndarray_mem.cpp
${SRC_DIR}/ndarray.cpp
${SRC_DIR}/ndarray_op.cu
)

my_cuda_add_library(ndarray ${SRC})
set(NDARRAY_LIB ${MYLIBS} PARENT_SCOPE)

SET(SRC
${SRC_DIR}/ndarray_example.cpp
${SRC_DIR}/ndarray_example.cu
)

my_cuda_add_executable(ndarrayTest ${SRC})
my_cuda_add_executable(cuda_bug cuda_bug.cu)
