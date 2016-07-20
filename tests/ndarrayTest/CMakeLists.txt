#include(${CMAKE_CURRENT_LIST_DIR}/../../src/ndarray/cmake/flags_functions.cmake.txt)
#add_flags(CMAKE_CXX_FLAGS "--std=c++11 -fPIC -fmax-errors=3 -Wall -Wfloat-conversion -Wc++11-compat -Wdiv-by-zero -Wfloat-equal -Wdisabled-optimization -Wno-unused-but-set-variable -Wno-unused-result -Wno-unused-variable") # -Q --help=warnings,joined,separate") #-Q --help=warnings
#add_flags(CUDA_NVCC_FLAGS "-expt-extended-lambda -ftemplate-backtrace-limit=2 -Xcompiler -fPIC -lineinfo -Xcompiler -Wno-unused-but-set-variable")

set(SOURCES
ndarray_example.cpp
ndarray_example.cu
)

set(HEADERS
ndarray_example.h
)
add_flags(CMAKE_CXX_FLAGS "-ggdb --std=c++11 -pg -fPIC -fmax-errors=3 -Wall -Wc++11-compat -Wdiv-by-zero -Wfloat-equal -Wdisabled-optimization -Wno-unused-but-set-variable -Wno-unused-result -Wno-unused-variable -Wno-unused-value") # -Q --help=warnings,joined,separate") #-Q --help=warnings #-Wfatal-errors
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
add_flags(CUDA_NVCC_FLAGS "-std=c++11 -pg -expt-extended-lambda -ftemplate-backtrace-limit=2 -Xcompiler -fPIC -lineinfo -Xcompiler -Wno-unused-but-set-variable -pg -lineinfo")

cuda_add_executable(ndarrayTest ${HEADERS} ${SOURCES})
target_link_libraries(ndarrayTest iucore)
#iu_add_test(ndarrayTest_iu)

iu_add_test(ndarrayTest)

