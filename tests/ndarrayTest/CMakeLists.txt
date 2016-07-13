#include(${CMAKE_CURRENT_LIST_DIR}/../../src/ndarray/cmake/flags_functions.cmake.txt)
#add_flags(CMAKE_CXX_FLAGS "--std=c++11 -fPIC -fmax-errors=3 -Wall -Wfloat-conversion -Wc++11-compat -Wdiv-by-zero -Wfloat-equal -Wdisabled-optimization -Wno-unused-but-set-variable -Wno-unused-result -Wno-unused-variable") # -Q --help=warnings,joined,separate") #-Q --help=warnings
#add_flags(CUDA_NVCC_FLAGS "-expt-extended-lambda -ftemplate-backtrace-limit=2 -Xcompiler -fPIC -lineinfo -Xcompiler -Wno-unused-but-set-variable")

set(SOURCES
../../src/ndarray/ndarray_example.cpp
../../src/ndarray/ndarray_example.cu
)

set(HEADERS
../../src/ndarray/ndarray_example.h
)

#cuda_add_executable(ndarrayTest_iu ${HEADERS} ${SOURCES})
#target_link_libraries(ndarrayTest_iu iucore)
#iu_add_test(ndarrayTest_iu)

iu_add_test(ndarrayTest)

