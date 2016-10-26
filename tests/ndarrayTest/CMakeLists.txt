set(SOURCES
ndarray_example.cpp
ndarray_example.cu
)

set(HEADERS
ndarray_example.h
)

#------------- for compiler flags see cmake/compiler_settings.cmake.txt -----------------
#========================================================================================

cuda_add_executable(ndarrayTest ${HEADERS} ${SOURCES})
target_link_libraries(ndarrayTest iucore)
#iu_add_test(ndarrayTest_iu)

iu_add_test(ndarrayTest)

