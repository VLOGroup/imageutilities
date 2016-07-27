
set(SOURCES
	main.cpp
	test.cu
)

set(HEADERS
	test.cuh
)

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-std=c++11")

cuda_add_executable(linearNdMemoryTest ${SOURCES} ${HEADERS})
target_link_libraries(linearNdMemoryTest iucore iumath)
iu_add_test(mathTest)
