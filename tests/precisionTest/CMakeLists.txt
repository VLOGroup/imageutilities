set(SOURCES
	main.cpp
)

set(HEADERS
)
add_flags(CMAKE_CXX_FLAGS "--std=c++11")
add_flags(CUDA_NVCC_FLAGS "--std=c++11")
cuda_add_executable(precisionTest ${HEADERS} ${SOURCES})
target_link_libraries(precisionTest iucore iumath)
iu_add_test(precisionTest)
