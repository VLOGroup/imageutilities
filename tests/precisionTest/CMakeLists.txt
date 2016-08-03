set(SOURCES
	main.cpp
)

set(HEADERS
)

if(NOT WIN32)
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-std=c++11")
endif(NOT WIN32)

cuda_add_executable(precisionTest ${HEADERS} ${SOURCES})
target_link_libraries(precisionTest iucore iumath)
iu_add_test(precisionTest)
