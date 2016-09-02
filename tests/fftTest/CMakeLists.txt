include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
)

if(NOT WIN32)
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-std=c++11")
endif(NOT WIN32)

cuda_add_executable(fftTest main.cu)
target_link_libraries(fftTest iucore iumath) 
cuda_add_cufft_to_target(fftTest)
iu_add_test(fftTest)
