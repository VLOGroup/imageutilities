include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
)

#------------- for compiler flags see cmake/compiler_settings.cmake.txt -----------------
#========================================================================================

cuda_add_executable(fftTest main.cu)
xtarget_link_libraries(fftTest iucore iumath) 
cuda_add_cufft_to_target(fftTest)
iu_add_test(fftTest)
