
set(SOURCES
	main.cpp
	
	#test.cu
)

set(HEADRES
	#test.cuh
)

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
)

#------------- for compiler flags see cmake/compiler_settings.cmake.txt -----------------
#========================================================================================

cuda_add_executable(mathTest main.cpp)
target_link_libraries(mathTest iucore iuio iumath)
iu_add_test(mathTest)
