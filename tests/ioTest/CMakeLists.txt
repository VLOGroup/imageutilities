
set(SOURCES
	main.cpp
	
	#test.cu
)

set(HEADRES
	#test.cuh
)

#include_directories(
#	${CMAKE_CURRENT_SOURCE_DIR}
#)

#------------- for compiler flags see cmake/compiler_settings.cmake.txt -----------------
#========================================================================================

cuda_add_executable(ioTest main.cpp)
xtarget_link_libraries(ioTest iucore iuio opencv_highgui)
iu_add_test(ioTest)
#add_dependencies(ioTest iucore)
