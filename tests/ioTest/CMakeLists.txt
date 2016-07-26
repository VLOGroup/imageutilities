
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
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11")

cuda_add_executable(ioTest main.cpp)
target_link_libraries(ioTest iucore iuio) 
iu_add_test(ioTest)
#add_dependencies(ioTest iucore)
