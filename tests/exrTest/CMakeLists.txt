
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

cuda_add_executable(exrTest main.cpp)
target_link_libraries(exrTest iucore iuio) 
iu_add_test(exrTest)
#add_dependencies(ioTest iucore)
