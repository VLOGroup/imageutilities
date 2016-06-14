
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

cuda_add_executable(ioTest main.cpp)
target_link_libraries(ioTest iucore iuio) 
iu_add_test(ioTest)
