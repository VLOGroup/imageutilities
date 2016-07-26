set(SOURCES
	main.cpp
	
    test.cu
)

set(HEADERS
    test.cuh
)

#include_directories(
#	${CMAKE_CURRENT_SOURCE_DIR}
#)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11")
cuda_add_executable(coreTest_copy ${HEADERS} ${SOURCES})
target_link_libraries(coreTest_copy iucore)
iu_add_test(coreTest_copy)
#add_dependencies(coreTest_copy iucore)
