
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

cuda_add_executable(ndarrayTest ${HEADERS} ${SOURCES})
target_link_libraries(ndarrayTest iucore)
iu_add_test(ndarrayTest)
#add_dependencies(coreTest_copy iucore)
