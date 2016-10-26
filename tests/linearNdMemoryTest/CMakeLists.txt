
set(SOURCES
	main.cpp
	test.cu
)

set(HEADERS
	test.cuh
)

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
)

#------------- for compiler flags see cmake/compiler_settings.cmake.txt -----------------
#========================================================================================


cuda_add_executable(linearNdMemoryTest ${SOURCES} ${HEADERS})
target_link_libraries(linearNdMemoryTest iucore iumath)
iu_add_test(mathTest)
