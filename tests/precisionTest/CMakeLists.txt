set(SOURCES
	main.cpp
)

set(HEADERS
)

#------------- for compiler flags see cmake/compiler_settings.cmake.txt -----------------
#========================================================================================


cuda_add_executable(precisionTest ${HEADERS} ${SOURCES})
xtarget_link_libraries(precisionTest iucore iumath)
iu_add_test(precisionTest)
