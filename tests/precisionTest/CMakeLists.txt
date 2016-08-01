set(SOURCES
	main.cpp
)

set(HEADERS
)

cuda_add_executable(precisionTest ${HEADERS} ${SOURCES})
target_link_libraries(precisionTest iucore iumath)
iu_add_test(precisionTest)