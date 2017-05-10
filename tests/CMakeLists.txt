project (tests)

cmake_minimum_required(VERSION 2.8)

set(CUDA_PROPAGATE_HOST_FLAGS OFF)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/tests)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/tests)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/tests)

# Include the private copy of the imageutilities (not the installed one)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

include_directories(${CMAKE_CURRENT_LIST_DIR}/../src/)

#------------- for compiler flags see cmake/compiler_settings.cmake.txt -----------------
#========================================================================================

# This is the path where the images and testdata are stored
SET( DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data)
SET( RESULTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/results)
file(MAKE_DIRECTORY ${RESULTS_DIR})
# process the config file to embed the DATA_DIR to the test programs
CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/config.h.cmake ${CMAKE_CURRENT_SOURCE_DIR}/config.h )

file(TO_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake" OT_IU_CMAKE_MODULE_PATH)
set(CMAKE_MODULE_PATH ${OT_IU_CMAKE_MODULE_PATH})

## add subdirectories which should be actually built
##-----------------------------------------------------------------------------

# include the test projects
#add_custom_target(build_and_test ${CMAKE_CTEST_COMMAND} -V)

macro(iu_add_test target)
  #add_test(NAME ${target} COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target} )
  #add_dependencies(build_and_test ${target})

  add_test(NAME ${target}_build COMMAND "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target ${target})# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target} )
  ADD_TEST(NAME ${target}_run COMMAND ${target})
  SET_TESTS_PROPERTIES ( ${target}_run PROPERTIES DEPENDS ${target}_build)
endmacro(iu_add_test)

function (xtarget_link_libraries target libs) # same as target_link_libraries but with additional improvements to allow windows find the library at runtime
	LIST(REMOVE_AT ARGV 0)
	SET(LIBS ${ARGV}) # this is the hack around to pass in a list
	target_link_libraries(${target} ${LIBS})

	if(WIN32)
		# create symbolic links / copy
		FOREACH(lib ${LIBS})
			get_property(T TARGET ${lib} PROPERTY TYPE)
			#message(T="${T}")
			if(${T} STREQUAL SHARED_LIBRARY)
				set(LINK "$<TARGET_FILE_DIR:${target}>/$<TARGET_FILE_NAME:${lib}>")
				set(TG "$<TARGET_FILE:${lib}>")
				#delete old links
				add_custom_command(TARGET ${target} PRE_BUILD COMMAND IF EXIST \"${LINK}\" rm -f \"${LINK}\")
				# copy
				#add_custom_command(TARGET ${target} POST_BUILD COMMAND ${MAKE_COMMAND} ARGS -E copy $<TARGET_FILE:${lib}> $<TARGET_FILE_DIR:${target}>)
				# symlinks
				add_custom_command(TARGET ${target} POST_BUILD COMMAND IF NOT EXIST \"${LINK}\" mklink \"${LINK}\" \"${TG}\")
			endif()
		ENDFOREACH()
		# set paths in registry (global binding to target name - not recommended)
		set(TFILE ".")
		get_property(slibs TARGET ${target} PROPERTY all_libs)
		set(LIBS ${slibs};${LIBS})
		set_property(TARGET ${target} PROPERTY all_libs ${LIBS})
		FOREACH(lib ${LIBS})
			#message(${ffile})
			#file(RELATIVE_PATH rel ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} $<TARGET_FILE:${file}>)
			#message(${rel})
			set(TFILE "${TFILE};$<TARGET_LINKER_FILE_DIR:${lib}>")
		ENDFOREACH()
		#message(${TFILE})
		#add_custom_command(TARGET ${target} POST_BUILD COMMAND  reg add "HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\App Paths\\${target}.exe" /ve /d "$<TARGET_FILE:${target}>" /f )
		#add_custom_command(TARGET ${target} POST_BUILD COMMAND  reg add "HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\App Paths\\${target}.exe" /v "Path" /d "${TFILE}" /f )
	endif()
endfunction()

add_subdirectory(coreTest_copy)
if(WITH_PYTHON)
  add_subdirectory(pythonTest)
endif(WITH_PYTHON)
if(WITH_MATLAB)
  add_subdirectory(matlabTest)
endif(WITH_MATLAB)
if(IU_USE_NDARRAY)
  add_subdirectory(ndarrayTest)
endif(IU_USE_NDARRAY)
if(IU_USE_MATH)
  add_subdirectory(linearNdMemoryTest)
  add_subdirectory(precisionTest)
  add_subdirectory(fftTest)
endif(IU_USE_MATH)
#add_subdirectory(pythonTest)
if(IU_USE_IO)
  add_subdirectory(ioTest)
  if(IU_USE_MATH)
    add_subdirectory(mathTest)
  endif(IU_USE_MATH)
  if(WITH_OPENEXR)
    find_package(OpenEXR REQUIRED)
    include_directories(${OPENEXR_INCLUDE_DIRS})
    add_subdirectory(exrTest)
  endif(WITH_OPENEXR)
endif(IU_USE_IO)

