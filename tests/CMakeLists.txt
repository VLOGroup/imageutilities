project (tests)

cmake_minimum_required(VERSION 2.8)


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/tests)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/tests)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/tests)

# Include the private copy of the imageutilities (not the installed one)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

# This is the path where the images and testdata are stored
SET( DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data)
SET( RESULTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/results)
file(MAKE_DIRECTORY ${RESULTS_DIR})
# process the config file to embed the DATA_DIR to the test programs
CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/config.h.cmake ${CMAKE_CURRENT_SOURCE_DIR}/config.h )

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


add_subdirectory(coreTest_copy)
add_subdirectory(ioTest)



message("Tests: " ${CUDA_NVCC_FLAGS})
