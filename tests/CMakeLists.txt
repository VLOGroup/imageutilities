project (tests)

cmake_minimum_required(VERSION 2.8)


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/tests)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/tests)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/tests)

# Include the private copy of the imageutilities (not the installed one)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

include_directories(${CMAKE_CURRENT_LIST_DIR}/../src/)

## Compiler Flags (to be added on demand)
#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Wall --std=c++11")
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-expt-extended-lambda -ftemplate-backtrace-limit=2 -Xcompiler -fPIC -lineinfo -Xcompiler -Wno-unused-but-set-variable -pg -lineinfo")

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


add_subdirectory(coreTest_copy)
#add_subdirectory(ndarrayTest)
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

