# tell cmake where FindNumPy.cmake file is
file(TO_CMAKE_PATH "${CMAKE_BINARY_DIR}/../cmake" OT_IU_CMAKE_MODULE_PATH)
set(CMAKE_MODULE_PATH ${OT_IU_CMAKE_MODULE_PATH})


set(SOURCES
    pyTest.cpp
    test.cuh
    test.cu
)

#------------- for compiler flags see cmake/compiler_settings.cmake.txt -----------------
#========================================================================================

find_package(PythonInterp) # run this such that lib and interpreter path match.
find_package(PythonLibs 2.7 REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})
message(STATUS "Python libraries: " ${PYTHON_LIBRARIES})

# Extract boost root dir. Should be the same as the Python lib dir.
get_filename_component(PYTHON_LIBRARY_DIR ${PYTHON_LIBRARIES} DIRECTORY)
get_filename_component(BOOST_ROOT ${PYTHON_LIBRARY_DIR} DIRECTORY)

message(STATUS "Boost root: " ${BOOST_ROOT})

find_package(NumPy REQUIRED)
include_directories(${NUMPY_INCLUDE_DIRS})

find_package(Boost REQUIRED COMPONENTS python)
message(STATUS "Boost libraries:" ${Boost_LIBRARIES})

if(NOT (${PYTHON_LIBRARY_DIR} MATCHES  ${Boost_LIBRARY_DIRS}))
    message(FATAL_ERROR "Boost library dir does not match python library dir. Is boost installed? "
                        "If you use anaconda, type `conda install boost`. "
                        "If you use system python (ubuntu), type `sudo apt-get install libboost-python`.")
endif(NOT (${PYTHON_LIBRARY_DIR} MATCHES  ${Boost_LIBRARY_DIRS}))
FIND_PACKAGE(Eigen3 REQUIRED)

include_directories(${NUMPY_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
INCLUDE_DIRECTORIES( ${EIGEN3_INCLUDE_DIR} )

# compile lib, this can be imported in python
cuda_add_library(pyTest SHARED ${SOURCES})
xtarget_link_libraries(pyTest iucore ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

if(WIN32)
    SET_TARGET_PROPERTIES(pyTest PROPERTIES PREFIX "lib")
    SET_TARGET_PROPERTIES(pyTest PROPERTIES SUFFIX ".pyd")
endif(WIN32)

# copy python script to test folder
file(INSTALL pyTest.py DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

# custom test targets to build & run
# run executes the python script
add_test(NAME pyTest_build COMMAND "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target pyTest)# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target} )
ADD_TEST(NAME pyTest_run COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/pyTest.py)
SET_TESTS_PROPERTIES ( pyTest_run PROPERTIES DEPENDS pyTest_build)
