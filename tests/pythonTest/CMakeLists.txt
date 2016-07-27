# tell cmake where FindNumPy.cmake file is
file(TO_CMAKE_PATH "${CMAKE_BINARY_DIR}/../cmake" OT_IU_CMAKE_MODULE_PATH)
set(CMAKE_MODULE_PATH ${OT_IU_CMAKE_MODULE_PATH})


set(SOURCES
    pyTest.cpp
)



find_package(Boost REQUIRED COMPONENTS python)
include_directories(${Boost_INCLUDE_DIRS})
find_package(PythonLibs 2.7 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
find_package(NumPy REQUIRED)
include_directories(${NUMPY_INCLUDE_DIRS})
FIND_PACKAGE(Eigen3 REQUIRED)
INCLUDE_DIRECTORIES( ${EIGEN3_INCLUDE_DIR} )

# compile lib, this can be imported in python
cuda_add_library(pyTest SHARED ${SOURCES})
target_link_libraries(pyTest iucore iuio ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

# copy python script to test folder
file(INSTALL pyTest.py DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

# custom test targets to build & run
# run executes the python script
add_test(NAME pyTest_build COMMAND "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target pyTest)# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target} )
ADD_TEST(NAME pyTest_run COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/pyTest.py)
SET_TESTS_PROPERTIES ( pyTest_run PROPERTIES DEPENDS pyTest_build)
