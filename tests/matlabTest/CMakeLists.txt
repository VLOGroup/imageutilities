# tell cmake where FindMatlab_ICG.cmake file is
file(TO_CMAKE_PATH "${CMAKE_BINARY_DIR}/../cmake" OT_IU_CMAKE_MODULE_PATH)
set(CMAKE_MODULE_PATH ${OT_IU_CMAKE_MODULE_PATH})

set(SOURCES
    matlabTest.cpp
)

find_package(Matlab_ICG)
include_directories(${MATLAB_INCLUDE_DIR})

#------------- for compiler flags see cmake/compiler_settings.cmake.txt -----------------
#========================================================================================

# compile lib, this can be used in matlab
cuda_add_library(matlabTest SHARED ${SOURCES})
xtarget_link_libraries(matlabTest iucore ${MATLAB_LIBRARIES})
SET_TARGET_PROPERTIES(matlabTest PROPERTIES PREFIX "")
SET_TARGET_PROPERTIES(matlabTest PROPERTIES SUFFIX ${MATLAB_SUFFIX})
if(WIN32)
  SET_TARGET_PROPERTIES(matlabTest PROPERTIES LINK_FLAGS "/export:mexFunction")
endif(WIN32)

# copy matlab script to test folder
file(INSTALL matlabTestMex.m DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

# custom test targets to build & run
# run executes the python script
add_test(NAME matlabTest_build COMMAND "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target matlabTest)# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target} )
ADD_TEST(NAME matlabTest_run COMMAND ${MATLAB_EXECUTABLE} -nodisplay -nosplash -nodesktop -r "try, run('${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/matlabTestMex.m'), catch err, disp(err.message), exit(1), end, exit")
SET_TESTS_PROPERTIES ( matlabTest_run PROPERTIES DEPENDS matlabTest_build)