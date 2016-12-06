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
target_link_libraries(matlabTest iucore ${MATLAB_LIBRERIES})
SET_TARGET_PROPERTIES(matlabTest PROPERTIES PREFIX "")
SET_TARGET_PROPERTIES(matlabTest PROPERTIES SUFFIX ${MATLAB_SUFFIX})
if(WIN32)
  SET_TARGET_PROPERTIES(matlabTest PROPERTIES LINK_FLAGS "/export:mexFunction")
endif(WIN32)

# copy python script to test folder
file(INSTALL matlabTestMex.m DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
