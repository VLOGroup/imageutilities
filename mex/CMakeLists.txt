# Copyright (c) ICG. All rights reserved.
#
# Institute for Computer Graphics and Vision
# Graz University of Technology / Austria
#
#
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.  See the above copyright notices for more information.
#
#
# Project     : VMLibraries
# Module      : Matlab Interface for tvl1 denoising model with directed gradients
# Language    : CMake
# Description : CMakeFile for VM library Matlab interface
#
# Author     :  Manuel Werlberger
# EMail      :  werlberger@icg.tugraz.at

project(imageutilities CXX C)
cmake_minimum_required(VERSION 2.8)

##-----------------------------------------------------------------------------
## CMake
INCLUDE(AddFileDependencies)
FILE(TO_CMAKE_PATH "$ENV{VMLIBRARIES_ROOT}/imageutilities/cmake" OT_VMLIBRARIES_CMAKE_MODULE_PATH)
SET(CMAKE_MODULE_PATH ${OT_VMLIBRARIES_CMAKE_MODULE_PATH})

##-----------------------------------------------------------------------------
## Compiler Flags
if(WIN32)
  SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT.lib ")
endif(WIN32)

##-----------------------------------------------------------------------------
## Matlab
FIND_PACKAGE(Matlab_ICG)
INCLUDE_DIRECTORIES(${MATLAB_INCLUDE_DIR})

##-----------------------------------------------------------------------------
## ImageUtilities
FIND_PACKAGE(ImageUtilities COMPONENTS iucore iumatlab QUIET)
INCLUDE(${IU_USE_FILE})
set(CUDA_NVCC_FLAGS ${IU_NVCC_FLAGS})

##-----------------------------------------------------------------------------
## Build options

# Ouput directory for Linux
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
# Ouput directory for Windows
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})

##------------------------------- TV-L1d 3D Model -----------------------------

##-----------------------------------------------------------------------------
## Source files
SET(MEX_FILES    ${CMAKE_CURRENT_SOURCE_DIR}/gpu_memsize_mex.cpp )

##-----------------------------------------------------------------------------
## Create Library
#message(STATUS "MATLAB_LIBRARIES=${MATLAB_LIBRARIES}")
#message(STATUS "IU_LIBRARIES=${IU_LIBRARIES}")
#message(STATUS "IMAGEUTILITIES_LIBRARIES=${IMAGEUTILITIES_LIBRARIES}")
CUDA_ADD_LIBRARY(gpu_memsize_mex SHARED ${MEX_FILES} )
TARGET_LINK_LIBRARIES(gpu_memsize_mex ${MATLAB_LIBRARIES} ${IU_LIBRARIES} ${IMAGEUTILITIES_LIBRARIES})
SET_TARGET_PROPERTIES(gpu_memsize_mex PROPERTIES PREFIX "" LINKER_LANGUAGE CXX)

##-----------------------------------------------------------------------------
## Change naming for the correct matlab wrapper
if(WIN32)
  if (CMAKE_CL_64)
      MESSAGE("Win 64")
      SET_TARGET_PROPERTIES(gpu_memsize_mex PROPERTIES SUFFIX .mexw64 LINK_FLAGS /export:mexFunction)
  else()
      MESSAGE("Win 32")
      SET_TARGET_PROPERTIES(gpu_memsize_mex PROPERTIES SUFFIX .mexw32 LINK_FLAGS /export:mexFunction)
  endif()
else(WIN32)
  if(APPLE)
    if (CMAKE_SIZEOF_VOID_P MATCHES "8")
      MESSAGE("Mac 64")
      SET_TARGET_PROPERTIES(gpu_memsize_mex PROPERTIES SUFFIX .mexmaci64)
    else()
      MESSAGE("Mac 32")
      SET_TARGET_PROPERTIES(gpu_memsize_mex PROPERTIES SUFFIX .mexmaci32)
    endif (CMAKE_SIZEOF_VOID_P MATCHES "8")
  else(APPLE)
    if (CMAKE_SIZEOF_VOID_P MATCHES "8")
      #MESSAGE("Linux 64")
      SET_TARGET_PROPERTIES(gpu_memsize_mex PROPERTIES SUFFIX .mexa64)
    else()
      #MESSAGE("Linux 32")
      SET_TARGET_PROPERTIES(gpu_memsize_mex PROPERTIES SUFFIX .mexglx)
    endif (CMAKE_SIZEOF_VOID_P MATCHES "8")
  endif(APPLE)
endif(WIN32)


# targets to be installed
set(IMAGEUTILITIES_MATLAB_TARGETS "")
set(IMAGEUTILITIES_MATLAB_TARGETS ${IMAGEUTILITIES_MATLAB_TARGETS} gpu_memsize_mex)

##-----------------------------------------------------------------------------
# install settings
OPTION(VMLIBRARIES_IMAGEUTILITIES_MATLAB_INSTALL_LOCAL "Installation of the Imageutilities matlab wrapper binaries in the local vmlibraries tree." ON)
if(VMLIBRARIES_IMAGEUTILITIES_MATLAB_INSTALL_LOCAL )
  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX $ENV{VMLIBRARIES_ROOT} CACHE PATH "Imageutilities install prefix" FORCE)
  endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
endif(VMLIBRARIES_IMAGEUTILITIES_MATLAB_INSTALL_LOCAL)
if(NOT VMLIBRARIES_IMAGEUTILITIES_MATLAB_INSTALL_LOCAL OR NOT CMAKE_INSTALL_PREFIX)
  if(NOT UNIX)
    set(CMAKE_INSTALL_PREFIX $ENV{ProgramFiles}/VMLibraries CACHE PATH "Imageutilities install prefix" FORCE)
  else(NOT UNIX)
    set(CMAKE_INSTALL_PREFIX /usr/local CACHE PATH "Imageutilities install prefix" FORCE)
  endif(NOT UNIX)
endif(NOT VMLIBRARIES_IMAGEUTILITIES_MATLAB_INSTALL_LOCAL OR NOT CMAKE_INSTALL_PREFIX)

# install targets
message(STATUS "install targets=${IMAGEUTILITIES_MATLAB_TARGETS}")
install( TARGETS ${IMAGEUTILITIES_MATLAB_TARGETS}
  RUNTIME         DESTINATION   mex
  ARCHIVE         DESTINATION   mex
  LIBRARY         DESTINATION   mex
  )
set(IMAGEUTILITIES_MATLAB_INSTALL_FILES
  gpu_memsize_mex.m
  )
install(FILES ${IMAGEUTILITIES_MATLAB_INSTALL_FILES} DESTINATION mex)
