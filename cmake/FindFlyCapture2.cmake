# - Find FlyCapture2
# This module finds if the FlyCapture2 SDK inlcudes and libraries are installed
#
# This module sets the following variables:
#
# FLYCAPTURE2_FOUND
#    True if the FlyCatpure SDK includes and libraries were found
# FLYCAPTURE2_INCLUDE_DIR
#    The include path of the FlyCapture2.h header file
# FLYCAPTURE2_LIBRARIES
#    The location of the library files

# check for 32 or 64bit
if(NOT WIN32 AND NOT APPLE)
  EXEC_PROGRAM(uname ARGS -m OUTPUT_VARIABLE CMAKE_CUR_PLATFORM)
  if( CMAKE_CUR_PLATFORM MATCHES "x86_64")
    set( HAVE_64_BIT 1 )
  else()
    set( HAVE_64_BIT 0 )
  endif()
else()
  if(CMAKE_CL_64)
    set( HAVE_64_BIT 1 )
  else()
    set( HAVE_64_BIT 0 )
  endif()
endif()

# set possible library paths depending on the platform architecture.
if(HAVE_64_BIT)
  set(CMAKE_LIB_ARCH_APPENDIX 64)
  set(FLYCAPTURE2_POSSIBLE_LIB_DIRS "lib64" "lib" "bin")
  #message( STATUS "FOUND 64 BIT SYSTEM")
else()
  set(CMAKE_LIB_ARCH_APPENDIX 32)
  set(FLYCAPTURE2_POSSIBLE_LIB_DIRS "lib" "bin")
  #message( STATUS "FOUND 32 BIT SYSTEM")
endif()

# FIND THE FlyCapture2++ include path
FIND_PATH(FLYCAPTURE2_INCLUDE_DIR FlyCapture2.h
  # Windows:
  "C:/Programme/Point Grey Research/FlyCapture2/include"
  "$ENV{VMLibraries_DIR}/extern/win${CMAKE_LIB_ARCH_APPENDIX}/include"
  # Linux
  "/usr/include/"
	"/usr/include/flycapture/"
  "$ENV{VMLibraries_DIR}/extern/linux/include"
)

FIND_LIBRARY(FLYCAPTURE2_LIBRARIES NAMES flycapture FlyCapture2
    PATHS 
    "/usr/lib${CMAKE_LIB_ARCH_APPENDIX}"
    "C:/Programme/Point Grey Research/FlyCapture2"
    PATH_SUFFIXES ${FLYCAPTURE2_POSSIBLE_LIB_DIRS}
)


IF(FLYCAPTURE2_INCLUDE_DIR AND FLYCAPTURE2_LIBRARIES)
    SET(FLYCAPTURE2_FOUND true)
ENDIF()

IF(FLYCAPTURE2_FOUND)
    IF(NOT FlyCapture2_FIND_QUIETLY)
        MESSAGE(STATUS "Found FlyCapture2: ${FLYCAPTURE2_LIBRARIES}")
    ENDIF(NOT FlyCapture2_FIND_QUIETLY)
ELSE(FLYCAPTURE2_FOUND)
    IF(FlyCapture2_FIND_REQUIRED)
        MESSAGE(FATAL_ERROR "Could not find the FlyCapture2 library")
    ENDIF(FlyCapture2_FIND_REQUIRED)
ENDIF(FLYCAPTURE2_FOUND)
