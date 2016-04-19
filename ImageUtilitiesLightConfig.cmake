# - Try to find the ImageutilitiesLight library
#
# Once done this will define
#
#  IMAGEUTILITIESLIGHT_FOUND - system has IMAGEUTILITIESLIGHT
#  IMAGEUTILITIESLIGHT_DEFINITIONS - definitions for IMAGEUTILITIESLIGHT (only defines needed for found and activated libraries are added)
#  IMAGEUTILITIESLIGHT_INCLUDE_DIR - the IMAGEUTILITIESLIGHT include directory
#  IMAGEUTILITIESLIGHT_LIBRARIES - Link these to use IMAGEUTILITIESLIGHT
#  IMAGEUTILITIESLIGHT_LIBRARY_DIR - link directories, useful for rpath

find_package(CUDA 5.0 REQUIRED)    # imageutiltites depend on cuda so we find it here

# set compute capability from environment variable (directly useable as nvcc flag)
if(NOT ANDROID)
    if ("${CUDA_NVCC_FLAGS}" MATCHES "-arch")
      message(WARNING "ImageUtilities detected nvcc compiler flag -arch is already set to ${CUDA_NVCC_FLAGS}, resetting it to CC $ENV{COMPUTE_CAPABILITY}!")
      string(REGEX REPLACE "-arch=sm_[0-9]+" "" CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})
    endif()

    if("$ENV{COMPUTE_CAPABILITY}" MATCHES "1.1")
    message("A minimum of compute capability 3.0 and CUDA 5.0 is needed!")
    elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "1.2")
    message("A minimum of compute capability 3.0 and CUDA 5.0 is needed!")
    elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "1.3")
    message("A minimum of compute capability 3.0 and CUDA 5.0 is needed!")
    elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "2.0")
    message("A minimum of compute capability 3.0 and CUDA 5.0 is needed!")
    elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "2.1")
    message("A minimum of compute capability 3.0 and CUDA 5.0 is needed!")
    elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "3.0")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_30")
    elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "3.2")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_32")
    elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "3.5")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_35")
    elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "3.7")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_37")
    elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "5.0")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_50")
    elseif("$ENV{COMPUTE_CAPABILITY}" MATCHES "5.2")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_52")
    else()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_30")
    endif()
else(NOT ANDROID)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_30 -target-cpu-arch=ARM -target-os-variant=Android")
endif(NOT ANDROID)

### trouble if FindPackage(ImageutilitiesLight) is called multiple times with different IU_MODULES
### do the full check every times
#if(IMAGEUTILITIESLIGHT_INCLUDE_DIR AND IMAGEUTILITIESLIGHT_LIBRARY_DIR)
  # in cache already
#  set(IMAGEUTILITIES_FOUND TRUE)

#else(IMAGEUTILITIESLIGHT_INCLUDE_DIR AND IMAGEUTILITIESLIGHT_LIBRARY_DIR)

    # derive directories from the environment variable
  set(IMAGEUTILITIESLIGHT_INCLUDE_DIR "$ENV{IMAGEUTILITIES_ROOT}/include" CACHE PATH "The include directory of the Imageutilities Library")

  set(POTENTIAL_LIBRARY_PATHS
    $ENV{IMAGEUTILITIES_ROOT}/lib
    $ENV{IMAGEUTILITIES_ROOT}/lib64
	)

  set(IU_MODULES "iucore")    # always include core module
  if(ImageUtilitiesLight_FIND_COMPONENTS)
  foreach(component ${ImageUtilitiesLight_FIND_COMPONENTS})
    string(TOLOWER ${component} _COMPONENT)
    set(IU_MODULES ${IU_MODULES} ${_COMPONENT})
  endforeach(component)
  endif(ImageUtilitiesLight_FIND_COMPONENTS)
  
  list(REMOVE_DUPLICATES IU_MODULES)
  if(NOT ImageUtilitiesLight_FIND_QUIETLY)
    message(STATUS "Imageutilities requested components: ${IU_MODULES}")
  endif()

#
  # Find absolute path to all libraries!
  #
  # the convention is that shared libraries are preferred over static libraries if both exist (this behaviour
  # can be switched using IMAGEUTILITIESLIGHT_PREFER_SHARED) and
  # that release versions are preferred over debug versions if both exist
  # Under Windows, we also prefer DLLs over static libs, however, if we are able to find both debug and
  # release versions, we use both of them for the visual studio project via the "optimized ... debug ..." clause
  #

  set(QUERY_STRING_STATIC_LIB "_static")
  set(QUERY_STRING_DEBUG_LIB "d")
  set(IMAGEUTILITIESLIGHT_PREFER_SHARED_LIBRARIES true)

  foreach(module ${IU_MODULES})

    if(IMAGEUTILITIESLIGHT_PREFER_SHARED_LIBRARIES)
      set(IMAGEUTILITIESLIGHT_NAMES_DEBUG   ${module}${QUERY_STRING_DEBUG_LIB} ${module}${QUERY_STRING_STATIC_LIB}${QUERY_STRING_DEBUG_LIB})
      set(IMAGEUTILITIESLIGHT_NAMES_RELEASE ${module} ${module}${QUERY_STRING_STATIC_LIB})
    else(IMAGEUTILITIESLIGHT_PREFER_SHARED_LIBRARIES)
      set(IMAGEUTILITIESLIGHT_NAMES_DEBUG   ${module}${QUERY_STRING_STATIC_LIB}${QUERY_STRING_DEBUG_LIB} ${module}${QUERY_STRING_DEBUG_LIB})
      set(IMAGEUTILITIESLIGHT_NAMES_RELEASE ${module}${QUERY_STRING_STATIC_LIB} ${module})
    endif(IMAGEUTILITIESLIGHT_PREFER_SHARED_LIBRARIES)

    unset(LIBRARY_RELEASE CACHE)
    unset(LIBRARY_DEBUG CACHE)
    find_library(LIBRARY_RELEASE
      NAMES ${IMAGEUTILITIESLIGHT_NAMES_RELEASE}
      PATHS ${POTENTIAL_LIBRARY_PATHS}
      PATH_SUFFIXES release
      NO_DEFAULT_PATH
    )
    find_library(LIBRARY_DEBUG
      NAMES ${IMAGEUTILITIESLIGHT_NAMES_DEBUG}
      PATHS ${POTENTIAL_LIBRARY_PATHS}
      PATH_SUFFIXES debug
      NO_DEFAULT_PATH
    )
    
    if(NOT LIBRARY_DEBUG AND NOT LIBRARY_RELEASE AND ImageUtilitiesLight_FIND_REQUIRED)
      message(FATAL_ERROR "${module} marked as required but not found")
    endif(NOT LIBRARY_DEBUG AND NOT LIBRARY_RELEASE AND ImageUtilitiesLight_FIND_REQUIRED)

    # 1. case: we only found a release library
    IF(LIBRARY_RELEASE AND NOT LIBRARY_DEBUG)
      SET(CURRENT_LIBRARY ${LIBRARY_RELEASE})
      # we derive the library dir from the common lib
      get_filename_component(IMAGEUTILITIESLIGHT_LIBRARY_DIR "${LIBRARY_RELEASE}" PATH)
      set(IMAGEUTILITIESLIGHT_LIBRARY_DIR "${IMAGEUTILITIESLIGHT_LIBRARY_DIR}" CACHE STRING "Library directory for variational method libraries")
    ENDIF(LIBRARY_RELEASE AND NOT LIBRARY_DEBUG)

    # 2. case: we only found a debug library
    IF(LIBRARY_DEBUG AND NOT LIBRARY_RELEASE)
      SET(CURRENT_LIBRARY ${LIBRARY_DEBUG})
      # we derive the library dir from the common lib
      get_filename_component(IMAGEUTILITIESLIGHT_LIBRARY_DIR "${LIBRARY_DEBUG}" PATH)
      set(IMAGEUTILITIESLIGHT_LIBRARY_DIR "${IMAGEUTILITIESLIGHT_LIBRARY_DIR}" CACHE PATH "Library directory for variational method libraries")
    ENDIF(LIBRARY_DEBUG AND NOT LIBRARY_RELEASE)

    # 3. case: we found debug and release library
    IF(LIBRARY_RELEASE AND LIBRARY_DEBUG)
    IF(WIN32 AND NOT CYGWIN)
        SET(CURRENT_LIBRARY optimized ${LIBRARY_RELEASE} debug ${LIBRARY_DEBUG})
        # we derive the library dir from the common lib
        get_filename_component(IMAGEUTILITIESLIGHT_LIBRARY_DIR "${LIBRARY_RELEASE}" PATH)
        string(REPLACE "/release" "" IMAGEUTILITIESLIGHT_LIBRARY_DIR ${IMAGEUTILITIESLIGHT_LIBRARY_DIR})
        string(REPLACE "/Release" "" IMAGEUTILITIESLIGHT_LIBRARY_DIR ${IMAGEUTILITIESLIGHT_LIBRARY_DIR})
        set(IMAGEUTILITIESLIGHT_LIBRARY_DIR "${IMAGEUTILITIESLIGHT_LIBRARY_DIR}" CACHE PATH "Library directory for variational method libraries")
      ELSE(WIN32 AND NOT CYGWIN)
        SET(CURRENT_LIBRARY ${LIBRARY_RELEASE})
        # we derive the library dir from the common lib
        get_filename_component(IMAGEUTILITIESLIGHT_LIBRARY_DIR "${LIBRARY_RELEASE}" PATH)
        set(IMAGEUTILITIESLIGHT_LIBRARY_DIR "${IMAGEUTILITIESLIGHT_LIBRARY_DIR}" CACHE PATH "Library directory for variational method libraries")
      ENDIF(WIN32 AND NOT CYGWIN)
    ENDIF(LIBRARY_RELEASE AND LIBRARY_DEBUG)

    #append lib to list
    set(IMAGEUTILITIES_LIBRARY ${IMAGEUTILITIES_LIBRARY} ${CURRENT_LIBRARY})

  endforeach(module)


  #
  # additional dependencies
  #

  list(FIND IU_MODULES iuio USE_IUIO)    # using iuio reuqires opencv
  if(USE_IUIO GREATER 0)
    if(NOT ImageUtilitiesLight_FIND_QUIETLY)
      message(STATUS "ImageUtilities using iuio, adding OpenCV as dependency")
    endif()
    
    find_package(OpenCV REQUIRED COMPONENTS opencv_core opencv_highgui)

    if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV not found (required by Imageutilities iuio")
    endif()

    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake/)
    find_package(FlyCapture2 QUIET)
    if(FLYCAPTURE2_FOUND)
      add_definitions(-DIUIO_PGR)
    endif(FLYCAPTURE2_FOUND)
    
    find_package(OpenEXR QUIET)
    if(OPENEXR_FOUND)
      add_definitions(-DIUIO_EXR)

      find_package(Eigen3 QUIET)
      if(EIGEN3_FOUND)
          add_definitions(-DIUIO_EIGEN3)
      endif(EIGEN3_FOUND)
    endif(OPENEXR_FOUND)
      
    set(IMAGEUTILITIES_LIBRARY ${IMAGEUTILITIES_LIBRARY} ${OpenCV_LIBS})
  endif()



  #
  # logic check if lib file are found!!
  #

  if(IMAGEUTILITIES_LIBRARY)
    set(IMAGEUTILITIESLIGHT_FOUND TRUE)
          
    # set the libraries variable
    unset(IMAGEUTILITIESLIGHT_LIBRARIES CACHE)
    set(IMAGEUTILITIESLIGHT_LIBRARIES ${IMAGEUTILITIES_LIBRARY} CACHE STRING "VMLibraries list of found libraries")
  endif(IMAGEUTILITIES_LIBRARY)
  
  if(IMAGEUTILITIESLIGHT_FOUND)
    if(NOT ImageUtilitiesLight_FIND_QUIETLY)
      message(STATUS "Found Imageutilities: ${IMAGEUTILITIESLIGHT_LIBRARY_DIR} ${IMAGEUTILITIESLIGHT_INCLUDE_DIR}")
    endif()
  else()
    if(ImageUtilitiesLight_FIND_REQUIRED)
      message(FATAL_ERROR "Could not find Imageutilities")
    endif()
  endif()

  
#endif(IMAGEUTILITIESLIGHT_INCLUDE_DIR AND IMAGEUTILITIESLIGHT_LIBRARY_DIR)
