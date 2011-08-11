# - Try to find ImageUtilities include dir and library
# This script locates the ICG ImageUtilities.
#
# TODO documentation
#
# IU_FOUND
# IU_ROOT_DIR
# IU_INCLUDE_DIRS
# IU_LIBRARIES
# IU_LIBRARY_DIR
#
# IU_EXT_DEP_LIBS is used to combine all external dependencies into a single variable....
# IU_DEFINITIONS is used to combine all definitions for e.g. compiler switches...


# Use FIND_PACKAGE(ImageUtilities COMPONENTS ...) to enable modules
if(ImageUtilities_FIND_COMPONENTS)
  foreach(component ${ImageUtilities_FIND_COMPONENTS})
    string(TOUPPER ${component} _COMPONENT)
    set(IU_USE_${_COMPONENT} 1)
    # message(STATUS "adding component IU_USE_${_COMPONENT}")
  endforeach(component)
  
  # To make sure we always use IUCORE when not specified in components
  # not that this behaviour is different to e.g. Qt4 -> we always need the core module for the other stuff!
  if(NOT IU_USE_IUCORE)
    set(IU_USE_IUCORE 1)
  endif(NOT IU_USE_IUCORE)
else(ImageUtilities_FIND_COMPONENTS)
  set(IU_USE_IUCORE 1)
endif(ImageUtilities_FIND_COMPONENTS)

# set path to use file
get_filename_component(VMLIBRARIES_CMAKE_ROOT "${CMAKE_CURRENT_LIST_FILE}" PATH CACHE)
set(VMLIBRARIES_DIR $ENV{VMLIBRARIES_ROOT} CACHE PATH "basepath for vmlibraries")
if(NOT VMLIBRARIES_DIR)
  set(VMLIBRARIES_DIR = ${VMLIBRARIES_CMAKE_ROOT}/../ CACHE PATH "basepath for vmlibraries")
endif(NOT VMLIBRARIES_DIR)
set(IU_USE_FILE ${VMLIBRARIES_CMAKE_ROOT}/UseImageUtilities.cmake CACHE FILEPATH "USE file for including the correct headers and libs.")

# set a variable for all possible modules
set(IU_MODULES iucore iuipp iumatlab iugui iuio iuiopgm iuvideocapture iupgrcamera)

################################################################################
#
#       Setting the INCLUDE and LIB Variables
#
################################################################################

if(IU_INCLUDE_DIRS AND IU_LIBRARY_DIR)
  # in cache already
  set(IU_FOUND TRUE)

else(IU_INCLUDE_DIRS AND IU_LIBRARY_DIR)

  # 

  #
  ## local installed (svn) version 
  #
  
  # include path
  set(IU_INCLUDE_DIRS "${VMLIBRARIES_DIR}/include" "${VMLIBRARIES_DIR}/include/iu" CACHE PATH "Include dir of the ImageUtilities library.")
  # check include dir
  ## TODO FIXME

  set(VMLIBRARIES_COMMON_INCLUDE_DIR "${VMLIBRARIES_DIR}/common/include" CACHE PATH "Common include dir of the vmlibraries project for small helper functions.")

  # library path
  set(POTENTIAL_LIBRARY_PATHS
    ${VMLIBRARIES_DIR}/lib
    # ${VMLIBRARIES_DIR}/lib/debug
    # ${VMLIBRARIES_DIR}/lib/release
    )

  # Find absolute path for the library!
  #
  # the convention is that shared libraries are preferred over static
  # libraries if both exist (this behaviour can be switched using
  # IU_PREFER_SHARED_LIBRARIES) and that release versions are preferred over
  # debug versions if both exist. Under Windows, we also prefer DLLs over
  # static libs, however, if we are able to find both debug and release
  # versions, we use both of them for the visual studio project via the
  # "optimized ... debug ..." clause
  #

  set(STATIC_APPEND "_static")
  set(DEBUG_APPEND "d")

  set(IU_PREFER_SHARED_LIBRARIES true)

  # search the lib for every module
  FOREACH(IU_MODULE ${IU_MODULES})
    # message(STATUS "searching library for IU_MODULE=${IU_MODULE}")
    string(TOUPPER ${IU_MODULE} _UPPER_IU_MODULE)
    
    # potential names
    if(IU_PREFER_SHARED_LIBRARIES)
      set(IU_NAMES_${_UPPER_IU_MODULE}_LIBRARY_DEBUG ${IU_MODULE}${DEBUG_APPEND} ${IU_MODULE}${STATIC_APPEND}${DEBUG_APPEND})
      set(IU_NAMES_${_UPPER_IU_MODULE}_LIBRARY_RELEASE ${IU_MODULE} ${IU_MODULE}${STATIC_APPEND})
    else(IU_PREFER_SHARED_LIBRARIES)
      set(IU_NAMES_${_UPPER_IU_MODULE}_LIBRARY_DEBUG ${IU_MODULE}${STATIC_APPEND}${DEBUG_APPEND} ${IU_MODULE}${DEBUG_APPEND})
      set(IU_NAMES_${_UPPER_IU_MODULE}_LIBRARY_RELEASE ${IU_MODULE}${STATIC_APPEND} ${IU_MODULE})
    endif(IU_PREFER_SHARED_LIBRARIES)

    # message(STATUS "POTENTIAL_LIBRARY_PATHS=${POTENTIAL_LIBRARY_PATHS}")
    # message(STATUS "prefer shared: potential names=IU_NAMES_${_UPPER_IU_MODULE}_LIBRARY_DEBUG ${IU_MODULE}${DEBUG_APPEND} ${IU_MODULE}${STATIC_APPEND}${DEBUG_APPEND}")
    # message(STATUS "prefer shared: potential names=IU_NAMES_${_UPPER_IU_MODULE}_LIBRARY_RELEASE ${IU_MODULE} ${IU_MODULE}${STATIC_APPEND}")
    # message(STATUS "prefer static: potential names=IU_NAMES_${_UPPER_IU_MODULE}_LIBRARY_DEBUG ${IU_MODULE}${STATIC_APPEND}${DEBUG_APPEND} ${IU_MODULE}${DEBUG_APPEND}")
    # message(STATUS "prefer static: potential names=IU_NAMES_${_UPPER_IU_MODULE}_LIBRARY_RELEASE ${IU_MODULE}${STATIC_APPEND} ${IU_MODULE})")

    # message(STATUS "IU_NAMES_${_UPPER_IU_MODULE}_LIBRARY_RELEASE=${IU_NAMES_IUCORE_LIBRARY_RELEASE}")

    # find the libraries of the selected module
    find_library(IU_${_UPPER_IU_MODULE}_LIBRARY_RELEASE
      NAMES ${IU_NAMES_${_UPPER_IU_MODULE}_LIBRARY_RELEASE}
      PATHS ${POTENTIAL_LIBRARY_PATHS}
      PATH_SUFFIXES release
      NO_DEFAULT_PATH
      )
    
    find_library(IU_${_UPPER_IU_MODULE}_LIBRARY_DEBUG
      NAMES ${IU_NAMES_${_UPPER_IU_MODULE}_LIBRARY_DEBUG}
      PATHS ${POTENTIAL_LIBRARY_PATHS}
      PATH_SUFFIXES debug
      NO_DEFAULT_PATH
      )

    # message("found libraries: IU_CORE_LIBRARY_RELEASE = ${IU_CORE_LIBRARY_RELEASE}")
    # message("found libraries: IU_CORE_LIBRARY_DEBUG = ${IU_CORE_LIBRARY_DEBUG}")
    
    # check which library was found:

    # 1. case: we only found a release library
    if(IU_${_UPPER_IU_MODULE}_LIBRARY_RELEASE AND NOT IU_${_UPPER_IU_MODULE}_LIBRARY_DEBUG)
      set(IU_CURRENT_${_UPPER_IU_MODULE}_LIBRARY ${IU_${_UPPER_IU_MODULE}_LIBRARY_RELEASE})

    # 2. case: we only found a debug library
    elseif(IU_${_UPPER_IU_MODULE}_LIBRARY_DEBUG AND NOT IU_${_UPPER_IU_MODULE}_LIBRARY_RELEASE) #
      set(IU_CURRENT_${_UPPER_IU_MODULE}_LIBRARY ${IU_${_UPPER_IU_MODULE}_LIBRARY_DEBUG})

    # 3. case: we found debug and release library
    elseif(IU_${_UPPER_IU_MODULE}_LIBRARY_RELEASE AND IU_${_UPPER_IU_MODULE}_LIBRARY_DEBUG)
      if(WIN32 AND NOT CYGWIN)
        set(IU_CURRENT_${_UPPER_IU_MODULE}_LIBRARY optimized ${IU_${_UPPER_IU_MODULE}_LIBRARY_RELEASE} debug ${IU_${_UPPER_IU_MODULE}_LIBRARY_DEBUG})
      else(WIN32 AND NOT CYGWIN)
        set(IU_CURRENT_${_UPPER_IU_MODULE}_LIBRARY ${IU_${_UPPER_IU_MODULE}_LIBRARY_RELEASE})
      endif(WIN32 AND NOT CYGWIN)

    endif(IU_${_UPPER_IU_MODULE}_LIBRARY_RELEASE AND NOT IU_${_UPPER_IU_MODULE}_LIBRARY_DEBUG)

    # set the actual library
    set(IU_${_UPPER_IU_MODULE}_LIBRARY ${IU_CURRENT_${_UPPER_IU_MODULE}_LIBRARY} CACHE FILEPATH "ImageUtilities library for module ${IU_MODULE}.")
    if(IU_${_UPPER_IU_MODULE}_LIBRARY)
      set(IU_${_UPPER_IU_MODULE}_FOUND 1)
    endif(IU_${_UPPER_IU_MODULE}_LIBRARY)
    mark_as_advanced(IU_${_UPPER_IU_MODULE}_LIBRARY IU_${_UPPER_IU_MODULE}_LIBRARY_RELEASE IU_${_UPPER_IU_MODULE}_LIBRARY_DEBUG)
    # message(STATUS "IU_${_UPPER_IU_MODULE}_LIBRARY=${IU_${_UPPER_IU_MODULE}_LIBRARY}")
    
  ENDFOREACH(IU_MODULE)


  ## derive library dir from core module (this one must be available)
  if(IU_IUCORE_FOUND)
    get_filename_component(IU_LIBRARY_DIR "${IU_IUCORE_LIBRARY}" PATH)
    string(REPLACE "/release" "" IU_LIBRARY_DIR ${IU_LIBRARY_DIR})
    string(REPLACE "/debug" "" IU_LIBRARY_DIR ${IU_LIBRARY_DIR})

    if(NOT IU_FIND_QUIETLY)
      message(STATUS "Found ImageUtilities (iu library):")
      message(STATUS "      IU_INCLUDE_DIRS = ${IU_INCLUDE_DIRS}")
      message(STATUS "      IU_LIBRARY_DIR = ${IU_LIBRARY_DIR}")
    endif(NOT IU_FIND_QUIETLY)

  else(IU_IUCORE_FOUND)
    if(IU_FIND_REQUIRED)
      message(WARNING "Could not find ImageUtilties library but marked as required!")
    endif(IU_FIND_REQUIRED)

  endif(IU_IUCORE_FOUND)


  ################################################################################
  #
  #       Module Dependencies
  #
  ################################################################################
  # external dependencies
  set(IU_IUCORE_LIB_DEPENDENCIES "")
  set(IU_IUIPP_LIB_DEPENDENCIES "")
  set(IU_IUMATLAB_LIB_DEPENDENCIES "")
  set(IU_IUGUI_LIB_DEPENDENCIES "")
  set(IU_IUIO_LIB_DEPENDENCIES "")
  set(IU_IUIOPGM_LIB_DEPENDENCIES "")
  set(IU_IUVIDEOCAPTURE_LIB_DEPENDENCIES "")
  set(IU_IUPGRCAMERA_LIB_DEPENDENCIES "")

  ## CORE module
  if(IU_IUCORE_FOUND)
    # CUDA
    find_package(CUDA 3.1 REQUIRED)
    find_package(CUDASDK REQUIRED)
    if(CUDA_FOUND AND CUDASDK_FOUND)
      cuda_include_directories(${CUDA_INCLUDE_DIRS} ${CUDA_CUT_INCLUDE_DIR})
      include_directories(${CUDA_INCLUDE_DIRS} ${CUDA_CUT_INCLUDE_DIR})
      set(IU_IUCORE_LIB_DEPENDENCIES ${IU_IUCORE_LIB_DEPENDENCIES} ${CUDA_LIBRARIES})

      # Checking cuda version
      # set defines due to some missing functions in cuda 3.1
      if((CUDA_VERSION_MAJOR EQUAL 3) AND (CUDA_VERSION_MINOR EQUAL 1))
        # using CUDA 3.1
        message(STATUS "IU: using CUDA 3.1")
        set(IU_DEFINITIONS ${IU_DEFINITIONS} -DCUDA_VERSION_31)
      endif()

      if((CUDA_VERSION_MAJOR EQUAL 3) AND (CUDA_VERSION_MINOR EQUAL 2))
        # using CUDA 3.2
        message(STATUS "IU: using CUDA 3.2")
        set(IU_DEFINITIONS ${IU_DEFINITIONS} -DCUDA_VERSION_32)
      elseif()
        message(STATUS "unknown CUDA version. some things might not be tested.")
      endif()

      # CUDA Sparse
      find_package(CUDASparse QUIET)
      if(CUDASparse_FOUND)
        message("Cuda sparse lib found")
        #include_directories(${CUDA_INCLUDE_DIRS})
        set(IU_IUCORE_LIB_DEPENDENCIES ${IU_IUCORE_LIB_DEPENDENCIES} ${CUDA_SPARSE_LIBRARY})
      endif(CUDASparse_FOUND)

    endif(CUDA_FOUND AND CUDASDK_FOUND)
  endif(IU_IUCORE_FOUND)

  ## IPP module
  if(IU_IUIPP_FOUND)
    # IPP
    find_package( IPP QUIET )
    if(IPP_INCLUDE_DIR)
      include_directories(${IPP_INCLUDE_DIRS})
      set(IU_IUIPP_LIB_DEPENDENCIES ${IU_IUIPP_LIB_DEPENDENCIES} ${IPP_LIBRARIES})
    endif(IPP_INCLUDE_DIR)

  endif(IU_IUIPP_FOUND)

  ## MATLAB module
  if(IU_IUMATLAB_FOUND)
    # MATLAB
    FIND_PACKAGE(Matlab_ICG)

    if(MATLAB_FOUND)
      include_directories(${MATLAB_INCLUDE_DIRS})
      set(IU_IUMATLAB_LIB_DEPENDENCIES ${IU_IUMATLAB_LIB_DEPENDENCIES} ${MATLAB_LIBRARIES})
    endif(MATLAB_FOUND)

  endif(IU_IUMATLAB_FOUND)


  ## GUI module
  if(IU_IUGUI_FOUND)
    # Qt4
    find_package(Qt4 COMPONENTS QtCore QtGui QtOpenGL)
    if(QT4_FOUND)
      include(${QT_USE_FILE})

       ## GLEW
       find_package( GLEW REQUIRED )
       include_directories(${GLEW_INCLUDE_DIR})
       ## OpenGL
       find_package( OpenGL REQUIRED )
       include_directories(${OPENGL_INCLUDE_DIR})

      set(IU_IUGUI_LIB_DEPENDENCIES ${IU_IUGUI_LIB_DEPENDENCIES} ${QT_LIBRARIES} ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES})
 
      find_package(Qwt)
      IF(QWT_FOUND)
        add_definitions(-DUSE_QWT)
	include_directories(${QWT_INCLUDE_DIR})
	set(IU_IUGUI_LIB_DEPENDENCIES ${IU_IUGUI_LIB_DEPENDENCIES} ${QWT_LIBRARIES})
      ENDIF(QWT_FOUND)
    endif(QT4_FOUND)


  endif(IU_IUGUI_FOUND)

  ## IO module
  if(IU_IUIO_FOUND)
    # OpenCV
    find_package( OpenCV QUIET )
    if(OpenCV_INCLUDE_DIRS)
      include_directories(${OpenCV_INCLUDE_DIRS})
      set(IU_IUIO_LIB_DEPENDENCIES ${IU_IUIO_LIB_DEPENDENCIES} ${OpenCV_LIBS})
    endif(OpenCV_INCLUDE_DIRS)

  endif(IU_IUIO_FOUND)

  ## IOPGM module
  # no external dependencies

  ## VIDEOCAPTURE module
  if(IU_IUVIDEOCAPTURE_FOUND)
    # Qt4 (if no gui module is used)
    if(NOT QT4_FOUND)
      find_package(Qt4 COMPONENTS QtCore)
      if(QT4_FOUND)
        include(${QT_USE_FILE})
        set(IU_IUVIDEOCAPTURE_LIB_DEPENDENCIES ${IU_IUVIDEOCAPTURE_LIB_DEPENDENCIES} ${QT_LIBRARIES})
      endif(QT4_FOUND)
    endif(NOT QT4_FOUND)

    # OpenCV
    if(NOT OpenCV_INCLUDE_DIRS)
      find_package( OpenCV QUIET )
      if(OpenCV_INCLUDE_DIRS)
        include_directories(${OpenCV_INCLUDE_DIRS})
        set(IU_IUVIDEOCAPTURE_LIB_DEPENDENCIES ${IU_IUVIDEOCAPTURE_LIB_DEPENDENCIES} ${OpenCV_LIBS})
      endif(OpenCV_INCLUDE_DIRS)
    endif(NOT OpenCV_INCLUDE_DIRS)
  endif(IU_IUVIDEOCAPTURE_FOUND)


  ## PGRCAMERA module
  if(IU_IUPGRCAMERA_FOUND)
    # Qt4 (if no gui module is used)
    if(NOT QT4_FOUND)
      find_package(Qt4 COMPONENTS QtCore)
      if(QT4_FOUND)
        include(${QT_USE_FILE})
        set(IU_IUPGRCAMERA_LIB_DEPENDENCIES ${IU_IUPGRCAMERA_LIB_DEPENDENCIES} ${QT_LIBRARIES})
      endif(QT4_FOUND)
    endif(NOT QT4_FOUND)

    # Flycapture2 (pointgrey stuff)
    find_package( FLYCAPTURE2 QUIET )
    if(FLYCAPTURE2_INCLUDE_DIRS)
      include_directories(${FLYCAPTURE2_INCLUDE_DIRS})
      set(IU_IUPGRCAMERA_LIB_DEPENDENCIES ${IU_IUPGRCAMERA_LIB_DEPENDENCIES} ${FLYCAPTURE2_LIBRARIES})
    endif(FLYCAPTURE2_INCLUDE_DIRS)

  endif(IU_IUPGRCAMERA_FOUND)

  mark_as_advanced(
    IU_IUCORE_LIB_DEPENDENCIES 
    IU_IUIPP_LIB_DEPENDENCIES 
    IU_IUMATLAB_LIB_DEPENDENCIES 
    IU_IUGUI_LIB_DEPENDENCIES
    IU_IUIO_LIB_DEPENDENCIES
    IU_IUIOPGM_LIB_DEPENDENCIES
    IU_IUVIDEOCAPTURE_LIB_DEPENDENCIES
    IU_IUPGRCAMERA_LIB_DEPENDENCIES
    IU_INCLUDE_DIRS 
    IU_LIBRARY_DIR
    IU_DEFINITIONS
    )

endif(IU_INCLUDE_DIRS AND IU_LIBRARY_DIR)
