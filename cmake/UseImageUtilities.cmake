# - Use Module for ImageUtilities
# Sets up the needed files (headers and source) for compiling the needed functionality directly into your application.

# IU_LIBRARIES contains all needed libraries afterwards. Note that also the
# dependencies are included (like e.g. Qt or OpenCV when activated)
# IU_LIB_DEPENDENCIES all the external dependencies besides the iu libs themselfs

# get base path where this cmake files is
get_filename_component(IU_USE_ROOT "${CMAKE_CURRENT_LIST_FILE}" PATH)

set(IU_LIBRARIES "") # we start with no variable and add all the needed ones
set(IU_LIB_DEPENDENCIES "") # we start with no external dependencies

#
## Mandatory libraries
#

##-----------------------------------------------------------------------------
## include ImageUtilities include directory (done here because the cuda macro has to be called beforehand
include_directories(${IU_INCLUDE_DIRS})
cuda_include_directories(${IU_INCLUDE_DIRS})

include_directories(${VMLIBRARIES_COMMON_INCLUDE_DIR})
cuda_include_directories(${VMLIBRARIES_COMMON_INCLUDE_DIR})

# IU modules
foreach (module IUCORE IUIPP IUGUI IUIO IUIOPGM IUVIDEOCAPTURE IUPGRCAMERA)
  if (IU_USE_${module} OR IU_USE_${module}_DEPENDS)

    if (IU_${module}_FOUND)
      set(IU_LIBRARIES ${IU_LIBRARIES} ${IU_${module}_LIBRARY})
      set(IU_LIB_DEPENDENCIES ${IU_LIB_DEPENDENCIES} ${IU_${module}_LIB_DEPENDENCIES})
      if (module MATCHES IUGUI)
        # for the gui module we have to read the QT_USE_FILE
        include(${QT_USE_FILE})
      endif (module MATCHES IUGUI)

    else (IU_${module}_FOUND)
      message("ImageUtilities module ${module} library not found.")
    endif (IU_${module}_FOUND)

  endif (IU_USE_${module} OR IU_USE_${module}_DEPENDS)

endforeach(module)

# concat the external lib dependencies to IU_LIBRARIES
set(IU_LIBRARIES ${IU_LIBRARIES} ${IU_LIB_DEPENDENCIES})
mark_as_advanced(IU_LIBRARIES)

# set definitions
add_definitions(${IU_DEFINITIONS})
