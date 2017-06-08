cmake_minimum_required(VERSION 2.8)
project(imageutilities)

enable_testing()

##-----------------------------------------------------------------------------
# CUDA
if(ANDROID)
    find_package(CUDAAndroid 5.0 REQUIRED)
else(ANDROID)
    find_package(CUDA 5.0 REQUIRED)    # texture objects are introduced in 5.0
endif(ANDROID)

cuda_include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

if(WIN32)
  string(REPLACE "\\" "/" CUDA_SDK_ROOT_DIR $ENV{CUDA_SDK_ROOT_DIR})
else(WIN32)
  set(CUDA_SDK_ROOT_DIR $ENV{CUDA_SDK_ROOT_DIR})
endif(WIN32)

if(NOT EXISTS ${CUDA_SDK_ROOT_DIR}/common/inc)
message("WRONG CUDA_SDK_ROOT_DIR:" ${CUDA_SDK_ROOT_DIR})
endif()
cuda_include_directories(${CUDA_SDK_ROOT_DIR}/common/inc)
include_directories(${CUDA_SDK_ROOT_DIR}/common/inc)

set(IU_IUCORE_LIB_DEPENDENCIES ${IU_IUCORE_LIB_DEPENDENCIES} ${CUDA_LIBRARIES})


#------------- for compiler flags see included file---------------------------------
include(${CMAKE_CURRENT_LIST_DIR}/iu_settings.cmake.txt)
#-----------------------------------------------------------------------------------

# This is to allow correct linking static libraries in windows, see Cmake's GENERATE_EXPORT_HEADER and WINDOWS_EXPORT_ALL_SYMBOLS, for a better automation of SHARED/STATIC on different platforms
if(NOT BUILD_SHARED_LIBS) # 
	add_definitions(-DIUCORE_USE_STATIC)
	add_definitions(-DIUGUI_USE_STATIC)
	add_definitions(-DIUIO_USE_STATIC)
	add_definitions(-DIUMATH_USE_STATIC)
	add_flags(CUDA_NVCC_FLAGS "-DIUCORE_USE_STATIC")
	add_flags(CUDA_NVCC_FLAGS "-DIUGUI_USE_STATIC")
	add_flags(CUDA_NVCC_FLAGS "-DIUIO_USE_STATIC")
	add_flags(CUDA_NVCC_FLAGS "-DIUMATH_USE_STATIC")
	message(STATUS "CUDA_NVCC_FLAGS= ${CUDA_NVCC_FLAGS}")
endif()

add_subdirectory(src)
add_subdirectory(tests)
