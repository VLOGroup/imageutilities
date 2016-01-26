ImageUtilities - Bridge the gap between CPU and GPU
===================================================

Installation:
-------------

Set up environment variable IMAGEUTILITIES_ROOT pointing to the root folder of the repository.  

Set up environment variable COMPUTE_CAPABILITY with the CC of your CUDA-enabled GPU  

In the root directory (where this file is):  

`$ cd build
$ cmake ..
$ make
$ make install`

Usage:
------

In your cmake-based project include the following lines  
`set(ImageUtilitiesLight_DIR $ENV{IMAGEUTILITIES_ROOT})
find_package(ImageUtilitiesLight REQUIRED COMPONENTS iucore)
include_directories(${IMAGEUTILITIESLIGHT_INCLUDE_DIR})
`

and link your application with  
`target_link_libraries(your_application
  your_libraries
  ${IMAGEUTILITIESLIGHT_LIBRARIES}
)`
