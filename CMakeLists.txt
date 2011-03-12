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
# Project     : ImageUtilities
# Module      : overall CMake
# Language    : CMake
# Description : overall CMakeFile for building the ImageUtilities library
#
# Author     : Manuel Werlberger
# EMail      : werlberger@icg.tugraz.at

PROJECT(imageutilities CXX C)
add_subdirectory(src)
add_subdirectory(mex)
add_subdirectory(tests)

#add_custom_target(doc COMMAND cd ${CMAKE_SOURCE_DIR} && doxygen ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
