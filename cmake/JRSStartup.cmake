# This file is heavily based on work by HS-Art Digital Service.
#
#  IMPORTANT: the statement 'project(..)' has to go BEFORE including this file!!!
#

#==============================================================================
# Set output param strings  out_os and out_cc
#
# e.g.: out_os = w32
#       out_cc = vc90
#==============================================================================
macro( JRSGetPlatformVars out_os out_cc )
  
  # Note that this 'trick' for determining whether we compile for 32 or 64 bit
  # seems to work only for windows.
  # For Mac, Unix etc., its better to set a CMAKE variable 'JRS_COMPILE_ARCH' to 32 or 64
  # Don't forget to set it in your 'generate_os_compiler.sh' script
  # e.g. by 'cmake "-DJRS_COMPILE_ARCH=64" -G "Xcode" [cmakelists_directory]
  # Also, For Mac, Unix etc., its necessary to set a 'JRS_COMPILER' string to
  # the short-name of compiler which is used and its version number
  # e.g "-DJRS_COMPILER=gcc42" -> gcc 4.2 compiler
  if (WIN32)
  if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
    set( ARCH64 1 )
  else( CMAKE_SIZEOF_VOID_P EQUAL 8 )
    set( ARCH64 0 )
  endif( CMAKE_SIZEOF_VOID_P EQUAL 8 )
  else()
	# non-windows (apple os/x, unix, etc..)
    if (JRS_COMPILE_ARCH EQUAL 32)
      set(ARCH64 0)
    elseif(JRS_COMPILE_ARCH EQUAL 64)
      set(ARCH64 1)
    else()
      message(FATAL_ERROR "[JRSGetPlatformVars] On non-windows systems, you have to define desired architecture(32,64bit) as command-line param JRS_COMPILE_ARCH to CMAKE!")
    endif()	
	if (NOT DEFINED JRS_COMPILER)      
      message(FATAL_ERROR "[JRSGetPlatformVars] On non-windows systems, you have to define the compiler and its version in command-line param JRS_COMPILER to CMAKE !")
    endif()
  endif()

  if( MSVC60 )
    set( ${out_cc} vc6 )
  elseif( MSVC71 )
    set( ${out_cc} vc71 )
  elseif( MSVC80 )
    set( ${out_cc} vc80 )
  elseif( MSVC90 )
    set( ${out_cc} vc90 )
  elseif( MSVC11 )
    set( ${out_cc} vc11 )	
  elseif( MINGW )
    set( ${out_cc} mingw )
  endif( MSVC60 )
  
  if( WIN32 )
    if( ARCH64 )
      set( ${out_os} w64 )
    else( ARCH64 )
      set ( ${out_os} w32 )
    endif( ARCH64 )
  elseif( APPLE )
    if( ARCH64 )
      set( ${out_os} osx64 )
    else( ARCH64 )
      set( ${out_os} osx32 )
    endif( ARCH64 )
	# compiler for apple has to be set by command-line parameter 'JRS_COMPILER'
	set(${out_cc} ${JRS_COMPILER})    	
  elseif( UNIX )
    if( ARCH64 )
      set( ${out_os} lx64 )
    else( ARCH64 )
      set( ${out_os} lx32 )
    endif( ARCH64 )
	# compiler for unix has to be set by command-line parameter 'JRS_COMPILER'
	set(${out_cc} ${JRS_COMPILER})    	
  endif( WIN32 )

endmacro( JRSGetPlatformVars )

# Set output param string out_libname
# according to given project name and major/minor version number parameters.
#
# e.g. Windows: 
#       in_projectname   = iplJrs
#       in_version_major = 3 
#       in_version_minor = 1
#       out_libname      = iplJrs3.1_w32_vc90
#
# e.g. Mac/Linux: the version is added in set_target_properties()
#       in_projectname   = iplJrs
#       in_version_major = 3 
#       in_version_minor = 1
#       out_libname      = iplJrs_osx64_gcc42 if shared lib
#       out_libname      = iplJrs_osx64_gcc42.3.1 if static lib
#
#==============================================================================
macro( JRSGetLibName out_libname in_projectname in_version_major in_version_minor )

  JRSGetPlatformVars( os cc )
  set (jgln_variant_id_str "")
  if(WIN32)
	# Note if a variant is set, it is also included in the libname
    set( ${out_libname} "${in_projectname}${jgln_variant_id_str}${in_version_major}.${in_version_minor}_${os}_${cc}")
  endif()
endmacro( )

#FTT use this for findPackage() with Windows and MAC/LINUX libname convention!
#==============================================================================
# Set output param strings out_libname and out_libname_debug
# according to given project name, major/minor version number.
# The name differs for WINDOWS and LINUX/MAC
#
# e.g. Windows: 
#       in_projectname   = iplJrs
#       in_version_major = 3 
#       in_version_minor = 1
#       out_libname      = iplJrs3.1_w32_vc90
#       out_libname_debug= iplJrs3.1_w32_vc90d
#
# in Mac/Linux:
#       in_projectname   = iplJrs
#       in_version_major = 3 
#       in_version_minor = 1
#       out_libname      = iplJrs_osx64_gcc42.3.1
#       out_libname_debug= iplJrs_osx64_gcc42d.3.1
#
# NOTE: if 'version_minor' is set to '-1', it is skipped (so only major version number is appended)
# e.g. JRSGetLibNames (libname libname_dbg myLib 2 -1)
#      will set 'libname' to 'myLib2_w32_vc90' and 'libname_dbg' to 'mylib_w32_vc90d'
# NOTE: If 'variable 'JRSGetLibNamesWithoutVariantFlag' is set to 1, then the variant is _not_ appended
#==============================================================================

macro( JRSGetLibNames out_libname out_libname_debug in_projectname in_version_major in_version_minor )
  JRSGetPlatformVars( os cc )
  set (jgln_variant_id_str "")
 if(WIN32)
  if (${in_version_minor} EQUAL -1)
	# we have only a major version number
	set (tmp378_version_suffix "${in_version_major}")
  else()
	# standard case, we have major and minor version number
	set (tmp378_version_suffix "${in_version_major}.${in_version_minor}")
  endif()
  set( ${out_libname}       "${in_projectname}${jgln_variant_id_str}${tmp378_version_suffix}_${os}_${cc}")
  set( ${out_libname_debug} "${in_projectname}${jgln_variant_id_str}${tmp378_version_suffix}_${os}_${cc}d")
else() # FTT MacOSX and Linux must have it differently
  set( ${out_libname}       "${in_projectname}${jgln_variant_id_str}_${os}_${cc}.${tmp378_version_suffix}")
  set( ${out_libname_debug} "${in_projectname}${jgln_variant_id_str}_${os}_${cc}d.${tmp378_version_suffix}")
endif()
endmacro()

macro ( JRSGenerateLinkLibrary out_libraries in_projectname in_version_major in_version_minor)
	JRSGetLibNames (jgll_libname_release jgll_libname_debug ${in_projectname} ${in_version_major} ${in_version_minor})
	set( ${out_libraries}
      debug      ${jgll_libname_debug}
      optimized  ${jgll_libname_release}
    )
endmacro()

##
## ( ) All necessary JRS macros have been defined. Now do the JRS-specific stuff
##

if (${OPTICALFLOW_FOR_JRS} EQUAL 1)
	message ("Compiling flow for JRS")
	# Get platform vars and save them in JRS_OS and JRS_CC
	JRSGetPlatformVars( JRS_OS JRS_CC )
	message( "[JRS] OS = ${JRS_OS} CC = ${JRS_CC}" )
	
	if (NOT arg_omit_version_number)		
		# append version name
		set (JRS_PROJECT_NAME ${CMAKE_PROJECT_NAME})
		JRSGetLibName(JRS_OUTPUT_LIBNAME ${CMAKE_PROJECT_NAME} ${TARGET_VERSION_MAJOR} ${TARGET_VERSION_MINOR} )
		message ("[JRS] Project name: ${JRS_PROJECT_NAME}  -- Library name: ${JRS_OUTPUT_LIBNAME}")
	endif()
	# postfix which will be added to name for debug configuration
	set (JRS_OUTPUT_DEBUG_POSTFIX "d")

endif()
