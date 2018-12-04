cmake_minimum_required(VERSION 3.5)
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE "Release")
endif()

project(fcl VERSION 1.2.4 LANGUAGES CXX)
include(GNUInstallDirs)

add_subdirectory(tests_c)

set(LIBSRC
  fcl/neuron.cpp
  fcl/bandpass.cpp
  fcl/layer.cpp
  fcl.cpp)

set(PUBLICHEADERS
  fcl.h
  )

set(PRIVATEHEADERS
  fcl/neuron.h
  fcl/bandpass.h
  fcl/layer.h
  fcl/globals.h
  )
  

if(UNIX)
  
  add_library(fcl
    SHARED
    ${LIBSRC}
    )
  
  target_include_directories(fcl PRIVATE fcl)
  
  set_target_properties(fcl PROPERTIES
    SOVERSION 1
    VERSION ${PROJECT_VERSION}
    PUBLIC_HEADER "${PUBLICHEADERS}"
    PRIVATE_HEADER "${PRIVATEHEADERS}")
  
  install(TARGETS fcl
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    PRIVATE_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fcl)
  
endif(UNIX)

  
add_library(fcl_static
  STATIC
  ${LIBSRC}
  )

find_package(Threads)

target_link_libraries(fcl_static ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(fcl_static PRIVATE fcl)

set_target_properties(fcl_static PROPERTIES
  VERSION ${PROJECT_VERSION}
  PUBLIC_HEADER "${PUBLICHEADERS}"
  PRIVATE_HEADER "${PRIVATEHEADERS}")

install(TARGETS fcl_static
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PRIVATE_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fcl)


find_package(Doxygen)

if(DOXYGEN_FOUND)
    add_custom_target(docs
                      COMMAND ${DOXYGEN_EXECUTABLE}
                      COMMENT "Generating API documentation with Doxygen"
                      VERBATIM)
endif()
