cmake_minimum_required(VERSION 3.5)
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE "Release")
endif()

project(deep_feedback_learning VERSION 1.0.1 LANGUAGES CXX)
include(GNUInstallDirs)

add_subdirectory(tests_c)

set(LIBSRC
  dfl/neuron.cpp
  dfl/bandpass.cpp
  dfl/layer.cpp
  deep_feedback_learning.cpp)

set(PUBLICHEADERS
  deep_feedback_learning.h
  )

set(PRIVATEHEADERS
  dfl/neuron.h
  dfl/bandpass.h
  dfl/layer.h
  dfl/globals.h
  )
  

if(UNIX)
  
  add_library(deep_feedback_learning
    SHARED
    ${LIBSRC}
    )
  
  target_include_directories(deep_feedback_learning PRIVATE dfl)
  
  set_target_properties(deep_feedback_learning PROPERTIES
    SOVERSION 1
    VERSION ${PROJECT_VERSION}
    PUBLIC_HEADER "${PUBLICHEADERS}"
    PRIVATE_HEADER "${PRIVATEHEADERS}")
  
  install(TARGETS deep_feedback_learning
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    PRIVATE_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dfl)
  
endif(UNIX)

  
add_library(deep_feedback_learning_static
  STATIC
  ${LIBSRC}
  )

target_link_libraries(deep_feedback_learning_static pthread)

target_include_directories(deep_feedback_learning_static PRIVATE dfl)

set_target_properties(deep_feedback_learning_static PROPERTIES
  VERSION ${PROJECT_VERSION}
  PUBLIC_HEADER "${PUBLICHEADERS}"
  PRIVATE_HEADER "${PRIVATEHEADERS}")

install(TARGETS deep_feedback_learning_static
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PRIVATE_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dfl)
