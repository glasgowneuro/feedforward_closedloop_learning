cmake_minimum_required(VERSION 3.11.0)

project(cldl VERSION 3.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

if (MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wconversion -Wextra -pedantic)
endif()

include(GNUInstallDirs)

add_definitions(-D_CRT_SECURE_NO_WARNINGS)

set(LIBSRC
  cldl_layer.cpp
  cldl_net.cpp
  cldl_neuron.cpp
  cldl_filterbank.cpp
  cldl_bandpass.cpp
  )

set(PUBLICHEADERS
  cldl_net.h
  cldl_filterbank.h
  )

set(PRIVATEHEADERS
  cldl_layer.h
  cldl_net.h
  cldl_neuron.h
  cldl_globals.h
  cldl_bandpsss.h
  )
  

add_library(cldl_static
  STATIC
  ${LIBSRC}
  )

find_package(Threads)

target_link_libraries(cldl_static ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(cldl_static PRIVATE cldl)

set_target_properties(cldl_static PROPERTIES
  VERSION ${PROJECT_VERSION}
  PUBLIC_HEADER "${PUBLICHEADERS}"
  PRIVATE_HEADER "${PRIVATEHEADERS}")

install(TARGETS cldl_static
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PRIVATE_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cldl)
