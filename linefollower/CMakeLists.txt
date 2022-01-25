cmake_minimum_required(VERSION 3.7.0)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(fcl-linefollower LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# required packages
find_package(Qt5 COMPONENTS Widgets Core OpenGL REQUIRED)
find_package(Threads)

# required libraries
find_library(iir REQUIRED)

set(PROJECT_LIBS
  Threads
  comedi
  iir)

add_executable(fcl-linefollower
  Linefollower.cpp
  Racer.cpp
  )

target_include_directories(fcl-linefollower PUBLIC ${PROJECT_SOURCE_DIR})

target_link_libraries(fcl-linefollower
  enkiviewer
  enki
  GL GLU glut
  Qt5::OpenGL
  Qt5::Widgets
  iir
  fcl
  )
