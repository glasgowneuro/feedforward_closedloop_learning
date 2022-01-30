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

set(OpenGL_GL_PREFERENCE GLVND)

# required packages
find_package(Qt5 COMPONENTS Widgets Core OpenGL REQUIRED)
find_package(Threads)

# required libraries
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)

include_directories(${GLUT_INCLUDE_DIR})
include_directories(${OpenGL_INCLUDE_DIRS})

add_executable(linefollower
  Linefollower.cpp
  Racer.cpp
  )

target_include_directories(linefollower PUBLIC ${PROJECT_SOURCE_DIR} ..)

target_link_libraries(linefollower
  enkiviewer
  enki
  ${OPENGL_LIBRARIES} ${GLUT_LIBRARY}
  Qt5::OpenGL
  Qt5::Widgets
  iir
  fcl
  )
