cmake_minimum_required(VERSION 3.5)
set (CMAKE_CXX_STANDARD 11)
include(GNUInstallDirs)

add_executable (test_bandpass test_bandpass.cpp)
target_link_libraries(test_bandpass fcl_static ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(test_bandpass PRIVATE .. ../fcl)

add_executable (test_fcl test_fcl.cpp)
target_link_libraries(test_fcl fcl_static ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(test_fcl PRIVATE .. ../fcl)

add_executable (test_layer test_layer.cpp)
target_link_libraries(test_layer fcl_static ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(test_layer PRIVATE .. ../fcl)

add_executable (test_neuron test_neuron.cpp)
target_link_libraries(test_neuron fcl_static ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(test_neuron PRIVATE .. ../fcl)

add_executable (test_fcl_filters test_fcl_filters.cpp)
target_link_libraries(test_fcl_filters fcl_static ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(test_fcl_filters PRIVATE .. ../fcl)
