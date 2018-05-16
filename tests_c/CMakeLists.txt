cmake_minimum_required(VERSION 3.5)
set (CMAKE_CXX_STANDARD 11)
include(GNUInstallDirs)

add_executable (test_bandpass test_bandpass.cpp)
target_link_libraries(test_bandpass deep_feedback_learning_static)
target_include_directories(test_bandpass PRIVATE .. ../dfl)

add_executable (test_deep_feedback_learning test_deep_feedback_learning.cpp)
target_link_libraries(test_deep_feedback_learning deep_feedback_learning_static)
target_include_directories(test_deep_feedback_learning PRIVATE .. ../dfl)

add_executable (test_layer test_layer.cpp)
target_link_libraries(test_layer deep_feedback_learning_static)
target_include_directories(test_layer PRIVATE .. ../dfl)

add_executable (test_neuron test_neuron.cpp)
target_link_libraries(test_neuron deep_feedback_learning_static)
target_include_directories(test_neuron PRIVATE .. ../dfl)
