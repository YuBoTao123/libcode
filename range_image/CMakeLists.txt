cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(range_image_creation)

find_package(PCL REQUIRED)
find_package( OpenCV REQUIRED )
MESSAGE(STATUS "OpenCV version: "${OpenCV_VERSION})
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_executable (range_image_creation main.cpp)
target_link_libraries (range_image_creation ${PCL_LIBRARIES}  ${OpenCV_LIBS})