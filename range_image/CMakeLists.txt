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

# CMakeLists.txt 示例
# cmake_minimum_required(VERSION 3.10)
# project(MyTensorrtProject)

# # 设置 C++ 标准
# set(CMAKE_CXX_STANDARD 17)

# # 添加 ONNX Runtime 头文件路径
# # include_directories(/usr/local/cuda/include)

# # 添加 ONNX Runtime 库文件路径
# # link_directories(/usr/local/cuda/lib64)
# find_package(CUDA REQUIRED)
# include_directories(${CUDA_INCLUDE_DIRS})

# list(APPEND PLUGINS "nvinfer")
# list(APPEND PLUGINS "nvonnxparser")
# list(APPEND PLUGINS "nvparsers")

# # 添加你的项目源文件
# add_executable(MyTensorrtProject main.cpp)


# # 链接 ONNX Runtime 库
# target_link_libraries(MyTensorrtProject     
#     nvinfer
#     nvonnxparser
#     nvparsers
#     ${CUDA_LIBRARIES}
#     pthread
#     dl
#     rt)
