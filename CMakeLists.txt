cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(range_image_creation)

set(CMAKE_CXX_STANDARD 17)


find_package(PCL REQUIRED)
find_package( OpenCV REQUIRED )
MESSAGE(STATUS "OpenCV version: "${OpenCV_VERSION})
find_package(manif REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )

# 使用 file(GLOB_RECURSE ...) 查找 .cpp 文件
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/range_image)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tensorrt)

# find_package(Torch REQUIRED)

# file(GLOB_RECURSE RANGE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/range_image/*.cpp")
# add_library(range_image
#     ${RANGE_SOURCES}
# )
# link_libraries(range_image
#     ${PCL_LIBRARIES}  
#     ${OpenCV_LIBS}
#     ${PCL_LIBRARY_DIRS}
#     ${TORCH_LIBRARIES}
# )


######################### TensorRT
# # 添加 ONNX Runtime 头文件路径
include_directories(/usr/local/cuda/include)
# 添加 ONNX Runtime 库文件路径
link_directories(/usr/local/cuda/lib64)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
list(APPEND PLUGINS "nvinfer")
list(APPEND PLUGINS "nvonnxparser")
list(APPEND PLUGINS "nvparsers")


file(GLOB_RECURSE TENSORRT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/range_image/*.cpp")
add_library(tensorrt SHARED
    ${TENSORRT_SOURCES}
)
link_libraries(tensorrt
    ${PCL_LIBRARIES}  
    ${OpenCV_LIBS}
    ${PCL_LIBRARY_DIRS}
    ${TORCH_LIBRARIES}
    nvinfer
    nvonnxparser
    nvparsers
    ${CUDA_LIBRARIES}
    pthread
    dl
    rt
)


## app
add_executable (range_image_creation app/main.cpp)
target_link_libraries (range_image_creation 
    ${PCL_LIBRARIES}  
    ${OpenCV_LIBS}
    ${PCL_LIBRARY_DIRS})

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
