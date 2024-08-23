cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(testcameraofficial)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 包含头文件目录
include_directories(${PROJECT_SOURCE_DIR}/include)

find_package(OpenCV REQUIRED)


# 添加可执行文件
add_executable(${PROJECT_NAME} testofficial.cpp)


# 链接MvCameraControl库
target_link_libraries(${PROJECT_NAME} /home/pnx/training_code/testOfficial/64/libMvCameraControl.so)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# 设置rpath以运行时定位动态库
set_target_properties(${PROJECT_NAME} PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "/home/pnx/training_code/testOfficial/64"
)

