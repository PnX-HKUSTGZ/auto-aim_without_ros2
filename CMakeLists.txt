cmake_minimum_required(VERSION 3.10)

# 工程名称
project(code_without_ros2)

# 设置C++标准为C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 包含头文件目录
include_directories(/home/pnx/training_code/cameraOpen/include)

# 查找OpenCV库
find_package(OpenCV REQUIRED)

# 添加可执行文件
add_executable(main main.cpp)

# 链接OpenCV库到可执行文件
target_link_libraries(main /home/pnx/training_code/cameraOpen/64/libMvCameraControl.so)
target_link_libraries(main ${OpenCV_LIBS})

