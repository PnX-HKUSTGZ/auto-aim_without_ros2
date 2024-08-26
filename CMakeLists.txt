cmake_minimum_required(VERSION 3.10)

# 工程名称
project(code_without_ros2)

# 设置C++标准为C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB_RECURSE DETECTOR_FILES ${PROJECT_SOURCE_DIR}/detector/src/*.cpp)

# 包含头文件目录
include_directories(/home/pnx/training_code/cameraOpen/include)

# 查找OpenCV库和yaml-cpp库
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# 添加可执行文件
add_executable(main main.cpp ${DETECTOR_FILES})

# 链接所需的库
target_link_libraries(main
    
    ${OpenCV_LIBS}
    yaml-cpp
    Eigen3::Eigen
    /home/pnx/training_code/cameraOpen/64/libMvCameraControl.so
)

