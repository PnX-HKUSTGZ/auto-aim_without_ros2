cmake_minimum_required(VERSION 3.10)

# 工程名称
project(code_without_ros2)

# 设置C++标准为C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "Debug")

file(GLOB_RECURSE DETECTOR_FILES 
    ${PROJECT_SOURCE_DIR}/detector/src/*.cpp
    ${PROJECT_SOURCE_DIR}/rm_serial_driver/src/*.cpp
    ${PROJECT_SOURCE_DIR}/tracker/src/*.cpp
    )

# 包含头文件目录
include_directories(${PROJECT_SOURCE_DIR}/cameraOpen/include)
include_directories(${PROJECT_SOURCE_DIR}/detector/include)
include_directories(${PROJECT_SOURCE_DIR}/rm_serial_driver/include)
include_directories(${PROJECT_SOURCE_DIR}/tracker/include)

# 查找OpenCV库和yaml-cpp库
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 3.3 REQUIRED )

# 添加可执行文件
add_executable(main main.cpp ${DETECTOR_FILES})

# 链接所需的库
target_link_libraries(main
    
    ${OpenCV_LIBS}
    yaml-cpp
    Eigen3::Eigen
    ${PROJECT_SOURCE_DIR}/cameraOpen/64/libMvCameraControl.so
    ${PROJECT_SOURCE_DIR}/rm_serial_driver/lib/libserial_driver.so
    ${PROJECT_SOURCE_DIR}/rm_serial_driver/lib/libio_context.so
)

