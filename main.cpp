#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iomanip>
#include <sstream>
#include"mvs/include/mvs/camera.h"

//定义一个相机线程和图片处理线程间通信的数据结构，包含帧和时间戳
struct FrameData {
    cv::Mat frame;
    std::chrono::time_point<std::chrono::system_clock> timestamp;
};

std::queue<FrameData> frameQueue;  // 用于存储带时间戳的帧
std::mutex mtx;  // 互斥锁，保护共享队列
std::condition_variable asdf;  // 条件变量，用于线程同步

bool capturing = true;  // 捕捉标志位
Camera camera;
std::string error_message;//打开摄像头时的错误信息

// 获取当前时间的时间戳


// 线程1：负责捕捉相机的视频帧
void captureFrames() {
    if (!camera.open_camera(error_message, ""))
    {
        std::cerr << "Error opening camera: " << error_message << std::endl;
        return;
    }
    
    struct image_info img_info;
    std::vector<uint8_t> image_data;

    while (capturing) {
        if (!camera.get_frame(&img_info, image_data, error_message, nullptr))
        {
            std::cerr << "Error getting image: " << error_message << std::endl;
            break;
        }

        

        // 假设 img_info 包含图像的宽度、高度和像素类型信息
        // 将 image_data 转换为 cv::Mat
        
        cv::Mat frame(img_info.height, img_info.width, CV_8UC3, image_data.data());

        // 克隆图像，避免在队列中数据被覆盖
        cv::Mat clonedFrame = frame.clone();
        // 获取当前时间戳
        auto timestamp = std::chrono::system_clock::now();

        // 将捕捉到的帧和时间戳放入队列
        std::unique_lock<std::mutex> lock(mtx);
        frameQueue.push({clonedFrame, timestamp});
        lock.unlock();

        // 通知处理线程
        asdf.notify_one();
    }

    camera.close_camera(error_message);  // 释放摄像头资源
}

// 线程2：负责处理视频帧
void processFrames() {
    while (capturing) {
        std::unique_lock<std::mutex> lock(mtx);
        asdf.wait(lock, [] { return !frameQueue.empty(); });  // 等待捕捉线程的通知

        // 从队列中取出一帧进行处理
        FrameData frameData = frameQueue.front();
        frameQueue.pop();
        lock.unlock();

        
        

        // 显示处理后的帧
        cv::imshow("Processed Frame with Timestamp", frameData.frame);

        // 按下'q'键退出
        if (cv::waitKey(1) == 'q') {
            capturing = false;
            break;
        }
    }
}

int main() {
    // 创建捕捉和处理线程
    std::thread captureThread(captureFrames);
    std::thread processThread(processFrames);

    // 等待线程结束
    captureThread.join();
    processThread.join();

    return 0;
}



