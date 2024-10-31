#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <opencv2/opencv.hpp>
#include <chrono>
#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>
#include "MvCameraControl.h"

struct FrameData {
    std::shared_ptr<cv::Mat> frame;
    std::chrono::time_point<std::chrono::system_clock> timestamp;
};

class Camera {
public:
    Camera();
    ~Camera();
    void startCapture();
    void stopCapture();
    static void __stdcall ImageCallBackEx(unsigned char * pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser);
    static std::mutex mtx;
    static std::condition_variable asdf;
    static std::queue<FrameData> frameQueue;

private:
    void* handle;
    static const size_t MAX_QUEUE_SIZE = 1;
    
};

#endif // CAMERA_HPP