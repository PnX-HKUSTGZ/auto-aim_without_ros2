#ifndef CAMERA_HPP
#define CAMERA_HPP
#include "MvCameraControl.h"
#include <opencv2/opencv.hpp> 
#include<chrono>

class Camera
{
public:
    Camera();

    void PressEnterToExit(void);

    bool PrintDeviceInfo(MV_CC_DEVICE_INFO* pstMVDevInfo);

    static void __stdcall ImageCallBackEx(unsigned char * pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser);

    void videoGet();

    static struct frame
    {
        cv::Mat mat;
        std::chrono::time_point<std::chrono::system_clock> time;
    };


};

#endif // 结束条件编译