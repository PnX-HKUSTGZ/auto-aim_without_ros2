#include "include/Camera.hpp"
#include <iostream>

//静态成员变量初始化
std::mutex Camera::mtx;
std::condition_variable Camera::asdf;
std::queue<FrameData> Camera::frameQueue;

Camera::Camera() : handle(nullptr) {}

Camera::~Camera() {
    if (handle != nullptr) {
        MV_CC_DestroyHandle(handle);
    }
    MV_CC_Finalize();
}

void Camera::startCapture() {
    int nRet = MV_OK;
    do {
        nRet = MV_CC_Initialize();
        if (MV_OK != nRet) {
            std::cerr << "Initialize SDK fail! nRet [0x" << std::hex << nRet << "]\n";
            break;
        }

        MV_CC_DEVICE_INFO_LIST stDeviceList;
        memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
        nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &stDeviceList);
        if (MV_OK != nRet) {
            std::cerr << "MV_CC_EnumDevices fail! nRet [0x" << std::hex << nRet << "]\n";
            break;
        }

        if (stDeviceList.nDeviceNum > 0) {
            unsigned int nIndex = 0;
            nRet = MV_CC_CreateHandle(&handle, stDeviceList.pDeviceInfo[nIndex]);
            if (MV_OK != nRet) {
                std::cerr << "MV_CC_CreateHandle fail! nRet [0x" << std::hex << nRet << "]\n";
                break;
            }

            nRet = MV_CC_OpenDevice(handle);
            if (MV_OK != nRet) {
                std::cerr << "MV_CC_OpenDevice fail! nRet [0x" << std::hex << nRet << "]\n";
                break;
            }

            nRet = MV_CC_SetEnumValue(handle, "TriggerMode", 0);
            if (MV_OK != nRet) {
                std::cerr << "MV_CC_SetTriggerMode fail! nRet [0x" << std::hex << nRet << "]\n";
                break;
            }

            nRet = MV_CC_RegisterImageCallBackEx(handle, ImageCallBackEx, handle);
            if (MV_OK != nRet) {
                std::cerr << "MV_CC_RegisterImageCallBackEx fail! nRet [0x" << std::hex << nRet << "]\n";
                break;
            }

            nRet = MV_CC_StartGrabbing(handle);
            if (MV_OK != nRet) {
                std::cerr << "MV_CC_StartGrabbing fail! nRet [0x" << std::hex << nRet << "]\n";
                break;
            }
        } else {
            std::cerr << "Find No Devices!\n";
            break;
        }
    } while (0);

    if (handle != nullptr) {
        MV_CC_DestroyHandle(handle);
        handle = nullptr;
    }
}

void Camera::stopCapture() {
    if (handle != nullptr) {
        MV_CC_StopGrabbing(handle);
        MV_CC_CloseDevice(handle);
        MV_CC_DestroyHandle(handle);
        handle = nullptr;
    }
    MV_CC_Finalize();
}

//利用回调函数将图像放入队列，传输给处理线程
void __stdcall Camera::ImageCallBackEx(unsigned char * pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser) {
    if (pFrameInfo) {
        cv::Mat mat(pFrameInfo->nHeight, pFrameInfo->nWidth, CV_8UC1, pData);
        cv::Mat imageRGB;
        cv::cvtColor(mat, imageRGB, cv::COLOR_BayerRG2RGB);

        std::shared_ptr<cv::Mat> imgPtr = std::make_shared<cv::Mat>(imageRGB);
        auto timestamp = std::chrono::system_clock::now();

        std::unique_lock<std::mutex> lock(mtx);
        asdf.wait(lock, []{ return frameQueue.size() < MAX_QUEUE_SIZE; });
        frameQueue.push({imgPtr, timestamp});
        lock.unlock();

        asdf.notify_one();
    }
}