#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <opencv2/imgproc.hpp>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include"cameraOpen/include/MvCameraControl.h"


//定义一个相机线程和图片处理线程间通信的数据结构，包含帧和时间戳
struct FrameData {
    cv::Mat frame;
    std::chrono::time_point<std::chrono::system_clock> timestamp;
};

std::queue<FrameData> frameQueue;  // 用于存储带时间戳的帧
std::mutex mtx;  // 互斥锁，保护共享队列
std::condition_variable asdf;  // 条件变量，用于线程同步

bool capturing = true;  // 捕捉标志位

std::string error_message;//打开摄像头时的错误信息


//线程1：打开相机通过回调函数获取图像
    void PressEnterToExit(void)
    {
        int c;
        while ( (c = getchar()) != '\n' && c != EOF );
        fprintf( stderr, "\nPress enter to exit.\n");
        while( getchar() != '\n');
    }
    bool PrintDeviceInfo(MV_CC_DEVICE_INFO* pstMVDevInfo)
    {
        if (NULL == pstMVDevInfo)
        {
            printf("The Pointer of pstMVDevInfo is NULL!\n");
            return false;
        }
        if (pstMVDevInfo->nTLayerType == MV_GIGE_DEVICE)
        {
            int nIp1 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24);
            int nIp2 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16);
            int nIp3 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8);
            int nIp4 = (pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff);
            // ch:打印当前相机ip和用户自定义名字 | en:print current ip and user defined name
            printf("Device Model Name: %s\n", pstMVDevInfo->SpecialInfo.stGigEInfo.chModelName);
            printf("CurrentIp: %d.%d.%d.%d\n" , nIp1, nIp2, nIp3, nIp4);
            printf("UserDefinedName: %s\n\n" , pstMVDevInfo->SpecialInfo.stGigEInfo.chUserDefinedName);
        }
        else if (pstMVDevInfo->nTLayerType == MV_USB_DEVICE)
        {
            printf("Device Model Name: %s\n", pstMVDevInfo->SpecialInfo.stUsb3VInfo.chModelName);
            printf("UserDefinedName: %s\n\n", pstMVDevInfo->SpecialInfo.stUsb3VInfo.chUserDefinedName);
        }
        else
        {
            printf("Not support.\n");
        }
        return true;
    }
    static void __stdcall ImageCallBackEx(unsigned char * pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser)
    {
        if (pFrameInfo)
        {
            printf("GetOneFrame, Width[%d], Height[%d], nFrameNum[%d]\n", 
                pFrameInfo->nWidth, pFrameInfo->nHeight, pFrameInfo->nFrameNum);
            
            cv::Mat mat(pFrameInfo->nHeight, pFrameInfo->nWidth, CV_8UC1, pData);
            cv::Mat imageRGB;
            cv::cvtColor(mat, imageRGB, cv::COLOR_BayerRG2RGB);
            
            cv::Mat clonedFrame = imageRGB.clone();
            auto timestamp = std::chrono::system_clock::now();
            
            // 将捕捉到的帧和时间戳放入队列
            std::unique_lock<std::mutex> lock(mtx);
            frameQueue.push({clonedFrame, timestamp});
            lock.unlock();

            // 通知处理线程
            asdf.notify_one();
            
            
        }
    }
    void videoGet()
    {
       
        int nRet = MV_OK;
        void* handle = NULL;
        do 
        {
            // ch:初始化SDK | en:Initialize SDK
            nRet = MV_CC_Initialize();
            if (MV_OK != nRet)
            {
                printf("Initialize SDK fail! nRet [0x%x]\n", nRet);
                break;
            }
            MV_CC_DEVICE_INFO_LIST stDeviceList;
            memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
            // 枚举设备
            // enum device
            nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &stDeviceList);
            if (MV_OK != nRet)
            {
                printf("MV_CC_EnumDevices fail! nRet [%x]\n", nRet);
                break;
            }
            if (stDeviceList.nDeviceNum > 0)
            {
                for (int i = 0; i < stDeviceList.nDeviceNum; i++)
                {
                    printf("[device %d]:\n", i);
                    MV_CC_DEVICE_INFO* pDeviceInfo = stDeviceList.pDeviceInfo[i];
                    if (NULL == pDeviceInfo)
                    {
                        break;
                    } 
                    PrintDeviceInfo(pDeviceInfo);            
                }  
            } 
            else
            {
                printf("Find No Devices!\n");
                break;
            }
            printf("Please Intput camera index: ");
            unsigned int nIndex = 0;
            
            if (nIndex >= stDeviceList.nDeviceNum)
            {
                printf("Intput error!\n");
                break;
            }
            // 选择设备并创建句柄
            // select device and create handle
            nRet = MV_CC_CreateHandle(&handle, stDeviceList.pDeviceInfo[nIndex]);
            if (MV_OK != nRet)
            {
                printf("MV_CC_CreateHandle fail! nRet [%x]\n", nRet);
                break;
            }
            // 打开设备
            // open device
            nRet = MV_CC_OpenDevice(handle);
            if (MV_OK != nRet)
            {
                printf("MV_CC_OpenDevice fail! nRet [%x]\n", nRet);
                break;
            }
            
            // ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
            if (stDeviceList.pDeviceInfo[nIndex]->nTLayerType == MV_GIGE_DEVICE)
            {
                int nPacketSize = MV_CC_GetOptimalPacketSize(handle);
                if (nPacketSize > 0)
                {
                    nRet = MV_CC_SetIntValue(handle,"GevSCPSPacketSize",nPacketSize);
                    if(nRet != MV_OK)
                    {
                        printf("Warning: Set Packet Size fail nRet [0x%x]!\n", nRet);
                    }
                }
                else
                {
                    printf("Warning: Get Packet Size fail nRet [0x%x]!\n", nPacketSize);
                }
            }
            
            // 设置触发模式为off
            // set trigger mode as off
            nRet = MV_CC_SetEnumValue(handle, "TriggerMode", 0);
            if (MV_OK != nRet)
            {
                printf("MV_CC_SetTriggerMode fail! nRet [%x]\n", nRet);
                break;
            }
            // 注册抓图回调
            // register image callback
            nRet = MV_CC_RegisterImageCallBackEx(handle, ImageCallBackEx, handle);
            if (MV_OK != nRet)
            {
                printf("MV_CC_RegisterImageCallBackEx fail! nRet [%x]\n", nRet);
                break; 
            }
            // 开始取流
            // start grab image
            nRet = MV_CC_StartGrabbing(handle);
            if (MV_OK != nRet)
            {
                printf("MV_CC_StartGrabbing fail! nRet [%x]\n", nRet);
                break;
            }
            PressEnterToExit();
            // 停止取流
            // end grab image
            nRet = MV_CC_StopGrabbing(handle);
            if (MV_OK != nRet)
            {
                printf("MV_CC_StopGrabbing fail! nRet [%x]\n", nRet);
                break;
            }
            // 关闭设备
            // close device
            nRet = MV_CC_CloseDevice(handle);
            if (MV_OK != nRet)
            {
                printf("MV_CC_CloseDevice fail! nRet [%x]\n", nRet);
                break;
            }
            // 销毁句柄
            // destroy handle
            nRet = MV_CC_DestroyHandle(handle);
            if (MV_OK != nRet)
            {
                printf("MV_CC_DestroyHandle fail! nRet [%x]\n", nRet);
                break;
            }
            handle = NULL;
        } while (0);
        if (handle != NULL)
        {
            MV_CC_DestroyHandle(handle);
            handle = NULL;
        }
        // ch:反初始化SDK | en:Finalize SDK
        MV_CC_Finalize();
        printf("exit\n");
        
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
    std::thread captureThread(videoGet);
    std::thread processThread(processFrames);

    // 等待线程结束
    captureThread.join();
    processThread.join();

    return 0;
}



