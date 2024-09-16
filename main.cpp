
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

//Get config value
#include"yaml-cpp/yaml.h"

//camera open
#include"cameraOpen/include/MvCameraControl.h"

//camera detection
#include "detector/include/armor.hpp"
#include "detector/include/number_classifier.hpp"
#include "detector/include/detector.hpp"
#include "detector/include/pnp_solver.hpp"
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>

//serialdriver
#include"rm_serial_driver/include/rm_serial_driver.hpp"
#include "rm_serial_driver/include/packet.hpp"
#include "rm_serial_driver/include/crc.hpp"


//定义一个相机线程和图片处理线程间通信的数据结构，包含帧和时间戳
struct FrameData {
    cv::Mat frame;
    std::chrono::time_point<std::chrono::system_clock> timestamp;
};

//串口到tracker的数据，云台与世界系之间的坐标变换
struct transform{
    std::chrono::time_point<std::chrono::system_clock> timestamp;
    Eigen::Quaternionf q;
    bool reset_tracker;
  };
std::queue<FrameData> frameQueue;  // 用于存储带时间戳的帧
std::queue<transform> transformQueue;//用于存储云台与世界系之间的坐标变换
std::mutex mtx;  // 互斥锁，保护共享队列（用于线程1和线程2）
std::mutex mtx2;//互斥锁，保护共享队列（用于线程3和线程2）
std::condition_variable asdf;  // 条件变量，用于线程同步
std::condition_variable asdf2;  // 条件变量，用于线程同步
const size_t MAX_QUEUE_SIZE = 1;//限制队列长度
bool capturing = true;  // 捕捉标志位

std::string error_message;//打开摄像头时的错误信息
rm_serial_driver::RMSerialDriver serialdriver;//创建一个串口驱动器


//初始化detector，应用于线程2中
std::unique_ptr<rm_auto_aim::Detector> initDetector()
{
    //设置二值化参数和探测颜色
    YAML::Node config = YAML::LoadFile("/home/pnx/training_code/config.yaml");
    std::cout << "config.yaml loaded" << std::endl;
    int binary_thres = config["detector"]["binary_thres"].as<int>();
    int detect_color = config["detector"]["detect_color"].as<int>();
    
    //填充light和armor类所需要的参数
    rm_auto_aim::Detector::LightParams l_params = {
    
    .min_ratio =  config["detector"]["light_min_ratio"].as<double>(),
    .max_ratio =  0.4,
    .max_angle = 40.0};
  
    rm_auto_aim::Detector::ArmorParams a_params = {
    .min_light_ratio = config["detector"]["armor_min_light_ratio"].as<double>(),
    .min_small_center_distance = 0.8,
    .max_small_center_distance = 3.2,
    .min_large_center_distance = 3.2,
    .max_large_center_distance = 5.5,
    .max_angle = 35.0};
    
    //初始化detector,创建一个跟踪器
    auto detector = std::make_unique<rm_auto_aim::Detector>(binary_thres, detect_color, l_params, a_params);

    //初始化数字分类器
    //设置模型路径和阈值
    auto pkg_path = config["detector"]["pkg_path"].as<std::string>();
    auto model_path = pkg_path + "/model/mlp.onnx";
    auto label_path = pkg_path + "/model/label.txt";
    double threshold = config["detector"]["classifier_threshold"].as<double>();
    std::vector<std::string> ignore_classes = 
    config["detector"]["ignore_classes"].as<std::vector<std::string>>();
    detector->classifier =
    std::make_unique<rm_auto_aim::NumberClassifier>(model_path, label_path, threshold, ignore_classes);

    return detector;
}


//线程1：打开相机通过回调函数获取图像,下面四个函数为应用到的函数，看看imagecallbackex函数即可，其他三个直接copy的官方文档
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
            asdf.wait(lock, []{ return frameQueue.size() < MAX_QUEUE_SIZE; });
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


int n = 0;
// 线程2：负责处理视频帧
void processFrames() {
    std::unique_ptr<rm_auto_aim::Detector>detector_ = initDetector();
    //获取相机内参和畸变系数
    YAML::Node config = YAML::LoadFile("/home/pnx/training_code/config.yaml");
    std::array<double,9> camera_matrix_data;
    int i = 0;
    for (const auto& row : config["detector"]["camera_matrix"]) {
        for (const auto& element : row) {
            camera_matrix_data[i] = element.as<double>();
            i++;
        }
    }
    
    std::vector<double> dist_coeffs_data;
    for (const auto& element : config["detector"]["dist_coeff"]) {
        dist_coeffs_data.push_back(element.as<double>());
        }
    //获取完相机参数后初始化位姿估计器
    std::unique_ptr<rm_auto_aim::PnPSolver> pnp_solver_ = 
    std::make_unique<rm_auto_aim::PnPSolver>(camera_matrix_data, dist_coeffs_data);

    std::vector<rm_auto_aim::Detector::Armormsg>armors_msg;//与tracker模块（目标跟踪与状态估计）交互的数据结构，创建在循环外部，防止反复析构增大开销
    rm_auto_aim::Detector::Armormsg armor_msg;

    while (capturing) {
    {
        std::unique_lock<std::mutex> lock(mtx);
        asdf.wait(lock, [] { return !frameQueue.empty(); });  // 等待捕捉线程的通知

        // 从队列中取出一帧进行处理
        FrameData frameData = frameQueue.front();
        frameQueue.pop();
        lock.unlock();
        asdf.notify_one();
        auto transmit_time = std::chrono::system_clock::now();
        auto transmit_latency = std::chrono::duration_cast<std::chrono::milliseconds>(transmit_time - frameData.timestamp).count();
        std::cout<<"transmit latency:"<<transmit_latency<<std::endl;

        //直接显示图像
        cv::imshow("originFrame", frameData.frame);
        
        auto startprocesstime = std::chrono::system_clock::now();

        auto armors = detector_->detect(frameData.frame);
        //计算延迟
        auto final_time = std::chrono::system_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(final_time - frameData.timestamp).count();
        
        auto endprocesstime = std::chrono::system_clock::now();
        auto process_time = std::chrono::duration_cast<std::chrono::milliseconds>(endprocesstime - startprocesstime).count();
       
        std::cout<<"process time:"<<process_time<<std::endl;
        // 显示数字
        if (!armors.empty()) {
        auto all_num_img = detector_->getAllNumbersImage();
        cv::imshow("All Numbers", all_num_img);
        }

        //把装甲板画出来
        detector_->drawResults(frameData.frame);
        
        // Draw latency
        std::stringstream latency_ss;
        latency_ss << "Latency: " << std::fixed << std::setprecision(2) << latency << "ms";
        auto latency_s = latency_ss.str();
        cv::putText(
        frameData.frame, latency_s, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
         
        cv::imshow("Result", frameData.frame);
        //目标检测结束，进入位姿估计
        

        armors_msg.clear();//把上一次的数据清空
        
        if (pnp_solver_ != nullptr) {
        
        for (const auto & armor : armors) {
        cv::Mat rvec, tvec;
        bool success = pnp_solver_->solvePnP(armor, rvec, tvec);
        if (success) {
            
            // Fill basic info
            armor_msg.timestamp = frameData.timestamp;
            armor_msg.type = rm_auto_aim::ARMOR_TYPE_STR[static_cast<int>(armor.type)];
            armor_msg.number = armor.number;

            // Fill pose
            armor_msg.pose.position.x = tvec.at<double>(0);
            armor_msg.pose.position.y = tvec.at<double>(1);
            armor_msg.pose.position.z = tvec.at<double>(2);
            // rvec to 3x3 rotation matrix
            cv::Mat rotation_matrix;
            cv::Rodrigues(rvec, rotation_matrix);//将旋转向量转换为旋转矩阵
            // rotation matrix to quaternion
            Eigen::Matrix<double, 3, 3> rotationMatrix;
            rotationMatrix <<
            rotation_matrix.at<double>(0, 0), rotation_matrix.at<double>(0, 1), rotation_matrix.at<double>(0, 2),
            rotation_matrix.at<double>(1, 0), rotation_matrix.at<double>(1, 1), rotation_matrix.at<double>(1, 2),
            rotation_matrix.at<double>(2, 0), rotation_matrix.at<double>(2, 1), rotation_matrix.at<double>(2, 2);
            Eigen::Quaterniond quaternion(rotationMatrix);

            armor_msg.pose.orientation = quaternion;
            
            
            
            // Fill the distance to image center
            armor_msg.distance_to_image_center = pnp_solver_->calculateDistanceToCenter(armor.center);
            
            
            armors_msg.push_back(armor_msg);
        } 
        else 
        {
            std::cout<<"solvePnP failed"<<std::endl;
        }
        }
    }
        else{
            std::cout<<"pnp_solver_ is nullptr"<<std::endl;}
    
    }
    {
    //位姿估计结束，进入目标追踪与状态估计（tracker模块）


    //从串口获取云台位姿（线程间通信）
    std::unique_lock<std::mutex> lock(mtx2);
    asdf2.wait(lock, [] { return !transformQueue.empty(); });  // 等待捕捉线程的通知
    
    // 从队列中取出一帧进行处理
    transform t = transformQueue.front();
    transformQueue.pop();
    lock.unlock();
    asdf2.notify_one();

    //输出四元数的四个值
    /* std::cout<<"Received transform data at "<<std::chrono::duration_cast<std::chrono::milliseconds>(t.timestamp.time_since_epoch()).count()<<"ms"<<std::endl;
    std::cout<<"q:"<<t.q.x()<<" "<<t.q.y()<<" "<<t.q.z()<<" "<<t.q.w()<<std::endl; */
    
    auto serial_transmit_time = std::chrono::system_clock::now();
    auto serial_transmit_latency = std::chrono::duration_cast<std::chrono::milliseconds>(serial_transmit_time - t.timestamp).count();
    std::cout<<"serial transmit latency:"<<serial_transmit_latency<<std::endl;

    if(armors_msg.empty())
    {
        std::cout<<"No armor detected"<<std::endl;
    }
    else
    {
        std::cout<<"Detected armors:"<<std::endl;
    }
  
    


    
    }   

        
        

        // 按下'q'键退出
        if (cv::waitKey(1) == 'q') {
            capturing = false;
            break;
        }
    auto end_time = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - frameData.timestamp).count();
    std::cout<<"Total duration:"<<duration<<std::endl;
    }
    
    
}

// 线程3：接收串口数据
void receiveData()
{
    std::vector<uint8_t> header(1);
  std::vector<uint8_t> data;
  data.reserve(sizeof(rm_serial_driver::ReceivePacket));

  while (true) {
    try {
      serialdriver.serial_driver_->port()->receive(header);

      if (header[0] == 0x5A) {
        data.resize(sizeof(rm_serial_driver::ReceivePacket) - 1);
        serialdriver.serial_driver_->port()->receive(data);

        data.insert(data.begin(), header[0]);
        rm_serial_driver::ReceivePacket packet = rm_serial_driver::fromVector(data);
        //crc校验
        bool crc_ok =
          crc16::Verify_CRC16_Check_Sum(reinterpret_cast<const uint8_t *>(&packet), sizeof(packet));
        if (crc_ok) {
          if (!serialdriver.initial_set_param_ || packet.detect_color != serialdriver.previous_receive_color_) {
            YAML::Node config = YAML::LoadFile("/home/pnx/training_code/config.yaml");
            config["detector"]["detect_color"] = packet.detect_color;//对颜色参数进行更改
            serialdriver.previous_receive_color_ = packet.detect_color;
          }

          // 发送世界系到云台系的变换数据到tracker
          transform t;
          YAML::Node config = YAML::LoadFile("/home/pnx/training_code/config.yaml");
          serialdriver.timestamp_offset_ = config["serialdriver"]["timestamp_offset"].as<double>();
          t.timestamp = std::chrono::system_clock::now() + 
              std::chrono::duration_cast<std::chrono::system_clock::duration>(std::chrono::duration<double>(serialdriver.timestamp_offset_));
          
          //给t.q四元数赋值
          t.q = Eigen::Quaternionf(packet.q[0], packet.q[1], packet.q[2], packet.q[3]);
          //给t.reset_tracker赋值,是否要重置追踪器
          t.reset_tracker = packet.reset_tracker;

          //放入队列
          std::unique_lock<std::mutex> lock(mtx2);
          asdf2.wait(lock, []{ return transformQueue.size() < MAX_QUEUE_SIZE; });
          transformQueue.push(t);
          // 通知处理线程
          asdf2.notify_one();

          
        } else {
          std::cerr<<"CRC error!";
        }
      } else {
        std::cerr<<"Header error!";
      }
    } catch (const std::exception & ex) {
      std::cerr<< "Error while receiving data: %s", ex.what();
      serialdriver.reopenPort();
    }
  }
}



int main() {
    // 创建捕捉和处理线程
    std::thread captureThread(videoGet);
    std::thread receiveThread(receiveData);
    //std::this_thread::sleep_for(std::chrono::milliseconds(1));
    std::thread processThread(processFrames);

    // 等待线程结束
    captureThread.join();
    processThread.join();
    receiveThread.join();

    return 0;
}



