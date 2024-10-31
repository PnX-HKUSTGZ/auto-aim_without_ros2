#ifndef FRAME_PROCESSOR_HPP
#define FRAME_PROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include <chrono>
#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>
#include "detector/include/armor.hpp"
#include "detector/include/number_classifier.hpp"
#include "detector/include/detector.hpp"
#include "detector/include/pnp_solver.hpp"
#include "tracker/include/tracker_node.hpp"
#include "ballistic_calculation/inlude/ballistic_calculation.hpp"
#include "../cameraOpen/include/Camera.hpp"

#include"../SerialReceiver/SerialReceiver.hpp"

class FrameProcessor {
public:
    FrameProcessor();
    void processFrames();
    std::unique_ptr<rm_auto_aim::Detector> initDetector();


private:
    
    std::unique_ptr<rm_auto_aim::Detector> detector_;
    std::unique_ptr<rm_auto_aim::PnPSolver> pnp_solver_;
    std::unique_ptr<rm_auto_aim::Ballistic> calculator;
    rm_auto_aim::ArmorTracker tracker;
    bool capturing = true;

    rm_auto_aim::Detector::Armormsg armor_msg;
    std::vector<rm_auto_aim::Detector::Armormsg>armors_msg;//与tracker模块（目标跟踪与状态估计）交互的数据结构
    rm_auto_aim::ArmorTracker::Target target_msg;//tracker模块得到的目标信息,提供给弹道解算模块
};

#endif // FRAME_PROCESSOR_HPP