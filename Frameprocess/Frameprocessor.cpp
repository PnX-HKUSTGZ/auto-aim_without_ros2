#include "FrameProcessor.hpp"
#include <iostream>
#include <yaml-cpp/yaml.h>


FrameProcessor::FrameProcessor() : capturing(true) {
    detector_ = initDetector();
    YAML::Node config = YAML::LoadFile("/home/pnx/training_code/config.yaml");
    std::array<double, 9> camera_matrix_data;
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

    pnp_solver_ = std::make_unique<rm_auto_aim::PnPSolver>(camera_matrix_data, dist_coeffs_data);
    calculator_ = std::make_unique<rm_auto_aim::Ballistic>();
}

std::unique_ptr<rm_auto_aim::Detector> FrameProcessor::initDetector() {
    YAML::Node config = YAML::LoadFile("/home/pnx/training_code/config.yaml");
    int binary_thres = config["detector"]["binary_thres"].as<int>();
    int detect_color = config["detector"]["detect_color"].as<int>();

    rm_auto_aim::Detector::LightParams l_params = {
        .min_ratio = config["detector"]["light_min_ratio"].as<double>(),
        .max_ratio = 0.4,
        .max_angle = 40.0
    };

    rm_auto_aim::Detector::ArmorParams a_params = {
        .min_light_ratio = config["detector"]["armor_min_light_ratio"].as<double>(),
        .min_small_center_distance = 0.8,
        .max_small_center_distance = 3.2,
        .min_large_center_distance = 3.2,
        .max_large_center_distance = 5.5,
        .max_angle = 35.0
    };

    auto detector = std::make_unique<rm_auto_aim::Detector>(binary_thres, detect_color, l_params, a_params);

    auto pkg_path = config["detector"]["pkg_path"].as<std::string>();
    auto model_path = pkg_path + "/model/mlp.onnx";
    auto label_path = pkg_path + "/model/label.txt";
    double threshold = config["detector"]["classifier_threshold"].as<double>();
    std::vector<std::string> ignore_classes = config["detector"]["ignore_classes"].as<std::vector<std::string>>();
    detector->classifier = std::make_unique<rm_auto_aim::NumberClassifier>(model_path, label_path, threshold, ignore_classes);

    return detector;
}

void FrameProcessor::processFrames() {
    while (capturing) {
    {
        std::unique_lock<std::mutex> lock(mtx);
        asdf.wait(lock, [this] { return !frameQueue.empty(); });  // 等待捕捉线程的通知

        // 从队列中取出一帧进行处理
        FrameData frameData = frameQueue.front();
        frameQueue.pop();
        lock.unlock();
        asdf.notify_one();
        auto transmit_time = std::chrono::system_clock::now();
        auto transmit_latency = std::chrono::duration_cast<std::chrono::milliseconds>(transmit_time - frameData.timestamp).count();
        std::cout<<"transmit latency:"<<transmit_latency<<std::endl;

        
        cv::imshow("originFrame", *frameData.frame);

     
        auto startprocesstime = std::chrono::system_clock::now();

        auto armors = detector_->detect(*frameData.frame);
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
        detector_->drawResults(*frameData.frame);
        
        // Draw latency
        std::stringstream latency_ss;
        latency_ss << "Latency: " << std::fixed << std::setprecision(2) << latency << "ms";
        auto latency_s = latency_ss.str();
        cv::putText(
        *frameData.frame, latency_s, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
         
        cv::imshow("Result", *frameData.frame);
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
    
    //位姿估计结束，进入目标追踪与状态估计（tracker模块）

    {
    //从串口获取云台位姿（线程间通信）
    std::unique_lock<std::mutex> lock(mtx2);
    asdf2.wait(lock, [this] { return !transformQueue.empty(); });  // 等待捕捉线程的通知
    
    // 从队列中取出一帧进行处理
    transform t = transformQueue.front();
    transformQueue.pop();
    lock.unlock();
    asdf2.notify_one();
    

    if(!armors_msg.empty()){
        // 计算逆四元数
        Eigen::Quaterniond q = t.q.cast<double>();//浮点数转成双精度

        
        //把相机系转化到世界系
        for (auto & armor : armors_msg) {
            rm_auto_aim::Detector::Pose ps;
            ps = armor.pose;
            //将地方装甲板从相机系转至云台系
            ps.position.x = tracker.gimbal2camra[0] + armor.pose.position.x;
            ps.position.y = tracker.gimbal2camra[1] + armor.pose.position.y;
            ps.position.z = tracker.gimbal2camra[2] + armor.pose.position.z;
            Eigen::Vector3d v(ps.position.x + tracker.gimbal2camra[0], tracker.gimbal2camra[1] + ps.position.y, tracker.gimbal2camra[2] + ps.position.z);
            //从云台系转至世界系
            Eigen::Vector3d v_rotated = q * v;
            armor.pose.position.x = v_rotated[0];
            armor.pose.position.y = v_rotated[1];
            armor.pose.position.z = v_rotated[2];

            armor.pose.orientation = q * armor.pose.orientation;}

        // Filter abnormal armors
        armors_msg.erase(
        std::remove_if(
        armors_msg.begin(), armors_msg.end(),
        [&tracker](const rm_auto_aim::Detector::Armormsg & armor) {
            return abs(armor.pose.position.z) > 1.2 ||
                Eigen::Vector2d(armor.pose.position.x, armor.pose.position.y).norm() >
                    tracker.max_armor_distance_;
        }),
        armors_msg.end());

        std::chrono::time_point<std::chrono::system_clock> time = armors_msg[0].timestamp;
        
        // Update tracker
        if (tracker.tracker_->tracker_state == rm_auto_aim::Tracker::LOST) {
        tracker.tracker_->init(armors_msg);
        target_msg.tracking = false;
        }  
        else {
            tracker.dt_ = std::chrono::duration_cast<std::chrono::duration<double>>(time - tracker.last_time_).count();

            tracker.tracker_->lost_thres = static_cast<int>(tracker.lost_time_thres_ / tracker.dt_);
            tracker.tracker_->update(armors_msg);

        

            if (tracker.tracker_->tracker_state == rm_auto_aim::Tracker::DETECTING) 
            {
                target_msg.tracking = false;
                std::cout<<"tracking false"<<std::endl;
            } 
            else if (
            tracker.tracker_->tracker_state == rm_auto_aim::Tracker::TRACKING ||
            tracker.tracker_->tracker_state == rm_auto_aim::Tracker::TEMP_LOST){
                target_msg.tracking = true;
                // Fill target message
                const auto & state = tracker.tracker_->target_state;
                target_msg.id = tracker.tracker_->tracked_id;
                target_msg.armors_num = static_cast<int>(tracker.tracker_->tracked_armors_num);
                target_msg.position.x = state(0);
                target_msg.velocity.x = state(1);
                target_msg.position.y = state(2);
                target_msg.velocity.y = state(3);
                target_msg.position.z = state(4);
                target_msg.velocity.z = state(5);
                target_msg.yaw = state(6);
                target_msg.v_yaw = state(7);
                target_msg.radius_1 = state(8);
                target_msg.radius_2 = tracker.tracker_->another_r;
                target_msg.dz = tracker.tracker_->dz;
            }

            //弹道解算
            calculator->target_msg = target_msg;
    
            //进入第一次大迭代
            double init_pitch = std::atan(target_msg.position.z / std::sqrt(target_msg.position.x * target_msg.position.x + target_msg.position.y * target_msg.position.y));
            double init_t = std::sqrt(target_msg.position.x * target_msg.position.x + target_msg.position.y * target_msg.position.y) / (cos(init_pitch) * calculator->bulletV);
            
            
            std::pair<double,double> first_iteration_result = calculator->iteration1(calculator->THRES1 , init_pitch , init_t);
            
            //预测并选择合适击打的装甲板
            double temp_theta = first_iteration_result.first;
            double temp_t = first_iteration_result.second;

            //预测平衡步兵的最佳装甲板
            double chosen_yaw;
            double z;
            double r;

            if(target_msg.armors_num == 2){
            std::vector<double>hit_aim = calculator->predictBalanceBestArmor(temp_t);
                
            chosen_yaw = hit_aim[0];
            z = hit_aim[1];
            r = hit_aim[2];
            }
            
            //else 
            if (target_msg.armors_num == 4){
            std::vector<double>hit_aim = calculator->predictInfantryBestArmor(temp_t);
                
            chosen_yaw = hit_aim[0];
            z = hit_aim[1];
            r = hit_aim[2];
            }
            else{
                std::cerr<<"Error: armors_num is not 2 or 4"<<std::endl;
            
            }
            

            //进入第二次大迭代
            std::pair<double,double> final_result = calculator->iteration2(calculator->THRES2 , temp_theta , temp_t , chosen_yaw , z , r);
            
            
            //发布消息
        rm_auto_aim::Ballistic::firemsg fire_msg;
            fire_msg.pitch = final_result.first;
            fire_msg.yaw = final_result.second;
            fire_msg.tracking = target_msg.tracking;
            fire_msg.id = target_msg.id;
           //串口发送
           serialdriver.sendData(fire_msg);
            }
                
                tracker.last_time_ = time;



    }
    else{
        std::cout<<"armors_msg is empty"<<std::endl;
    }
    }//将两个线程间通信分隔开


    if (cv::waitKey(1) == 'q') {
            capturing = false;
            break;
        }
      
    
        
        

         

    }
}