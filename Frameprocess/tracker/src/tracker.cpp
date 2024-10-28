// Copyright 2022 Chen Jun

#include "../include/tracker.hpp"

#include <Eigen/Dense>
#include <Eigen/Geometry>

// STD
#include <cfloat>
#include <memory>
#include <string>

namespace rm_auto_aim


// 利用扩展卡尔曼滤波器（EKF）来推算出当前目标装甲板所在的机器人的速度、角速度（偏航速度）等状态信息
{
// 构造追踪器为空的状态
Tracker::Tracker(double max_match_distance, double max_match_yaw_diff)
: tracker_state(LOST),
  tracked_id(std::string("")),
  measurement(Eigen::VectorXd::Zero(4)),
  target_state(Eigen::VectorXd::Zero(9)),
  max_match_distance_(max_match_distance),
  max_match_yaw_diff_(max_match_yaw_diff)
{
}
//初始化追踪器
void Tracker::init(const Armors & armors_msg)
{
  if (armors_msg.empty()) {
    return;
  }

  // Simply choose the armor that is closest to image center
  double min_distance = DBL_MAX;
  tracked_armor = armors_msg[0];//基于输入的装甲板消息选择最接近图像中心的装甲板作为追踪目标
  for (const auto & armor : armors_msg) {
    if (armor.distance_to_image_center < min_distance) {
      min_distance = armor.distance_to_image_center;
      tracked_armor = armor;
    }
  }

  initEKF(tracked_armor);//中心的装甲板作为追踪目标，并初始化EKF
  

  tracked_id = tracked_armor.number;
  tracker_state = DETECTING;//将追踪状态设为detecting

  updateArmorsNum(tracked_armor);//对追踪的装甲板进行分类？
}

void Tracker::update(const Armors & armors_msg)
//根据经过EKF加权后的观测和预测来更新装甲板的追踪状态
{
  // KF predict
  Eigen::VectorXd ekf_prediction = ekf.predict();//

  bool matched = false;
  // Use KF prediction as default target state if no matched armor is found
  target_state = ekf_prediction;

  if (!armors_msg.empty()) {
    // Find the closest armor with the same id
    Armor same_id_armor;
    int same_id_armors_count = 0;
    auto predicted_position = getArmorPositionFromState(ekf_prediction);//预测
    double min_position_diff = DBL_MAX;
    double yaw_diff = DBL_MAX;//导入差值上限
    for (const auto & armor : armors_msg) {//遍历所有观测到的装甲板
      // Only consider armors with the same id
      if (armor.number == tracked_id) {
        same_id_armor = armor;
        same_id_armors_count++;
        // Calculate the difference between the predicted position and the current armor position
        auto p = armor.pose.position;
        Eigen::Vector3d position_vec(p.x, p.y, p.z);
        double position_diff = (predicted_position - position_vec).norm();
        if (position_diff < min_position_diff) {
          // Find the closest armor
          min_position_diff = position_diff;
          yaw_diff = abs(orientationToYaw(armor.pose.orientation) - ekf_prediction(6));
          tracked_armor = armor;
        }
      }
    }

    // Store tracker info
    info_position_diff = min_position_diff;
    info_yaw_diff = yaw_diff;

    // Check if the distance and yaw difference of closest armor are within the threshold
    if (min_position_diff < max_match_distance_ && yaw_diff < max_match_yaw_diff_) {
      // Matched armor found
      matched = true;
      auto p = tracked_armor.pose.position;
      // Update EKF
      double measured_yaw = orientationToYaw(tracked_armor.pose.orientation);
      //四元数方向转换为偏航角
      measurement = Eigen::Vector4d(p.x, p.y, p.z, measured_yaw);
      target_state = ekf.update(measurement);
      //RCLCPP_DEBUG(rclcpp::get_logger("armor_tracker"), "EKF update");
    } else if (same_id_armors_count == 1 && yaw_diff > max_match_yaw_diff_) {
      // Matched armor not found, but there is only one armor with the same id
      // and yaw has jumped, take this case as the target is spinning and armor jumped
      handleArmorJump(same_id_armor);//处理装甲板跳变
    } else {
      // No matched armor found
      //RCLCPP_WARN(rclcpp::get_logger("armor_tracker"), "No matched armor found!");
    }
  }

  // Prevent radius from spreading
  if (target_state(8) < 0.12) {
    target_state(8) = 0.12;
    ekf.setState(target_state);
  } else if (target_state(8) > 0.4) {
    target_state(8) = 0.4;
    ekf.setState(target_state);
  }

  // Tracking state machine
  if (tracker_state == DETECTING) {
    if (matched) {
      detect_count_++;
      if (detect_count_ > tracking_thres) {
        detect_count_ = 0;
        tracker_state = TRACKING;
      }
    } else {
      detect_count_ = 0;
      tracker_state = LOST;
    }
  } else if (tracker_state == TRACKING) {
    if (!matched) {
      tracker_state = TEMP_LOST;
      lost_count_++;
    }
  } else if (tracker_state == TEMP_LOST) {
    if (!matched) {
      lost_count_++;
      if (lost_count_ > lost_thres) {
        lost_count_ = 0;
        tracker_state = LOST;
      }
    } else {
      tracker_state = TRACKING;
      lost_count_ = 0;
    }
  }
}

void Tracker::initEKF(const Armor & a)
{
  double xa = a.pose.position.x;
  double ya = a.pose.position.y;
  double za = a.pose.position.z;
  last_yaw_ = 0;
  double yaw = orientationToYaw(a.pose.orientation);

  // Set initial position at 0.2m behind the target
  target_state = Eigen::VectorXd::Zero(9);
  double r = 0.26;
  double xc = xa + r * cos(yaw);
  double yc = ya + r * sin(yaw);
  dz = 0, another_r = r;
  target_state << xc, 0, yc, 0, za, 0, yaw, 0, r;

  ekf.setState(target_state);
}

void Tracker::updateArmorsNum(const Armor & armor)
{
  if (armor.type == "large" && (tracked_id == "3" || tracked_id == "4" || tracked_id == "5")) {
    tracked_armors_num = ArmorsNum::BALANCE_2;
  } else if (tracked_id == "outpost") {
    tracked_armors_num = ArmorsNum::OUTPOST_3;
  } else {
    tracked_armors_num = ArmorsNum::NORMAL_4;
  }
}

void Tracker::handleArmorJump(const Armor & current_armor)
{
  double yaw = orientationToYaw(current_armor.pose.orientation);
  target_state(6) = yaw;
  updateArmorsNum(current_armor);
  // Only 4 armors has 2 radius and height
  if (tracked_armors_num == ArmorsNum::NORMAL_4) {
    dz = target_state(4) - current_armor.pose.position.z;//dz是处理跳变的
    target_state(4) = current_armor.pose.position.z;
    std::swap(target_state(8), another_r);
  }
  std::cout<<"Armor jump"<<std::endl;

  // If position difference is larger than max_match_distance_,
  // take this case as the ekf diverged, reset the state
  auto p = current_armor.pose.position;
  Eigen::Vector3d current_p(p.x, p.y, p.z);
  Eigen::Vector3d infer_p = getArmorPositionFromState(target_state);
  if ((current_p - infer_p).norm() > max_match_distance_) {
    double r = target_state(8);
    target_state(0) = p.x + r * cos(yaw);  // xc
    target_state(1) = 0;                   // vxc
    target_state(2) = p.y + r * sin(yaw);  // yc
    target_state(3) = 0;                   // vyc
    target_state(4) = p.z;                 // za
    target_state(5) = 0;                   // vza
    std::cout<<"Reset State!"<<std::endl;
  }

  ekf.setState(target_state);
  
}

double Tracker::orientationToYaw(const Eigen::Quaterniond &q) {
    // Convert quaternion to rotation matrix
    Eigen::Matrix3d rotationMatrix = q.toRotationMatrix();

    // Extract roll, pitch, and yaw from the rotation matrix
    double roll, pitch, yaw;
    pitch = std::asin(-rotationMatrix(2, 0));  // -asin(r31)
    
    if (std::abs(rotationMatrix(2, 0)) < 0.99999) {  // Not at a singularity
        roll = std::atan2(rotationMatrix(2, 1), rotationMatrix(2, 2));  // atan2(r32, r33)
        yaw = std::atan2(rotationMatrix(1, 0), rotationMatrix(0, 0));   // atan2(r21, r11)
    } else {  // Gimbal lock case
        roll = 0;
        yaw = std::atan2(-rotationMatrix(0, 1), rotationMatrix(1, 1));  // atan2(-r12, r22)
    }

    // Make yaw change continuous (-pi to pi to -inf to inf)
    yaw = last_yaw_ + std::atan2(std::sin(yaw - last_yaw_), std::cos(yaw - last_yaw_));
    last_yaw_ = yaw;

    return yaw;
}

Eigen::Vector3d Tracker::getArmorPositionFromState(const Eigen::VectorXd & x)//从EKF的状态向量计算装甲板的预测位置。
{
  // Calculate predicted position of the current armor
  double xc = x(0), yc = x(2), za = x(4);
  double yaw = x(6), r = x(8);
  double xa = xc - r * cos(yaw);
  double ya = yc - r * sin(yaw);
  return Eigen::Vector3d(xa, ya, za);
}

}  // namespace rm_auto_aim
