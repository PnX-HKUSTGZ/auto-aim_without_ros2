// Copyright 2022 Chen Jun

#ifndef ARMOR_PROCESSOR__PROCESSOR_NODE_HPP_
#define ARMOR_PROCESSOR__PROCESSOR_NODE_HPP_


// STD
#include <memory> 
#include <string>
#include <vector>

#include "../include/tracker.hpp"



namespace rm_auto_aim
{

class ArmorTracker
{
public:
   ArmorTracker();

  //parameters for ordinate system transformation
  std::vector<double> gimbal2camra;

  // Maximum allowable armor distance in the XOY plane
  double max_armor_distance_;

  // The time when the last message was received
  std::chrono::time_point<std::chrono::system_clock> last_time_;
  double dt_;

  // Armor tracker
  double s2qxyz_, s2qyaw_, s2qr_;
  double r_xyz_factor, r_yaw;
  double lost_time_thres_;
  std::unique_ptr<Tracker> tracker_;

  struct Vector3
  {
    double x;
    double y;
    double z;
  };

  struct Target
  {
    std::chrono::time_point<std::chrono::system_clock> timestamp;
    bool tracking;
    std::string id;
    int armors_num;
    rm_auto_aim::Detector::Point position;
    Vector3 velocity;
    double yaw;
    double v_yaw;
    double radius_1;
    double radius_2;
    double dz;
  };

};

}  // namespace rm_auto_aim

#endif  // ARMOR_PROCESSOR__PROCESSOR_NODE_HPP_
