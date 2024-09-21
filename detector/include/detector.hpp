// Copyright 2022 Chen Jun
// Licensed under the MIT License.

#ifndef ARMOR_DETECTOR__DETECTOR_HPP_
#define ARMOR_DETECTOR__DETECTOR_HPP_

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>

// STD

#include <string>
#include <vector>

#include "armor.hpp"
#include "number_classifier.hpp"
#include <Eigen/Geometry> 
#include <chrono>


namespace rm_auto_aim
{
class Detector
{
public:
  struct Point
  {
    double x;
    double y;
    double z;

  };

  struct Pose
  {
    Point position;
    Eigen::Quaterniond orientation;
  };

//定义包含装甲板所有信息的结构体，负责位姿估计结束后与状态估计模块的信息交互
  
   struct Armormsg
  {
    std::chrono::time_point<std::chrono::system_clock> timestamp;
    std::string number;
    std::string type;
    float distance_to_image_center;
    Pose pose;

  };
  
  
 

  struct LightParams //灯条选择的限制参数
  {
    // width / height
    double min_ratio;
    double max_ratio;
    // vertical angle
    double max_angle;
  };

  struct ArmorParams//装甲板选择的限制参数
  {
    double min_light_ratio;
    // light pairs distance
    double min_small_center_distance;
    double max_small_center_distance;
    double min_large_center_distance;
    double max_large_center_distance;
    // horizontal angle
    double max_angle;
  };

  Detector(const int & bin_thres, const int & color, const LightParams & l, const ArmorParams & a);

  std::vector<Armor> detect(const cv::Mat & input);

  cv::Mat preprocessImage(const cv::Mat & input);
  std::vector<Light> findLights(const cv::Mat & rbg_img, const cv::Mat & binary_img);
  std::vector<Armor> matchLights(const std::vector<Light> & lights);

  // For debug usage
  cv::Mat getAllNumbersImage();
  void drawResults(cv::Mat & img);

  int binary_thres;
  int detect_color;
  LightParams l;
  ArmorParams a;

  std::unique_ptr<NumberClassifier> classifier;
  cv::Mat binary_img;

  

private:
  bool isLight(const Light & possible_light);
  bool containLight(
    const Light & light_1, const Light & light_2, const std::vector<Light> & lights);
  ArmorType isArmor(const Light & light_1, const Light & light_2);

  std::vector<Light> lights_;
  std::vector<Armor> armors_;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__DETECTOR_HPP_
