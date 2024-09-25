// Copyright (c) 2022 ChenJun
// Licensed under the Apache-2.0 License.

#ifndef RM_SERIAL_DRIVER__RM_SERIAL_DRIVER_HPP_
#define RM_SERIAL_DRIVER__RM_SERIAL_DRIVER_HPP_




#include <serial_driver/serial_driver.hpp>
#include"../../ballistic_calculation/inlude/ballistic_calculation.hpp"

#include <Eigen/Dense>
#include <Eigen/Geometry>


// C++ system
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>




namespace rm_serial_driver
{
class RMSerialDriver
{
public:
  RMSerialDriver();

  ~RMSerialDriver();

  void getParams();

  
  void reopenPort();

  void sendData(const rm_auto_aim::Ballistic::firemsg &msg);

 



  // Serial port
  std::unique_ptr<IoContext> owned_ctx_;
  std::string device_name_;
  std::unique_ptr<drivers::serial_driver::SerialPortConfig> device_config_;
  std::unique_ptr<drivers::serial_driver::SerialDriver> serial_driver_;

  // Param client to set detect_colr
  
  bool initial_set_param_ = false;
  uint8_t previous_receive_color_ = 0;

  double timestamp_offset_ = 0;
 
  std::thread receive_thread_;

  
};
}  // namespace rm_serial_driver

#endif  // RM_SERIAL_DRIVER__RM_SERIAL_DRIVER_HPP_
