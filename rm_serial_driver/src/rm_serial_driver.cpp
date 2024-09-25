// Copyright (c) 2022 ChenJun
// Licensed under the Apache-2.0 License.




#include "../include/rm_serial_driver.hpp"

#include"yaml-cpp/yaml.h"

// C++ system
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <exception>
#include "../include/crc.hpp"
#include "../include/packet.hpp"
#include "../include/rm_serial_driver.hpp"



namespace rm_serial_driver
{
RMSerialDriver::RMSerialDriver():
  owned_ctx_{new IoContext(2)},
  serial_driver_{new drivers::serial_driver::SerialDriver(*owned_ctx_)}
{
  std::cout << "start RMSerialDriver!" << std::endl;

  getParams();

  try {
    serial_driver_->init_port(device_name_, *device_config_);
    if (!serial_driver_->port()->is_open()) {
      serial_driver_->port()->open();
      
    }
  } catch (const std::exception& ex) {
    std::cerr << "Error creating serial port: " << device_name_ << " - " << ex.what() << std::endl;
}

}

RMSerialDriver::~RMSerialDriver()
{
  if (receive_thread_.joinable()) {
    receive_thread_.join();
  }

  if (serial_driver_->port()->is_open()) {
    serial_driver_->port()->close();
  }

  if (owned_ctx_) {
    owned_ctx_->waitForExit();
  }
}


void RMSerialDriver::getParams()//使用configyaml替换
{
  using FlowControl = drivers::serial_driver::FlowControl;
  using Parity = drivers::serial_driver::Parity;
  using StopBits = drivers::serial_driver::StopBits;

  uint32_t baud_rate{};
  auto fc = FlowControl::NONE;
  auto pt = Parity::NONE;
  auto sb = StopBits::ONE;
  //加载参数文件
  YAML::Node config = YAML::LoadFile("/home/pnx/training_code/config.yaml");
  std::cout << "config.yaml loaded" << std::endl;
  //读取参数
  timestamp_offset_ = config["serialdriver"]["timestamp_offset"].as<double>();

  try {
    device_name_ = config["serialdriver"]["device_name"].as<std::string>();
  } catch (YAML::Exception & ex) {
    std::cout<<"The device_name provided was invalid";
    throw ex;
  }

  try {
    baud_rate = config["serialdriver"]["baud_rate"].as<uint32_t>();
  } catch (YAML::Exception & ex) {
    std::cout<<"The baud_rate provided was invalid";
    throw ex;
  }

  try {
    const auto fc_string = config["serialdriver"]["flow_control"].as<std::string>();

    if (fc_string == "none") {
      fc = FlowControl::NONE;
    } else if (fc_string == "hardware") {
      fc = FlowControl::HARDWARE;
    } else if (fc_string == "software") {
      fc = FlowControl::SOFTWARE;
    } else {
      throw std::invalid_argument{
        "The flow_control parameter must be one of: none, software, or hardware."};
    }
  } catch (YAML::Exception & ex) {
    std::cout<<"The flow_control provided was invalid";
    throw ex;
  }

  //设置奇偶校验位
  try {
    const auto pt_string = config["serialdriver"]["parity"].as<std::string>();

    if (pt_string == "none") {
      pt = Parity::NONE;
    } else if (pt_string == "odd") {
      pt = Parity::ODD;
    } else if (pt_string == "even") {
      pt = Parity::EVEN;
    } else {
      throw std::invalid_argument{"The parity parameter must be one of: none, odd, or even."};
    }
  } catch (YAML::Exception & ex) {
    std::cout<<"The parity provided was invalid";
    throw ex;
  }


  try {
    const auto sb_string = config["serialdriver"]["stop_bits"].as<std::string>();

    if (sb_string == "1" || sb_string == "1.0") {
      sb = StopBits::ONE;
    } else if (sb_string == "1.5") {
      sb = StopBits::ONE_POINT_FIVE;
    } else if (sb_string == "2" || sb_string == "2.0") {
      sb = StopBits::TWO;
    } else {
      throw std::invalid_argument{"The stop_bits parameter must be one of: 1, 1.5, or 2."};
    }
  } catch (YAML::Exception & ex) {
    std::cout<<"The stop_bits provided was invalid";
    throw ex;
  }

  device_config_ =
    std::make_unique<drivers::serial_driver::SerialPortConfig>(baud_rate, fc, pt, sb);
    std::cout << "device_config_ created" << std::endl;
}

void RMSerialDriver::reopenPort()
{
  std::cerr << "Attempting to reopen port" << std::endl;
    try {
        if (serial_driver_->port()->is_open()) {
            serial_driver_->port()->close();
        }
        serial_driver_->port()->open();
        std::cout << "Successfully reopened port" << std::endl;
    } catch (const std::exception & ex) {
        std::cerr << "Error while reopening port: " << ex.what() << std::endl;
        
        if (true) {  // Wait for 1 second before trying to reopen the port
            std::this_thread::sleep_for(std::chrono::seconds(1));
            reopenPort();
        }
    }
}

void RMSerialDriver::sendData(const rm_auto_aim::Ballistic::firemsg &msg)
{
  const static std::map<std::string, uint8_t> id_unit8_map{
    {"", 0},  {"outpost", 0}, {"1", 1}, {"1", 1},     {"2", 2},
    {"3", 3}, {"4", 4},       {"5", 5}, {"guard", 6}, {"base", 7}};

  try {
    SendPacket packet;
    packet.tracking = msg.tracking;
    packet.id = id_unit8_map.at(msg.id);
    packet.pitch = msg.pitch;
    packet.yaw = msg.yaw;

    crc16::Append_CRC16_Check_Sum(reinterpret_cast<uint8_t *>(&packet), sizeof(packet));

    std::vector<uint8_t> data = toVector(packet);

    serial_driver_->port()->send(data);
    std::cout << "Data sent" << std::endl;  

    
  } catch (const std::exception &ex) {
    std::cerr << "Error while sending data: " << ex.what() << std::endl;
    reopenPort();
}
}



}  // namespace rm_serial_driver


