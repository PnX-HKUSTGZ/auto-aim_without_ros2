#include "SerialReceiver.hpp"
#include <iostream>
#include <yaml-cpp/yaml.h>

SerialReceiver::SerialReceiver() {}

void SerialReceiver::receiveData() {
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
                bool crc_ok = crc16::Verify_CRC16_Check_Sum(reinterpret_cast<const uint8_t *>(&packet), sizeof(packet));
                if (crc_ok) {
                    if (!serialdriver.initial_set_param_ || packet.detect_color != serialdriver.previous_receive_color_) {
                        YAML::Node config = YAML::LoadFile("/home/pnx/training_code/config.yaml");
                        config["detector"]["detect_color"] = packet.detect_color;
                        serialdriver.previous_receive_color_ = packet.detect_color;
                    }

                    Transform t;
                    YAML::Node config = YAML::LoadFile("/home/pnx/training_code/config.yaml");
                    serialdriver.timestamp_offset_ = config["serialdriver"]["timestamp_offset"].as<double>();
                    t.timestamp = std::chrono::system_clock::now() + 
                        std::chrono::duration_cast<std::chrono::system_clock::duration>(std::chrono::duration<double>(serialdriver.timestamp_offset_));
                    t.q = Eigen::Quaternionf(packet.q[0], packet.q[1], packet.q[2], packet.q[3]);
                    t.reset_tracker = packet.reset_tracker;

                    std::unique_lock<std::mutex> lock(mtx2);
                    asdf2.wait(lock, [this]{ return transformQueue.size() < MAX_QUEUE_SIZE; });
                    transformQueue.push(t);
                    asdf2.notify_one();
                } else {
                    std::cerr << "CRC error!";
                }
            } else {
                std::cerr << "Header error!";
            }
        } catch (const std::exception & ex) {
            std::cerr << "Error while receiving data: " << ex.what();
            serialdriver.reopenPort();
        }
    }
}