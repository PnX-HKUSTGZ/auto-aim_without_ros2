#ifndef SERIAL_RECEIVER_HPP
#define SERIAL_RECEIVER_HPP

#include <queue>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <Eigen/Dense>
#include "rm_serial_driver/include/rm_serial_driver.hpp"
#include "rm_serial_driver/include/packet.hpp"
#include "rm_serial_driver/include/crc.hpp"

struct Transform {
    std::chrono::time_point<std::chrono::system_clock> timestamp;
    Eigen::Quaternionf q;
    bool reset_tracker;
};

class SerialReceiver {
public:
    SerialReceiver();
    void receiveData();

    static  std::queue<Transform> transformQueue;
    static std::mutex mtx2;
    static std::condition_variable asdf2;
    static rm_serial_driver::RMSerialDriver serialdriver;

private:
    
    const size_t MAX_QUEUE_SIZE = 1;
};

#endif // SERIAL_RECEIVER_HPP