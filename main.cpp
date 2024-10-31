#include <thread>
#include "Camera.hpp"
#include "Frameprocess/FrameProcessor.hpp"
#include "SerialReceiver/SerialReceiver.hpp"

int main() {
    Camera camera;
    FrameProcessor frameProcessor;
    SerialReceiver serialReceiver;

    std::thread captureThread(&Camera::startCapture, &camera);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    std::thread receiveThread(&SerialReceiver::receiveData, &serialReceiver);
    std::thread processThread(&FrameProcessor::processFrames, &frameProcessor);
    
    captureThread.join();
    processThread.join();
    receiveThread.join();

    return 0;
}