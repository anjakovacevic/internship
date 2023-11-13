#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    std::cout << "CUDA enabled: " << cv::cuda::getCudaEnabledDeviceCount() << std::endl;
    return 0;
}
