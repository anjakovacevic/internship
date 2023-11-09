#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main()
{

    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        cout << "Unable to connect to camera" << endl;
        return 1;
    }

    return 0;
}