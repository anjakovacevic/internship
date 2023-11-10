#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <iostream>
#include <vector>

#include <dlib/geometry/vector.h>

using namespace dlib;
using namespace std;


double eye_aspect_ratio(const dlib::full_object_detection& landmarks, const std::vector<unsigned long>& points) {
    dlib::point p1 = landmarks.part(points[0]);
    dlib::point p2 = landmarks.part(points[1]);
    dlib::point p3 = landmarks.part(points[2]);
    dlib::point p4 = landmarks.part(points[3]);
    dlib::point p5 = landmarks.part(points[4]);
    dlib::point p6 = landmarks.part(points[5]);

    double vertical1 = dlib::length(p2 - p6);
    double vertical2 = dlib::length(p3 - p5);
    double horizontal = dlib::length(p1 - p4);

    double ear = (vertical1 + vertical2) / (2.0 * horizontal);
    return ear;
}


int main() {
    const double EYE_AR_THRESH = 0.2; 
    const int EYE_AR_CONSEC_FRAMES = 48; // number of consecutive frames to check for
    int blink_counter = 0;
    bool is_drowsy = false;

    try {
        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }

        image_window win;

        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("C:/Users/anja.kovacevic/datasets_and_models/shape_predictor_68_face_landmarks.dat") >> pose_model;

        // Grab and process frames until the main window is closed by the user.
        while (!win.is_closed()) {
            // Grab a frame
            cv::Mat temp;
            if (!cap.read(temp)) {
                break;
            }
            cv_image<bgr_pixel> cimg(temp);

            // Detect faces
            std::vector<rectangle> faces = detector(cimg);
            std::vector<full_object_detection> shapes;
            for (unsigned long i = 0; i < faces.size(); ++i) {
                full_object_detection shape = pose_model(cimg, faces[i]);
                shapes.push_back(shape);
                double leftEAR = eye_aspect_ratio(shape, {36, 37, 38, 39, 40, 41});
                double rightEAR = eye_aspect_ratio(shape, {42, 43, 44, 45, 46, 47});
                double avgEAR = (leftEAR + rightEAR) / 2.0;

                if (avgEAR < EYE_AR_THRESH) {
                    blink_counter++;
                    if (blink_counter >= EYE_AR_CONSEC_FRAMES) {
                        is_drowsy = true;
                        // Trigger Drowsiness Alert
                        // For example, display a warning message or sound an alarm
                    }
                } else {
                    blink_counter = 0;
                    is_drowsy = false;
                }   
            }

            if (is_drowsy) {
                // Set the text characteristics
                string text = "ALERT!";
                int fontFace = cv::FONT_HERSHEY_SIMPLEX;
                double fontScale = 1;
                int thickness = 2;
                cv::Point textOrg(10, 30); // Change coordinates as needed

                // Put the text on the frame
                cv::putText(temp, text, textOrg, fontFace, fontScale, cv::Scalar(0, 0, 255), thickness);
            }

            cv_image<bgr_pixel> updated_cimg(temp);
            
            win.clear_overlay();
            win.set_image(updated_cimg);
            // win.add_overlay(render_face_detections(shapes));
        }
    } catch (exception& e) {
        cout << e.what() << endl;
    }
    return 0;
}
