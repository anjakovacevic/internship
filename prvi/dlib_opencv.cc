// #include <iostream>
// #include <opencv2/highgui.hpp>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/core.hpp>
// #include <dlib/opencv.h>
// #include <dlib/image_processing.h>
// #include <dlib/image_processing/frontal_face_detector.h>
// // #include <opencv/calib3d/calib3d.hpp>
// // #include <opencv/imgproc/imgproc.hpp>
// int main()
// {
//     // open cam
//     cv::VideoCapture cap(0);
//     dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
//     if (!cap.isOpened())
//     {
//         std::cout << "Unable to connect to camera" << std::endl;
//         return EXIT_FAILURE;
//     }

//     // main loop
//     while (1)
//     {
//         // Grab a frame
//         cv::Mat temp;
//         cap >> temp;

//         // press esc to end
//         cv::imshow("demo", temp);
//         unsigned char key = cv::waitKey(1);
//         if (key == 27)
//         {
//             break;
//         }
//     }

//     return 0;
// }

// #include <iostream>
// #include <dlib/opencv.h>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/calib3d/calib3d.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
// #include <dlib/image_processing/frontal_face_detector.h>
// #include <dlib/image_processing/render_face_detections.h>
// #include <dlib/image_processing.h>

// // Intrisics can be calculated using opencv sample code under opencv/sources/samples/cpp/tutorial_code/calib3d
// // Normally, you can also apprximate fx and fy by image width, cx by half image width, cy by half image height instead
// double K[9] = {6.5308391993466671e+002, 0.0, 3.1950000000000000e+002, 0.0, 6.5308391993466671e+002, 2.3950000000000000e+002, 0.0, 0.0, 1.0};
// double D[5] = {7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000};

// int main()
// {
//     // open cam
//     cv::VideoCapture cap(0);
//     if (!cap.isOpened())
//     {
//         std::cout << "Unable to connect to camera" << std::endl;
//         return EXIT_FAILURE;
//     }
//     // Load face detection and pose estimation models (dlib).
//     dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
//     dlib::shape_predictor predictor;
//     dlib::deserialize("C:/Users/anja.kovacevic/datasets_and_models/shape_predictor_68_face_landmarks.dat") >> predictor;

//     // fill in cam intrinsics and distortion coefficients
//     cv::Mat cam_matrix = (cv::Mat_<double>(3, 3) << K[0], 0, K[2], 0, K[4], K[5], 0, 0, 1);
//     cv::Mat dist_coeffs = (cv::Mat_<double>(1, 5) << D[0], D[1], 0, 0, D[4]);

//     // fill in 3D ref points(world coordinates), model referenced from http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
//     std::vector<cv::Point3d> object_pts;
//     object_pts.push_back(cv::Point3d(6.825897, 6.760612, 4.402142));   // #33 left brow left corner
//     object_pts.push_back(cv::Point3d(1.330353, 7.122144, 6.903745));   // #29 left brow right corner
//     object_pts.push_back(cv::Point3d(-1.330353, 7.122144, 6.903745));  // #34 right brow left corner
//     object_pts.push_back(cv::Point3d(-6.825897, 6.760612, 4.402142));  // #38 right brow right corner
//     object_pts.push_back(cv::Point3d(5.311432, 5.485328, 3.987654));   // #13 left eye left corner
//     object_pts.push_back(cv::Point3d(1.789930, 5.393625, 4.413414));   // #17 left eye right corner
//     object_pts.push_back(cv::Point3d(-1.789930, 5.393625, 4.413414));  // #25 right eye left corner
//     object_pts.push_back(cv::Point3d(-5.311432, 5.485328, 3.987654));  // #21 right eye right corner
//     object_pts.push_back(cv::Point3d(2.005628, 1.409845, 6.165652));   // #55 nose left corner
//     object_pts.push_back(cv::Point3d(-2.005628, 1.409845, 6.165652));  // #49 nose right corner
//     object_pts.push_back(cv::Point3d(2.774015, -2.080775, 5.048531));  // #43 mouth left corner
//     object_pts.push_back(cv::Point3d(-2.774015, -2.080775, 5.048531)); // #39 mouth right corner
//     object_pts.push_back(cv::Point3d(0.000000, -3.116408, 6.097667));  // #45 mouth central bottom corner
//     object_pts.push_back(cv::Point3d(0.000000, -7.415691, 4.070434));  // #6 chin corner

//     // 2D ref points(image coordinates), referenced from detected facial feature
//     std::vector<cv::Point2d> image_pts;

//     // result
//     cv::Mat rotation_vec;                       // 3 x 1
//     cv::Mat rotation_mat;                       // 3 x 3 R
//     cv::Mat translation_vec;                    // 3 x 1 T
//     cv::Mat pose_mat = cv::Mat(3, 4, CV_64FC1); // 3 x 4 R | T
//     cv::Mat euler_angle = cv::Mat(3, 1, CV_64FC1);

//     // reproject 3D points world coordinate axis to verify result pose
//     std::vector<cv::Point3d> reprojectsrc;
//     reprojectsrc.push_back(cv::Point3d(10.0, 10.0, 10.0));
//     reprojectsrc.push_back(cv::Point3d(10.0, 10.0, -10.0));
//     reprojectsrc.push_back(cv::Point3d(10.0, -10.0, -10.0));
//     reprojectsrc.push_back(cv::Point3d(10.0, -10.0, 10.0));
//     reprojectsrc.push_back(cv::Point3d(-10.0, 10.0, 10.0));
//     reprojectsrc.push_back(cv::Point3d(-10.0, 10.0, -10.0));
//     reprojectsrc.push_back(cv::Point3d(-10.0, -10.0, -10.0));
//     reprojectsrc.push_back(cv::Point3d(-10.0, -10.0, 10.0));

//     // reprojected 2D points
//     std::vector<cv::Point2d> reprojectdst;
//     reprojectdst.resize(8);

//     // temp buf for decomposeProjectionMatrix()
//     cv::Mat out_intrinsics = cv::Mat(3, 3, CV_64FC1);
//     cv::Mat out_rotation = cv::Mat(3, 3, CV_64FC1);
//     cv::Mat out_translation = cv::Mat(3, 1, CV_64FC1);

//     // text on screen
//     std::ostringstream outtext;

//     // main loop
//     while (1)
//     {
//         // Grab a frame
//         cv::Mat temp;
//         cap >> temp;
//         dlib::cv_image<dlib::bgr_pixel> cimg(temp);

//         // Detect faces
//         std::vector<dlib::rectangle> faces = detector(cimg);

//         // Find the pose of each face
//         if (faces.size() > 0)
//         {
//             // track features
//             dlib::full_object_detection shape = predictor(cimg, faces[0]);

//             // draw features
//             for (unsigned int i = 0; i < 68; ++i)
//             {
//                 cv::circle(temp, cv::Point(shape.part(i).x(), shape.part(i).y()), 2, cv::Scalar(0, 0, 255), -1);
//             }

//             // fill in 2D ref points, annotations follow https://ibug.doc.ic.ac.uk/resources/300-W/
//             image_pts.push_back(cv::Point2d(shape.part(17).x(), shape.part(17).y())); // #17 left brow left corner
//             image_pts.push_back(cv::Point2d(shape.part(21).x(), shape.part(21).y())); // #21 left brow right corner
//             image_pts.push_back(cv::Point2d(shape.part(22).x(), shape.part(22).y())); // #22 right brow left corner
//             image_pts.push_back(cv::Point2d(shape.part(26).x(), shape.part(26).y())); // #26 right brow right corner
//             image_pts.push_back(cv::Point2d(shape.part(36).x(), shape.part(36).y())); // #36 left eye left corner
//             image_pts.push_back(cv::Point2d(shape.part(39).x(), shape.part(39).y())); // #39 left eye right corner
//             image_pts.push_back(cv::Point2d(shape.part(42).x(), shape.part(42).y())); // #42 right eye left corner
//             image_pts.push_back(cv::Point2d(shape.part(45).x(), shape.part(45).y())); // #45 right eye right corner
//             image_pts.push_back(cv::Point2d(shape.part(31).x(), shape.part(31).y())); // #31 nose left corner
//             image_pts.push_back(cv::Point2d(shape.part(35).x(), shape.part(35).y())); // #35 nose right corner
//             image_pts.push_back(cv::Point2d(shape.part(48).x(), shape.part(48).y())); // #48 mouth left corner
//             image_pts.push_back(cv::Point2d(shape.part(54).x(), shape.part(54).y())); // #54 mouth right corner
//             image_pts.push_back(cv::Point2d(shape.part(57).x(), shape.part(57).y())); // #57 mouth central bottom corner
//             image_pts.push_back(cv::Point2d(shape.part(8).x(), shape.part(8).y()));   // #8 chin corner

//             // calc pose
//             cv::solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs, rotation_vec, translation_vec);

//             // reproject
//             cv::projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs, reprojectdst);

//             // draw axis
//             cv::line(temp, reprojectdst[0], reprojectdst[1], cv::Scalar(0, 0, 255));
//             cv::line(temp, reprojectdst[1], reprojectdst[2], cv::Scalar(0, 0, 255));
//             cv::line(temp, reprojectdst[2], reprojectdst[3], cv::Scalar(0, 0, 255));
//             cv::line(temp, reprojectdst[3], reprojectdst[0], cv::Scalar(0, 0, 255));
//             cv::line(temp, reprojectdst[4], reprojectdst[5], cv::Scalar(0, 0, 255));
//             cv::line(temp, reprojectdst[5], reprojectdst[6], cv::Scalar(0, 0, 255));
//             cv::line(temp, reprojectdst[6], reprojectdst[7], cv::Scalar(0, 0, 255));
//             cv::line(temp, reprojectdst[7], reprojectdst[4], cv::Scalar(0, 0, 255));
//             cv::line(temp, reprojectdst[0], reprojectdst[4], cv::Scalar(0, 0, 255));
//             cv::line(temp, reprojectdst[1], reprojectdst[5], cv::Scalar(0, 0, 255));
//             cv::line(temp, reprojectdst[2], reprojectdst[6], cv::Scalar(0, 0, 255));
//             cv::line(temp, reprojectdst[3], reprojectdst[7], cv::Scalar(0, 0, 255));

//             // calc euler angle
//             cv::Rodrigues(rotation_vec, rotation_mat);
//             cv::hconcat(rotation_mat, translation_vec, pose_mat);
//             cv::decomposeProjectionMatrix(pose_mat, out_intrinsics, out_rotation, out_translation, cv::noArray(), cv::noArray(), cv::noArray(), euler_angle);

//             // show angle result
//             outtext << "X: " << std::setprecision(3) << euler_angle.at<double>(0);
//             cv::putText(temp, outtext.str(), cv::Point(50, 40), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0));
//             outtext.str("");
//             outtext << "Y: " << std::setprecision(3) << euler_angle.at<double>(1);
//             cv::putText(temp, outtext.str(), cv::Point(50, 60), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0));
//             outtext.str("");
//             outtext << "Z: " << std::setprecision(3) << euler_angle.at<double>(2);
//             cv::putText(temp, outtext.str(), cv::Point(50, 80), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0));
//             outtext.str("");

//             image_pts.clear();
//         }

//         // press esc to end
//         cv::imshow("demo", temp);
//         unsigned char key = cv::waitKey(1);
//         if (key == 27)
//         {
//             break;
//         }
//     }

//     return 0;
// }

/************************************************************************************************************************
 * Author     : Manu BN
 * Description: Face Alignment using Dlib & OpenCV for a single face.
 *              Haar Cascade is used only for cropping the face at the end
 *              Download "shape_predictor_68_face_landmarks.dat" file from "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" *
 * Contact    : manubn88@gmail.com
 ***********************************************************************************************************************/

#include "opencv2/objdetect/objdetect.hpp"
#include "dlib/image_processing/shape_predictor.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>
#include <iostream>
// #include "opencv2/photo.hpp"
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <opencv2/opencv.hpp>
// #include <opencv2/objdetect/objdetect_c.h>

using namespace std;
using namespace cv;
using namespace dlib;

// function declarations
void Rect2rectangle(Rect &r, dlib::rectangle &rec);

void dlib_point2cv_Point(full_object_detection &S, std::vector<Point> &L, double &scale);

int main()
{

    // Image Path
    Mat image = imread("/home/user/Cartoon/lenna.png");
    // Declaring a variable "image" to store input image given.
    Mat gray, resized, detected_edges, Laugh_L, Laugh_R;

    // try block, tries to read the image and if there is a problem reading the image,it throws an exception.

    // image = imread("/home/user/Cartoon/Indian Face/SHIBAN QADARI.jpg");
    if (image.empty())
    {
        throw(0);
        cout << "Exception: Image is not read properly, please check image path" << endl;
        return -1;
    }

    // Resize
    // cv::resize(image,image,Size(480,640));

    // scale for resizing.
    double scale = 1;

    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor pose_model;

    // converts original image to gray scale and stores it in "gray".
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // resize the gray scale image for speeding the face detection.
    cv::resize(gray, resized, Size(), scale, scale);

    // converts original image to gray scale and stores it in "gray".
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // resize the gray scale image for speeding the face detection.
    cv::resize(gray, resized, Size(), scale, scale);

    // cout<<"Resized Image"<<" Rows "<< resized.rows<<" Cols "<<resized.cols<<endl;
    //  cout<<"Original Image"<<" Rows "<< image.rows<<" Cols "<<image.cols<<endl;
    // Histogram equalization is performed on the resized image to improve the contrast of the image which can help in detection.
    equalizeHist(resized, resized);

    // Canny( resized, detected_edges, 50,200, 3 );

    /**Object of Shape predictor class "sp" is created to load "shape_predictor_68_face_landmarks.dat" file which is a pre-trained
       cascade of regression tree implemented using "One Millisecond face alignment with an ensemble of regression trees"*/
    shape_predictor sp;
    deserialize("C:/Users/anja.kovacevic/datasets_and_models/shape_predictor_68_face_landmarks.dat") >> sp;

    // Conver OpenCV image Dlib image i.e. cimg
    cv_image<rgb_pixel> cimg(image);

    // Detect faces using DLib
    std::vector<dlib::rectangle> face_DLib = detector(cimg);

    if (face_DLib.empty())
    {
        cout << "No face is deteced by DLib" << endl;
    }

    // Convert DLib Rect to OpenCV Rect
    cv::Rect R;
    R.x = face_DLib[0].left();
    R.y = face_DLib[0].top();
    R.width = face_DLib[0].width();
    R.height = face_DLib[0].height();

    // Draw the Dlib detected face on the image
    //  cv::rectangle(image,R, Scalar(0,255, 0), 1, 1);

    // for(unsigned int i=0; i<shapes.size();++i)             // if running on many faces
    // {
    // landmarks vector is declared to store the 68 landmark points. The rest are for individual face components
    std::vector<cv::Point> landmarks, R_Eyebrow, L_Eyebrow, L_Eye, R_Eye, Mouth, Jaw_Line, Nose;

    /**at each index of "shapes" vector there is an object of full_object_detection class which stores the 68 landmark
    points in dlib::point from, which needs to be converted back to cv::Point for displaying.*/
    // std::vector<full_object_detection> shape;
    full_object_detection shape = sp(dlib::cv_image<unsigned char>(resized), face_DLib[0]);
    dlib_point2cv_Point(shape, landmarks, scale);

    // Extract each part using the pre fixed indicies
    for (size_t s = 0; s < landmarks.size(); s++)
    {
        // circle(image,landmarks[s], 2.0, Scalar( 255,0,0 ), 1, 8 );
        // putText(image,to_string(s),landmarks[s],FONT_HERSHEY_PLAIN,0.8,Scalar(0,0,0));

        // Right Eyebrow indicies
        if (s >= 22 && s <= 26)
        {
            R_Eyebrow.push_back(landmarks[s]);
            // circle( image,landmarks[s], 2.0, Scalar( 0, 0, 255 ), 1, 8 );
        }
        // Left Eyebrow indicies
        else if (s >= 17 && s <= 21)
        {
            L_Eyebrow.push_back(landmarks[s]);
        }
        // Left Eye indicies
        else if (s >= 36 && s <= 41)
        {
            L_Eye.push_back(landmarks[s]);
        }
        // Right Eye indicies
        else if (s >= 42 && s <= 47)
        {
            R_Eye.push_back(landmarks[s]);
        }
        // Mouth indicies
        else if (s >= 48 && s <= 67)
        {
            Mouth.push_back(landmarks[s]);
        }
        // Jawline Indicies
        else if (s >= 0 && s <= 16)
        {
            Jaw_Line.push_back(landmarks[s]);
        }
        // Nose Indicies
        else if (s >= 27 && s <= 35)
        {
            Nose.push_back(landmarks[s]);
        }
    }

    // 2D image points. If you change the image, you need to change vector
    std::vector<cv::Point2d> image_points;
    image_points.push_back(landmarks[30]); // Nose tip
    image_points.push_back(landmarks[8]);  // Chin
    image_points.push_back(landmarks[45]); // Left eye left corner
    image_points.push_back(landmarks[36]); // Right eye right corner
    image_points.push_back(landmarks[54]); // Left Mouth corner
    image_points.push_back(landmarks[48]); // Right mouth corner

    // 3D model points.
    std::vector<cv::Point3d> model_points;
    model_points.push_back(cv::Point3d(0.0f, 0.0f, 0.0f));          // Nose tip
    model_points.push_back(cv::Point3d(0.0f, -330.0f, -65.0f));     // Chin
    model_points.push_back(cv::Point3d(-225.0f, 170.0f, -135.0f));  // Left eye left corner
    model_points.push_back(cv::Point3d(225.0f, 170.0f, -135.0f));   // Right eye right corner
    model_points.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f)); // Left Mouth corner
    model_points.push_back(cv::Point3d(150.0f, -150.0f, -125.0f));  // Right mouth corner

    // Camera internals
    double focal_length = image.cols; // Approximate focal length.
    Point2d center = cv::Point2d(image.cols / 2, image.rows / 2);
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0, center.x, 0, focal_length, center.y, 0, 0, 1);
    cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type); // Assuming no lens distortion

    // cout << "Camera Matrix " << endl << camera_matrix << endl ;
    // Output rotation and translation
    cv::Mat rotation_vector; // Rotation in axis-angle form
    cv::Mat translation_vector;

    // Solve for pose
    cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

    // Draw line between two eyes
    cv::line(image, landmarks[45], landmarks[36], Scalar(0, 255, 0), 4);

    // Access the last element in the Rotation Vector
    double rot = rotation_vector.at<double>(0, 2);

    // Conver to degrees
    double theta_deg = rot / 3.14 * 180;

    cout << theta_deg << " In degrees" << endl;

    Mat dst;
    // Rotate around the center
    Point2f pt(image.cols / 2., image.rows / 2.);
    Mat r = getRotationMatrix2D(pt, theta_deg, 1.0);

    // determine bounding rectangle
    cv::Rect bbox = cv::RotatedRect(pt, image.size(), theta_deg).boundingRect();

    // adjust transformation matrix
    r.at<double>(0, 2) += bbox.width / 2.0 - center.x;
    r.at<double>(1, 2) += bbox.height / 2.0 - center.y;

    // Apply affine transform
    warpAffine(image, dst, r, bbox.size());

    // Now detect face using Haar Cascade in the image
    CascadeClassifier face_cascade;
    face_cascade.load("C:/Users/anja.kovacevic/datasets_and_models/haarcascade_frontalface_alt2.xml");
    std::vector<Rect> face;
    face_cascade.detectMultiScale(dst, face, 1.1, 2, 0 | CASCADE_SCALE_IMAGE);

    cv::rectangle(dst, face[0], Scalar(0, 0, 0), 1, 1);

    // Now crop the face
    Mat Cropped_Face = dst(face[0]);

    // Display image.
    // cv::imshow("Output", dst);

    imshow("Original Image", image);

    imshow("Cropped Face", Cropped_Face);

    waitKey(0);

    return 0;
}

/**-------------------------------------------------------------------------------*/

// Function Definitions.

/** This function converts cv::Rect to dlib::rectangle.
    This function is needed because in this implementation I have used opencv and dlib bothand they
    both have their own image processing library so when using a dlib method, its arguments should be
    as expected */
void Rect2rectangle(Rect &r, dlib::rectangle &rec)
{
    rec = dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1, (long)r.br().y - 1);
}

/** This function converts dlib::point to cv::Point and stores in a vector of landmarks
    This function is needed because in this implementation I have used opencv and dlib bothand they
    both have their own image processing library so when using a dlib method, its arguments should be
    as expected */
void dlib_point2cv_Point(full_object_detection &S, std::vector<Point> &L, double &scale)
{
    for (unsigned int i = 0; i < S.num_parts(); ++i)
    {
        L.push_back(Point(S.part(i).x() * (1 / scale), S.part(i).y() * (1 / scale)));
    }
}
