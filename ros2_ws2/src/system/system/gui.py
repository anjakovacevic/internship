#!/usr/bin/env python3

# This is a very simple gui for showcasing the ros2 functionalities with my 
# implementations of gaze capture, drowsiness and user recognition.
# It runs a ROS2 node "gui_node" which gathers the info from nodes running using
# the launch_file for these operations. 
# If you would like to show the gaze capture results or track the drowsiness, 
# click on the appropriate buttons.
# For the login or user recognition, there will be text indicating if system knows
# you already or you need to register.
# One registered, the system should geet you at your next arrival.

import sys
import cv2
import threading
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QFont
import rclpy
from sensor_msgs.msg import Image
from custom_interfaces.msg import Gaze
from rclpy.qos import QoSProfile
from cv_bridge import CvBridge 
from scipy.spatial import distance as dist
from PyQt5.QtCore import QTimer

import face_recognition
import pickle
import os
import datetime
import subprocess

class GazeCaptureGUI(QWidget):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.predicted_pose_subscriber = self.node.create_subscription(
            Gaze, 'predicted_pose', self.predicted_pose_callback, QoSProfile(depth=10))
        self.subscription = self.node.create_subscription(
            Image, 'image_raw', self.webcam_callback, 10)
        self.sub_face = self.node.create_subscription(Image, 'roi_face', self.face_callback, 10)
        
        self.db_dir = "./db"
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)
        self.log_path = "./log.txt"

        self.cv_bridge = CvBridge()    

        self.video_size = (640, 480)
        self.initUI()

        self.cv_bridge = CvBridge()
        self.predicted_pose = None
        self.button_clicked = False 
        self.button2_clicked = False
        self.most_recent_capture_arr = None

        self.current_face_encoding = None
        self.known_faces = self.load_known_faces()
        self.face_recognition_timer = QTimer(self)
        self.face_recognition_timer.timeout.connect(self.attempt_recognition)
        self.face_recognition_timer.start(2000)  # Check every 2 seconds


        # self.identification()

    def load_known_faces(self):
        """Load known face encodings and their names from the database directory."""
        known_faces = {}
        for filename in os.listdir(self.db_dir):
            if filename.endswith(".pickle"):
                with open(os.path.join(self.db_dir, filename), "rb") as file:
                    known_faces[filename[:-7]] = pickle.load(file)  # Remove '.pickle' from filename
        return known_faces
    
    def initUI(self):
        self.setWindowTitle('ROS2 Project')
        self.setGeometry(100, 100, self.video_size[0] + 300, self.video_size[1] + 200)
        
        self.setStyleSheet("""
        QWidget {
            background-color: #2b2b2b;
        }
        QPushButton {
            background-color: #3c3f41;
            color: white;
            border: 1px solid #555;
            border-radius: 5px;
            padding: 5px;
            min-height: 30px;
        }
        QPushButton:hover {
            background-color: #505354;
        }
        QLabel {
            color: #aaa;
        }
        """)

        layout = QHBoxLayout()  
        self.setLayout(layout)

        self.label = QLabel()
        layout.addWidget(self.label)

        layout.addStretch(1) 

        button_text_layout = QVBoxLayout()
        button_text_layout.setAlignment(Qt.AlignLeft) 

        self.button1 = QPushButton("Gaze Capture")
        self.button1.clicked.connect(self.detectGaze)
        button_text_layout.addWidget(self.button1)

        self.dynamic_text1 = QLabel()
        button_text_layout.addWidget(self.dynamic_text1)
        self.dynamic_text1.setAlignment(Qt.AlignCenter)

        self.button2 = QPushButton("Drowsiness")
        self.button2.clicked.connect(self.drowsiness)
        button_text_layout.addWidget(self.button2)

        self.dynamic_text2 = QLabel()
        button_text_layout.addWidget(self.dynamic_text2)
        self.dynamic_text2.setAlignment(Qt.AlignCenter)

        # self.login_button = QPushButton('Login', self)
        # self.login_button.clicked.connect(self.login)
        # button_text_layout.addWidget(self.login_button)

        self.attendance_message_label = QLabel(self)
        self.attendance_message_label.setText(" ")
        button_text_layout.addWidget(self.attendance_message_label)

        self.register_button = QPushButton('Register New User', self)
        self.register_button.clicked.connect(self.register_new_user)
        button_text_layout.addWidget(self.register_button)

        layout.addLayout(button_text_layout)

        layout.addStretch(1)
        
        font = QFont("Arial", 12)
        self.label.setFont(font)
        self.button1.setFont(font)
        self.button2.setFont(font)
        self.register_button.setFont(font)
        self.dynamic_text1.setFont(font)
        self.dynamic_text2.setFont(font)
        # self.login_button.setFont(font)
        self.attendance_message_label.setFont(font)

    def webcam_callback(self, image):
        cv_image = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        self.most_recent_capture_arr = rgb_image
        qt_image = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        pixmap = pixmap.scaled(self.video_size[0], self.video_size[1], Qt.KeepAspectRatio)
        self.label.setPixmap(pixmap)

    def predicted_pose_callback(self, msg):
        self.predicted_pose = msg.predicted_pose
        if self.predicted_pose is not None and self.button_clicked:  # Only update if button clicked
            formatted_pose = [f'{pose:.2f}' for pose in self.predicted_pose]
            pose_str = ', '.join(formatted_pose)
            self.dynamic_text1.setText(f'Predicted Pose: {pose_str}')
        else:
            self.dynamic_text1.setText("Start")

    def detectGaze(self):
        print("Button Clicked!")
        self.button_clicked = True  

    def face_callback(self, data):
        self.face = self.cv_bridge.imgmsg_to_cv2(data, 'bgr8')

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def drowsiness(self):
        gray_face = cv2.cvtColor(self.face, cv2.COLOR_BGR2GRAY)
        eyes = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        eye_rects = eyes.detectMultiScale(
            gray_face, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        for ex, ey, ew, eh in eye_rects:
            eye = gray_face[ey : ey + eh, ex : ex + ew]
            ear = self.eye_aspect_ratio(eye)

            if ear < 0.7 and self.button2_clicked:
                self.dynamic_text2.setText("DROWSINESS!")
            else:
                self.dynamic_text2.setText("OK")

    def log_user(self, name):
        """Log the recognized user."""
        welcome_text = f"Welcome back, {name}!"
        self.attendance_message_label.setText(welcome_text)
        # Logging logic...

    def suggest_registration_or_retry(self):
        """Suggest the user to register or retry login."""
        self.attendance_message_label.setText("Unrecognized face. Please register or retry.")


    def attempt_recognition(self):
        """Attempt to recognize the current face. If unrecognized, offer registration or retry."""
        if self.most_recent_capture_arr is None:
            return

        # Convert the most recent capture to an encoding
        face_locations = face_recognition.face_locations(self.most_recent_capture_arr)
        face_encodings = face_recognition.face_encodings(self.most_recent_capture_arr, face_locations)

        if face_encodings:
            # Only consider the first face detected for simplicity
            self.current_face_encoding = face_encodings[0]

            # See if this face matches any of our known faces
            matches = face_recognition.compare_faces(list(self.known_faces.values()), self.current_face_encoding)
            if True in matches:
                first_match_index = matches.index(True)
                name = list(self.known_faces.keys())[first_match_index]
                self.log_user(name)
            else:
                # No known face matches, suggest registration or retry
                self.suggest_registration_or_retry()
        else:
            self.attendance_message_label.setText("No face detected. Please try again.")

    def login(self):
        unknown_img_path = "./.tmp.jpg"
        cv2.imwrite(unknown_img_path, self.most_recent_capture_arr)
        output = str(
            subprocess.check_output(["face_recognition", self.db_dir, unknown_img_path])
        )
        name = output.split(",")[1][:-5]

        if name in ["unknown_person", "no_persons_found"]:
            # Optionally, clear the label or show an error message
            self.attendance_message_label.setText("Login failed. Please try again.")
        else:
            welcome_text = f"Welcome back, {name}!"
            self.attendance_message_label.setText(welcome_text)
            with open(self.log_path, "a") as f:
                f.write(f"{name}, {datetime.datetime.now()}\n")

        os.remove(unknown_img_path)

    def register_new_user(self):
        # Placeholder for user registration logic
        self.registration_window = QWidget()
        self.registration_window.setWindowTitle('Register New User')
        self.registration_window.setGeometry(100, 100, 400, 200)

        self.registration_window.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
            }
            QPushButton {
                background-color: #3c3f41;
                color: white;
                border: 1px solid #555;
                border-radius: 5px;
                padding: 5px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #505354;
            }
            QLabel, QLineEdit {
                color: #aaa;
                border: 1px solid #555;
                border-radius: 5px;
                padding: 5px;
            }
            """)
        
        layout = QVBoxLayout()
        self.name_entry = QLineEdit(self.registration_window)
        self.register_btn = QPushButton('Register')
        self.register_btn.clicked.connect(self.perform_registration)
        
        layout.addWidget(QLabel('Enter user name:'))
        layout.addWidget(self.name_entry)
        layout.addWidget(self.register_btn)
        
        self.registration_window.setLayout(layout)
        self.registration_window.show()
    
    def perform_registration(self):
        user_name = self.name_entry.text()
        if user_name:
            known_img_path = os.path.join(self.db_dir, f"{user_name}.jpg")
            cv2.imwrite(known_img_path, self.most_recent_capture_arr)
            known_image = face_recognition.load_image_file(known_img_path)
            known_encoding = face_recognition.face_encodings(known_image)[0]
            with open(os.path.join(self.db_dir, f"{user_name}.pickle"), "wb") as file:
                pickle.dump(known_encoding, file)
            QMessageBox.information(self, "Registration Successful", f"User {user_name} registered successfully!")
            self.registration_window.close()
        else:
            QMessageBox.warning(self, "Registration Failed", "Please enter a valid user name.")

    def closeEvent(self, event):
        self.node.get_logger().info("Shutting down...")
        self.node.destroy_node()
        rclpy.shutdown()
        event.accept()

def main():
    rclpy.init()
    node = rclpy.create_node('gui_node')
    
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    app = QApplication(sys.argv)
    gui = GazeCaptureGUI(node)
    gui.show()

    sys.exit(app.exec_())    

if __name__ == '__main__':
    main()
