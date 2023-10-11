from mtcnn import MTCNN
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import playsound
import threading

# Function to calculate eye aspect ratio (EAR)
# facial landmark-based metric used to determine the state of the driver's eyes
# A is the Euclidean distance between the vertical eye landmarks 
# B is the Euclidean distance between the horizontal eye landmarks
# C is the Euclidean distance between the leftmost and rightmost eye landmarks
# When the driver's eyes are open, the EAR is relatively high, and when they are closed (or mostly closed), the EAR decreases significantly.

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)

eye_detector = MTCNN()

counter = 0
alarm_on = False
alarm_sound = 'alarm.mp3'

alarm_thread = threading.Thread(target=lambda: None)  # A dummy lambda function

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face, mediapipe expects RGB input
    results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert to RGB

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            face = frame[y:y + h, x:x + w]

            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # Use MTCNN for eye detection
            eye_results = eye_detector.detect_faces(face)
            if eye_results:
                for eye_result in eye_results:
                    x_eye, y_eye, w_eye, h_eye = eye_result['box']
                    eye = gray_face[y_eye:y_eye + h_eye, x_eye:x_eye + w_eye]
                    ear = eye_aspect_ratio(eye)

                    # this parameter is changeable, depends when you want to react on the drowsiness and sleeping
                    if ear < 0.9:
                        counter += 1
                        alarm_on = True
                    
                        if counter >= 30:  # If drowsy for 50 consecutive frames, trigger alarm
                            if not alarm_thread.is_alive():
                                alarm_thread = threading.Thread(target=lambda: playsound.playsound(alarm_sound))
                                alarm_thread.start()
                            cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                else:
                    counter = 0
                    alarm_on = False

                cv2.putText(frame, f"EAR: {ear:.2f}", (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(face, (x_eye, y_eye), (x_eye + w_eye, y_eye + h_eye), (0, 255, 0), 2)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
