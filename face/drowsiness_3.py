import dlib
import cv2
import numpy as np
from scipy.spatial import distance as dist
import pygame
import threading
from imutils import face_utils

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize Dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Download the model from Dlib's website

# Initialize Dlib's face landmark points for eyes
(l_start, l_end) = face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye']
(r_start, r_end) = face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye']

counter = 0
alarm_on = False
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound('alarm.mp3')
alarm_thread = threading.Thread(target=lambda: None)

def thread_sound():
    if alarm_on:
        channel = alarm_sound.play()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)
    
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        left_eye = shape[l_start:l_end]
        right_eye = shape[r_start:r_end]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        ear = (left_ear + right_ear) / 2.0

        # Draw boxes around the eyes
        left_eye_box = cv2.convexHull(left_eye)
        right_eye_box = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [left_eye_box], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eye_box], -1, (0, 255, 0), 1)


        if ear < 0.23:  # You may need to adjust this threshold based on your specific use case
            counter += 1
            alarm_on = True
            
            if counter >= 20:  # If drowsy for 30 consecutive frames, trigger alarm
                if not alarm_thread.is_alive():
                    alarm_thread = threading.Thread(target=thread_sound)
                    alarm_thread.start()
                    
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            counter = 0
            alarm_on = False
        
        if cv2.waitKey(1) & 0xFF == ord('s'):
            counter = 0
            alarm_on = False
            # alarm_thread = threading.Thread(target=lambda: None)
            # alarm_thread = threading.Thread(target=thread_sound)
            alarm_thread.join()
            break

        cv2.putText(frame, f"EAR: {ear:.2f}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
