import cv2
import dlib
import numpy as np
from threading import Thread
import imutils
from tkinter import Tk, Button

def sound_alarm():
    # Play an alarm sound here.
    # You can use any library you prefer for this.
    # This is a simple example using standard beep.
    import os
    duration = 1  # seconds
    freq = 440  # Hz
    os.system(f'play -nq -t alsa synth {duration} sine {freq}')

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Threshold for EAR (eye aspect ratio) below which alarm should be triggered
EAR_THRESHOLD = 0.2
# Number of consecutive frames the eye must be below the threshold to set off the alarm
EYE_AR_CONSEC_FRAMES = 20

COUNTER = 0
ALARM_ON = False

print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this pretrained model

# Indexes for the left and right eye in the facial landmarks
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

print("[INFO] Starting video stream...")
vs = cv2.VideoCapture(0)

# GUI to provide "I'm awake" button
root = Tk()
root.title("Drowsiness Detection")

def awake_button_action():
    global COUNTER
    COUNTER = 0

awake_button = Button(root, text="I'm Awake!", command=awake_button_action)
awake_button.pack(pady=20)

while True:
    ret, frame = vs.read()
    frame = imutils.resize(frame, width=720)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(0, 68)])
        
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        
        ear = (leftEAR + rightEAR) / 2.0
        
        left_eye_box = cv2.convexHull(leftEye)
        right_eye_box = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [left_eye_box], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eye_box], -1, (0, 255, 0), 1)
        
        if ear < EAR_THRESHOLD:
            COUNTER += 1
            
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    Thread(target=sound_alarm).start()
        else:
            COUNTER = 0
            ALARM_ON = False
        # Add this code inside the main loop, before displaying the frame
        cv2.putText(frame, f"Drowsiness: {COUNTER}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    
    root.update()
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()
root.destroy()
