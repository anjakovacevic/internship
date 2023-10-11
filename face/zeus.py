import cv2
import mediapipe as mp
from datetime import datetime

# Initialize MediaPipe components
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Initialize variables for frame rate calculation
start_time = datetime.now()
frame_count = 0

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

# Initialize the Face Mesh model
with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    
    while cap.isOpened():
        # Capture a frame from the webcam
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

        # Calculate frame rate
        frame_count += 1
        elapsed_time = (datetime.now() - start_time).total_seconds()
        frame_rate = frame_count / elapsed_time

        # Display frame rate in the console
        print(f"Frame rate: {frame_rate:.2f} fps")

        # Display the processed image with face mesh annotations
        cv2.imshow('MediaPipe FaceMesh', image)
        
        # Check for the 'Esc' key press to exit the application
        if cv2.waitKey(5) & 0xFF == 27:
            break

        # Check for user keypress events
        key = cv2.waitKey(10)

        # If the 'q' key is pressed, exit the application
        if key == ord("q"):
            break

        # If the 'c' key is pressed, capture a screenshot with a timestamp
        if key == ord("c"):
            now = datetime.now()
            date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
            print(date_time)
            cv2.imwrite("{}.png".format(date_time), image)

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()