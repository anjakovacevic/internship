import cv2
from mtcnn import MTCNN
import matplotlib.pyplot as plt

detector = MTCNN()

cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Detect faces in the frame
    faces = detector.detect_faces(frame)

    # Draw bounding boxes around detected faces
    for face in faces:
        x, y, width, height = face['box']
        x2, y2 = x + width, y + height
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

    # Display the frame with bounding boxes in real-time
    cv2.imshow('Face Detection', frame)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
