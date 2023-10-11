import face_recognition
import numpy as np
import cv2

videocapture=cv2.VideoCapture(0)
while True:
    ret, frame = videocapture.read()
    #rgb_frame = frame[:,:,::-1]
    # print(frame)
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    #print(face_locations, len(face_locations))

    #print(face_encodings, len(face_encodings))
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videocapture.release()
cv2.destroyAllWindows()
