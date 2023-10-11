import cv2

# Replace this with the IP address and port from your phone's camera server
camera_url = 'http://10.10.104.56:8080'

cap = cv2.VideoCapture(camera_url + '/video')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame as needed (e.g., display it)
    cv2.imshow('Phone Camera Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
