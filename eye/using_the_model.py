import torch
import cv2
from ITrackerModel import ITrackerModel  # Replace with the actual import path of your model definition

# Instantiate the model
model = ITrackerModel()

# Load pre-trained weights (if available)
checkpoint_path = r'C:\Users\anja.kovacevic\A\checkpoint.pth'  # Replace with the path to your checkpoint file

if torch.cuda.is_available():
    checkpoint = torch.load(checkpoint_path)
else:
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Load the state dict
model.load_state_dict(checkpoint['state_dict'])

# Set the model in evaluation mode (important for models with batch normalization and dropout)
model.eval()

# Open the camera (usually camera index 0 is the built-in camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Preprocess the frame (resize, normalize, etc.)
    # You need to implement preprocessing according to the model's requirements

    # Convert the preprocessed frame to a PyTorch tensor
    input_tensor = torch.tensor(frame)  # You may need to adjust this based on your preprocessing

    # Perform inference using the model
    with torch.no_grad():
        output = model(input_tensor)

    # Extract gaze estimation from the model's output
    gaze_x, gaze_y = output[0], output[1]

    # Overlay the estimated gaze point on the frame (you may need to adjust the coordinates)
    cv2.circle(frame, (int(gaze_x), int(gaze_y)), 5, (0, 0, 255), -1)

    # Display the frame with the gaze point
    cv2.imshow('Gaze Estimation', frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()