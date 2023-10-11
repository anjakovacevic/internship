import cv2
import numpy as np

def get_roi(frame):
    """Detect the face and return the region of interest"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        return frame[y:y+h, x:x+w]
    return None

def extract_signal(roi):
    """Extract mean values of RGB channels"""
    r = np.mean(roi[:,:,2])
    g = np.mean(roi[:,:,1])
    b = np.mean(roi[:,:,0])
    return r, g, b

def process_signal(r, g, b):
    """Calculate the rPPG signal"""
    rPPG = r - g
    return rPPG

def get_heart_rate(rPPG_signal, fps):
    """Calculate heart rate from rPPG signal"""
    # Fourier transform to get the frequency domain.
    fft = np.fft.fft(rPPG_signal)
    frequencies = np.fft.fftfreq(len(fft), 1.0/fps)
    # Filter the frequencies in the typical human heart rate range: 0.67 to 4 Hz (40-240bpm)
    fft[frequencies <= 0.67] = 0
    fft[frequencies >= 4] = 0
    # Get the peak frequency which represents the heart rate
    peak_frequency = frequencies[np.argmax(np.abs(fft))]
    heart_rate = peak_frequency * 60  # Convert from Hz to bpm
    return heart_rate

def main():
    cap = cv2.VideoCapture(0)
    rPPG_signal = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        roi = get_roi(frame)
        if roi is not None:
            r, g, b = extract_signal(roi)
            rPPG = process_signal(r, g, b)
            rPPG_signal.append(rPPG)

            if len(rPPG_signal) >= fps*5:  # Use last 5 seconds for heart rate estimation
                heart_rate = get_heart_rate(rPPG_signal[-fps*5:], fps)
                print(f"Heart Rate: {heart_rate:.2f} bpm")
        
        cv2.imshow('ROI', roi)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
