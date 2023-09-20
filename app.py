import cv2
import numpy as np

class EMAFilter:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.ema_value = None

    def update(self, value):
        value = np.array(value)
        if self.ema_value is None:
            self.ema_value = value
        else:
            self.ema_value = (1 - self.alpha) * self.ema_value + self.alpha * value
        return self.ema_value

def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    # Create an instance of the EMAFilter for each pupil
    left_pupil_filter = EMAFilter()
    right_pupil_filter = EMAFilter()
    
    # Variables for face detection thresholding
    detection_interval = 5
    frame_count = 0
    last_detected_faces = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces only every detection_interval frames
        if frame_count % detection_interval == 0:
            last_detected_faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        faces = last_detected_faces
        frame_count += 1

        black_background = np.zeros_like(frame)

        for (x, y, w, h) in faces:
            center_x = x + w // 2
            center_y = y + h // 2
            left_pupil_pos = (center_x - w // 10, center_y - h // 10)
            right_pupil_pos = (center_x + w // 10, center_y - h // 10)
            
            # Update and get the smoothed positions using EMA
            smoothed_left_pupil = left_pupil_filter.update(left_pupil_pos)
            smoothed_right_pupil = right_pupil_filter.update(right_pupil_pos)
            
            cv2.circle(black_background, tuple(map(int, smoothed_left_pupil)), 10, (255, 255, 255), -1)
            cv2.circle(black_background, tuple(map(int, smoothed_right_pupil)), 10, (255, 255, 255), -1)

        cv2.imshow('Face Tracking', black_background)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("An error occurred:", e)
    finally:
        input("Press Enter to exit...")
