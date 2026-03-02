import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openfilter_stub import Filter
import cv2

class Camera(Filter):
    def __init__(self, source=0):
        super().__init__()
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            print(f"Warning: Cannot open video source: {source}")

    def run(self, data):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return {"frame": frame}

    def close(self):
        if hasattr(self, 'cap'):
            self.cap.release()