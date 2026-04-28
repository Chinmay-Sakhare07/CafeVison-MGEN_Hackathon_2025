# First filter in the pipeline — opens the video source and injects one frame
# per step into the data dict. Supports webcam (source=0) or any video file.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openfilter_stub import Filter
import cv2


class Camera(Filter):
    """
    Wraps cv2.VideoCapture so the rest of the pipeline never needs to know
    where frames come from. Returning None from run() is the pipeline's
    stop signal when the video ends.
    """

    # Opens the video source. source=0 for webcam, file path for video file.
    # cv2.VideoCapture handles all format decoding (mp4, avi, mov) transparently.
    def __init__(self, source=0):
        super().__init__()
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            print(f"Warning: Cannot open video source: {source}")

    # Reads the next frame and injects it into the shared data dict.
    # Frame is a NumPy array of shape (height, width, 3) in BGR colour order.
    # Returns None when video ends — Graph interprets this as stop signal.
    def run(self, data):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return {"frame": frame}

    # Releases the file handle on shutdown.
    # Skipping this locks the file on Windows until Python restarts.
    def close(self):
        if hasattr(self, "cap"):
            self.cap.release()