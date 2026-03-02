import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openfilter_stub import Filter
from ultralytics import YOLO

class YoloDetectTrack(Filter):
    def __init__(self, model_path="yolov8n.pt"):
        super().__init__()
        self.model = YOLO(model_path)

    def run(self, data):
        frame = data["frame"]
        results = self.model.track(frame, persist=True, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.model.names[cls_id]
                xyxy = box.xyxy[0].tolist()
                tid = int(box.id[0]) if box.id is not None else -1
                detections.append({
                    "id": tid, "label": label, "conf": conf, "bbox": xyxy
                })
        data["detections"] = detections
        return data
