import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from norfair import Tracker, Detection
from openfilter_stub import Filter
from ultralytics import YOLO

# Labels we care about — people for occupancy, cups for coffee detection
TRACKED_LABELS = {"person", "cup"}


class YoloDetectTrack(Filter):
    def __init__(self, model_path="yolov8n.pt", distance_threshold=50):
        super().__init__()
        self.model = YOLO(model_path)

        # Norfair tracker for people only — cups don't need persistent IDs
        # because we just care whether a cup exists in the zone, not which
        # specific cup it is across frames.
        self.tracker = Tracker(
            distance_function="euclidean",
            distance_threshold=distance_threshold,
            hit_counter_max=15,
            initialization_delay=2,
        )

        # Cache last known bounding box per Norfair track ID
        self._last_bbox = {}

    def run(self, data):
        frame = data["frame"]

        # ── Step 1: YOLOv8 detection only ─────────────────────────────────
        results = self.model(frame, verbose=False)

        # ── Step 2: Separate people (tracked) from cups (pass-through) ────
        norfair_detections = []
        raw_info = {}       # (cx, cy) → info dict, for people only
        cup_detections = [] # cups go straight through, no tracking needed

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label  = self.model.names[cls_id]
                conf   = float(box.conf[0])

                if label not in TRACKED_LABELS:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                if label == "person":
                    det = Detection(
                        points=np.array([[cx, cy]]),
                        scores=np.array([conf])
                    )
                    norfair_detections.append(det)
                    raw_info[(round(cx, 1), round(cy, 1))] = {
                        "label": label,
                        "conf":  conf,
                        "bbox":  [x1, y1, x2, y2]
                    }

                elif label == "cup":
                    # Cups pass straight through with a static ID of -1
                    # Zone logic only needs to know a cup is present
                    cup_detections.append({
                        "id":    -1,
                        "label": "cup",
                        "conf":  conf,
                        "bbox":  [x1, y1, x2, y2]
                    })

        # ── Step 3: Norfair tracks people across frames ────────────────────
        tracked_objects = self.tracker.update(detections=norfair_detections)

        # ── Step 4: Build people detections with stable Norfair IDs ────────
        person_detections = []
        for obj in tracked_objects:
            tid = obj.id
            cx, cy = float(obj.estimate[0][0]), float(obj.estimate[0][1])

            best_info = None
            best_dist = float("inf")
            for (rcx, rcy), info in raw_info.items():
                dist = ((rcx - cx) ** 2 + (rcy - cy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_info = info

            if best_info and best_dist < 80:
                bbox = best_info["bbox"]
                conf = best_info["conf"]
                self._last_bbox[tid] = bbox
            elif tid in self._last_bbox:
                old = self._last_bbox[tid]
                w = old[2] - old[0]
                h = old[3] - old[1]
                bbox = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
                conf = 0.0
            else:
                bbox = [cx - 40, cy - 90, cx + 40, cy + 90]
                conf = 0.0

            person_detections.append({
                "id":    tid,
                "label": "person",
                "conf":  conf,
                "bbox":  bbox
            })

        # ── Step 5: Combine people + cups for downstream filters ───────────
        data["detections"] = person_detections + cup_detections
        return data