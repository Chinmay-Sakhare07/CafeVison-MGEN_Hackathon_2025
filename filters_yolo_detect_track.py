# Second filter in the pipeline - detects people and cups with YOLOv8, then
# tracks people across frames using Norfair's Kalman filter-based tracker.
# Cups pass through untracked since zone logic only needs their presence, not identity.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from norfair import Tracker, Detection
from openfilter_stub import Filter
from ultralytics import YOLO

TRACKED_LABELS = {"person", "cup"}


class YoloDetectTrack(Filter):
    """
    Runs YOLOv8 detection on every frame then hands person detections to Norfair
    for persistent ID assignment. Cups skip tracking entirely and pass straight through.
    Downstream filters receive a unified detections list with stable IDs for people
    and id=-1 for cups.
    """

    # Loads YOLOv8 nano model and initialises the Norfair tracker with Kalman filters.
    # distance_threshold controls how far a detection can be from a track and still match
    def __init__(self, model_path="yolov8n.pt", distance_threshold=50):
        super().__init__()

        self.model = YOLO(model_path)
        # Norfair tracker - euclidean distance matching, Kalman filter prediction,
        # keeps tracks alive for 15 frames without a detection (~0.5s at 30fps)
        self.tracker = Tracker(
            distance_function="euclidean",
            distance_threshold=distance_threshold,
            hit_counter_max=15,       # frames to keep a track alive with no detection
            initialization_delay=2,   # frames before a new track gets a confirmed ID
        )
        # Stores last known bbox per track ID so we can reconstruct it when detector misses a person
        self._last_bbox = {}

    # Runs detection and tracking on the current frame, writes results to data["detections"].
    # Detections list contains dicts with keys: id, label, conf, bbox ([x1,y1,x2,y2]).
    def run(self, data):
        frame = data["frame"]

        # Step 1 - YOLOv8 detection only, no built-in tracking
        # verbose=False suppresses per-frame console output at 30 lines/second
        results = self.model(frame, verbose=False)

        # Step 2 - Parse detections into people (need tracking) and cups (pass-through)
        norfair_detections = []
        raw_info = {}       # maps (cx, cy) → full bbox info so we can recover it after Norfair assigns IDs
        cup_detections = [] # cups go straight to output, zone logic only needs presence not identity

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label  = self.model.names[cls_id]  # e.g. 0 → "person", 41 → "cup"
                conf   = float(box.conf[0])        # confidence score 0.0–1.0

                if label not in TRACKED_LABELS:
                    continue

                # xyxy gives top-left and bottom-right corners - tolist() converts tensor to Python list
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # Center point is what Norfair tracks - more stable than corners across frames
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                if label == "person":
                    # Wrap center point in Norfair's Detection format for the tracker
                    det = Detection(
                        points=np.array([[cx, cy]]),
                        scores=np.array([conf])
                    )
                    norfair_detections.append(det)
                    # Store full box keyed by center so we can recover it after tracking
                    raw_info[(round(cx, 1), round(cy, 1))] = {
                        "label": label,
                        "conf":  conf,
                        "bbox":  [x1, y1, x2, y2]
                    }

                elif label == "cup":
                    # id=-1 signals no persistent tracking - zone logic only needs to know a cup is present
                    cup_detections.append({
                        "id":    -1,
                        "label": "cup",
                        "conf":  conf,
                        "bbox":  [x1, y1, x2, y2]
                    })

        # Step 3 - Norfair predicts positions via Kalman filter, matches detections to tracks
        # via Hungarian algorithm, and returns tracked objects with stable IDs
        tracked_objects = self.tracker.update(detections=norfair_detections)

        # Step 4 - Rebuild full detections with Norfair IDs and real bounding boxes
        # Norfair only returns center points so we match back to raw YOLOv8 boxes
        person_detections = []
        for obj in tracked_objects:
            tid = obj.id
            # obj.estimate is Norfair's Kalman-smoothed position for this track
            cx, cy = float(obj.estimate[0][0]), float(obj.estimate[0][1])

            # Find the closest raw detection to recover the real bounding box
            best_info = None
            best_dist = float("inf")
            for (rcx, rcy), info in raw_info.items():
                dist = ((rcx - cx) ** 2 + (rcy - cy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_info = info

            if best_info and best_dist < 80:
                # Case 1: real detection found nearby - use exact YOLOv8 bbox and cache it
                bbox = best_info["bbox"]
                conf = best_info["conf"]
                self._last_bbox[tid] = bbox
            elif tid in self._last_bbox:
                # Case 2: detector missed this frame but Norfair kept the track alive -
                # reconstruct box at predicted center using last known width and height
                old = self._last_bbox[tid]
                w = old[2] - old[0]
                h = old[3] - old[1]
                bbox = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
                conf = 0.0  # 0.0 flags this as a predicted position, not a live detection
            else:
                # Case 3: brand new track with no cached size - rough fixed estimate until real box arrives
                bbox = [cx - 40, cy - 90, cx + 40, cy + 90]
                conf = 0.0

            person_detections.append({
                "id":    tid,
                "label": "person",
                "conf":  conf,
                "bbox":  bbox
            })

        # Step 5 - Merge tracked people and pass-through cups into one list for downstream filters
        data["detections"] = person_detections + cup_detections
        return data