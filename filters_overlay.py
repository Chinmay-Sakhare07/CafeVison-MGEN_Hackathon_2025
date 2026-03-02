import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openfilter_stub import Filter
import cv2
import numpy as np

class Overlay(Filter):
    def __init__(self, seating_poly, barista_poly):
        super().__init__()
        self.seating_poly = seating_poly
        self.barista_poly = barista_poly

    def run(self, data):
        frame = data["frame"].copy()
        cv2.polylines(frame, [np.array(self.seating_poly, int)], True, (0,255,0), 2)
        cv2.polylines(frame, [np.array(self.barista_poly, int)], True, (255,0,0), 2)

        for det in data.get("detections", []):
            x1, y1, x2, y2 = map(int, det["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f'{det["label"]} {det["id"]}', (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        metrics = data.get("metrics", {})
        txt = f'☕ Coffees: {metrics.get("coffees_made",0)} | 👥 Customers: {metrics.get("customers_inside",0)}'
        cv2.putText(frame, txt, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        data["frame"] = frame
        return data
