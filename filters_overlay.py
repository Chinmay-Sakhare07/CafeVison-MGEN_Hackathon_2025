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

        # ── Draw zone polygons ─────────────────────────────────────────────
        # Green = seating zone, Blue = barista/counter zone
        cv2.polylines(frame, [np.array(self.seating_poly, int)], True, (0, 255, 0), 2)
        cv2.polylines(frame, [np.array(self.barista_poly,  int)], True, (255, 0, 0), 2)

        # Zone labels so it's obvious which is which on screen
        cv2.putText(frame, "SEATING ZONE",
                    (self.seating_poly[0][0], self.seating_poly[0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(frame, "BARISTA ZONE",
                    (self.barista_poly[0][0], self.barista_poly[0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

        # ── Draw bounding boxes per detection ──────────────────────────────
        for det in data.get("detections", []):
            x1, y1, x2, y2 = map(int, det["bbox"])
            label = det["label"]

            if label == "person":
                # Yellow box for people, with their Norfair track ID
                color = (0, 255, 255)
                text  = f'person {det["id"]}'
            elif label == "cup":
                # Orange box for cups — distinct from people
                color = (0, 165, 255)
                text  = "cup"
            else:
                color = (200, 200, 200)
                text  = label

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # ── HUD scoreboard — no emojis, plain ASCII only ───────────────────
        metrics = data.get("metrics", {})
        coffees   = metrics.get("coffees_made",     0)
        customers = metrics.get("customers_inside", 0)

        hud = f'Coffees: {coffees}  |  Customers in seat: {customers}'
        cv2.putText(frame, hud, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

        data["frame"] = frame
        return data