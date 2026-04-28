# Fourth filter in the pipeline — draws zone boundaries, bounding boxes, and HUD metrics
# on top of the frame using OpenCV. Pure visualisation, zero business logic.
# Reads data["detections"] and data["metrics"], writes annotated frame back to data["frame"].

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openfilter_stub import Filter
import cv2        # OpenCV drawing functions — polylines, rectangle, putText
import numpy as np  # required to convert polygon point lists to the array format cv2.polylines expects


class Overlay(Filter):
    """
    Visualisation layer — the only filter that modifies the frame.
    Draws on a copy of the frame so the original is never mutated.
    Colours: green = seating zone, blue = barista zone,
             yellow = people, orange = cups, white = HUD text.
    """

    # Store zone coordinates so they can be redrawn on every frame
    def __init__(self, seating_poly, barista_poly):
        super().__init__()
        self.seating_poly = seating_poly
        self.barista_poly = barista_poly

    # Draws all visual elements onto a copy of the frame and writes it back to data["frame"].
    # Uses .copy() so the original frame is never mutated — defensive programming.
    def run(self, data):
        # Draw on a copy — never mutate the original in case another filter reads it downstream
        frame = data["frame"].copy()

        # Draw zone polygon outlines — True closes the shape back to the first point,
        # 2 is line thickness in pixels, colours are in BGR not RGB (OpenCV convention)
        cv2.polylines(frame, [np.array(self.seating_poly, int)], True, (0, 255, 0), 2)   # green seating zone
        cv2.polylines(frame, [np.array(self.barista_poly,  int)], True, (255, 0, 0), 2)  # blue barista zone

        # Zone labels positioned 8px above the first corner of each polygon
        # cv2.putText args: frame, text, (x,y) bottom-left of text, font, scale, colour, thickness
        cv2.putText(frame, "SEATING ZONE",
                    (self.seating_poly[0][0], self.seating_poly[0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(frame, "BARISTA ZONE",
                    (self.barista_poly[0][0], self.barista_poly[0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

        # Draw a bounding box and label for every detection from YoloDetectTrack.
        # data.get() with default [] prevents KeyError if detections key is missing
        for det in data.get("detections", []):
            # map(int, ...) converts float bbox coords to integers — pixel positions must be whole numbers
            x1, y1, x2, y2 = map(int, det["bbox"])
            label = det["label"]

            if label == "person":
                # Yellow in BGR — includes Norfair track ID so you can follow individuals on screen
                color = (0, 255, 255)
                text  = f'person {det["id"]}'
            elif label == "cup":
                # Orange in BGR — visually distinct from yellow person boxes
                color = (0, 165, 255)
                text  = "cup"
            else:
                color = (200, 200, 200)
                text  = label

            # cv2.rectangle draws the box outline — passing -1 as thickness would fill it solid
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Label sits 5px above the top-left corner of the bounding box
            cv2.putText(frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Read metrics written by ZoneLogic — default to 0 if keys are missing
        metrics   = data.get("metrics", {})
        coffees   = metrics.get("coffees_made",     0)
        customers = metrics.get("customers_inside", 0)

        # Plain ASCII only — Windows + OpenCV cannot render Unicode emoji in cv2.putText()
        # Positioned at (20, 35) — 20px from left edge, 35px from top
        hud = f'Coffees: {coffees}  |  Customers in seat: {customers}'
        cv2.putText(frame, hud, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

        # Replace the clean frame with the annotated version — cafe_vision.py reads this for display
        data["frame"] = frame
        return data