import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openfilter_stub import Filter
from shapely.geometry import Polygon, Point
import time

class ZoneLogic(Filter):
    def __init__(self, seating_poly, barista_poly):
        super().__init__()
        self.seating_zone = Polygon(seating_poly)
        self.barista_zone = Polygon(barista_poly)
        self.customer_state = {}
        self.coffee_counter = 0

    def run(self, data):
        detections = data.get("detections", [])
        now = time.time()

        for det in detections:
            if det["label"] != "person":
                continue
            center = Point((det["bbox"][0] + det["bbox"][2]) / 2,
                           (det["bbox"][1] + det["bbox"][3]) / 2)
            tid = det["id"]

            # customer in seating zone
            if self.seating_zone.contains(center):
                if tid not in self.customer_state:
                    self.customer_state[tid] = now
            else:
                if tid in self.customer_state:
                    stay = now - self.customer_state.pop(tid)
                    data.setdefault("events", []).append({
                        "type": "customer_exit",
                        "id": tid,
                        "duration_sec": round(stay, 2)
                    })

            # barista zone → coffee made
            if self.barista_zone.contains(center):
                data.setdefault("events", []).append({
                    "type": "coffee_made",
                    "timestamp": now
                })
                self.coffee_counter += 1

        data["metrics"] = {
            "customers_inside": len(self.customer_state),
            "coffees_made": self.coffee_counter
        }
        return data
