import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openfilter_stub import Filter
from shapely.geometry import Polygon, Point
from collections import deque
import time

# How many seconds must pass before a cup in the barista zone
# triggers another "coffee made" event.
COFFEE_COOLDOWN_SEC = 5.0

# How many recent frames to average the customer count over.
# 15 frames = ~0.5 seconds of smoothing at 30fps.
# Eliminates single-frame flickers without making the number feel laggy.
SMOOTHING_FRAMES = 15


class ZoneLogic(Filter):
    def __init__(self, seating_poly, barista_poly):
        super().__init__()
        self.seating_zone = Polygon(seating_poly)
        self.barista_zone = Polygon(barista_poly)

        # Rolling window — stores the raw customer count from each
        # of the last SMOOTHING_FRAMES frames. deque automatically
        # drops the oldest value when a new one is added.
        self._count_history = deque(maxlen=SMOOTHING_FRAMES)

        # Coffee counter and cooldown timer
        self.coffee_counter = 0
        self._last_coffee_time = 0.0

    def run(self, data):
        detections = data.get("detections", [])
        now = time.time()

        # ── Step 1: Raw live count for this frame ──────────────────────────
        customers_now = 0
        for det in detections:
            label = det["label"]
            center = Point(
                (det["bbox"][0] + det["bbox"][2]) / 2,
                (det["bbox"][1] + det["bbox"][3]) / 2
            )

            if label == "person":
                if self.seating_zone.contains(center):
                    customers_now += 1

            # ── Cups → coffee detection with cooldown ──────────────────────
            elif label == "cup":
                if self.barista_zone.contains(center):
                    if (now - self._last_coffee_time) >= COFFEE_COOLDOWN_SEC:
                        self.coffee_counter += 1
                        self._last_coffee_time = now
                        data.setdefault("events", []).append({
                            "type":      "coffee_made",
                            "timestamp": now
                        })

        # ── Step 2: Add this frame's count to the rolling window ──────────
        self._count_history.append(customers_now)

        # ── Step 3: Smoothed count = average of the last N frames ─────────
        # round() gives a clean integer — e.g. 6.8 → 7, not 6.8
        smoothed = round(sum(self._count_history) / len(self._count_history))

        data["metrics"] = {
            "customers_inside": smoothed,
            "coffees_made":     self.coffee_counter
        }
        return data