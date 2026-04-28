# Third filter in the pipeline - reads detections from YoloDetectTrack and answers
# two questions: how many customers are seated right now, and how many coffees have been made.
# Uses Shapely for point-in-polygon zone testing and a rolling average to smooth the count.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openfilter_stub import Filter
from shapely.geometry import Polygon, Point  # geometry library for zone containment testing
from collections import deque               # fixed-size rolling window, auto-drops oldest value
import time                                 # Unix timestamps for cooldown tracking

# Seconds between coffee events - prevents one cup sitting on the counter
COFFEE_COOLDOWN_SEC = 5.0

# Frames to average the customer count over - 15 frames ≈ 0.5s at 30fps.
# Short enough to respond when someone sits or leaves, long enough to
# absorb single-frame detection misses without the number flickering
SMOOTHING_FRAMES = 15


class ZoneLogic(Filter):
    """
    Analytics brain of the pipeline. Converts raw detections into meaningful metrics.
    People inside the seating polygon are counted as seated customers.
    Cups inside the barista polygon trigger a coffee event with a cooldown gate.
    Customer count is smoothed over the last N frames to eliminate flicker.
    """

    # Builds Shapely polygons from the coordinate lists passed in from cafe_vision.py.
    # Initialises the rolling window, coffee counter, and cooldown timer.
    def __init__(self, seating_poly, barista_poly):
        super().__init__()
        # Shapely Polygon objects - used for point-in-polygon containment testing
        self.seating_zone = Polygon(seating_poly)
        self.barista_zone = Polygon(barista_poly)
        # Rolling window of raw per-frame counts - maxlen auto-drops the oldest when full
        self._count_history = deque(maxlen=SMOOTHING_FRAMES)
        # Cumulative total of coffees counted since the pipeline started
        self.coffee_counter = 0
        # Timestamp of last coffee event - initialised to 0 so the first coffee is always countable
        self._last_coffee_time = 0.0

    # Counts seated customers and detects coffees for the current frame.
    # Writes smoothed customer count and coffee total to data["metrics"].
    # Appends a coffee_made event to data["events"] when cooldown allows.
    def run(self, data):
        # data.get() with default [] prevents KeyError if no detections exist this frame
        detections = data.get("detections", [])
        # Capture time once and reuse - avoids tiny drift from multiple time.time() calls
        now = time.time()

        # Step 1 - Live count: how many person centers are inside the seating zone right now.
        # Deliberately not ID-based - ID tracking caused counts to climb to 28 from 8 real people
        # because every Norfair ID reassignment looked like a new customer entry.
        customers_now = 0
        for det in detections:
            label = det["label"]
            # Bounding box center point - this is what we test against the zone polygon
            center = Point(
                (det["bbox"][0] + det["bbox"][2]) / 2,   # cx = average of left and right edges
                (det["bbox"][1] + det["bbox"][3]) / 2    # cy = average of top and bottom edges
            )

            if label == "person":
                # Shapely's contains() uses ray casting algorithm - returns True if center
                # is strictly inside the polygon boundary
                if self.seating_zone.contains(center):
                    customers_now += 1

            elif label == "cup":
                if self.barista_zone.contains(center):
                    # Cooldown gate - (now - last_time) gives seconds elapsed since last coffee
                    # Without this, one cup on the counter = 30 coffee events per second
                    if (now - self._last_coffee_time) >= COFFEE_COOLDOWN_SEC:
                        self.coffee_counter += 1
                        self._last_coffee_time = now
                        # setdefault creates the events key with [] if absent, then appends -
                        # avoids a KeyError when the first event of the run is logged
                        data.setdefault("events", []).append({
                            "type":      "coffee_made",
                            "timestamp": now
                        })

        # Step 2 - Push this frame's raw count into the rolling window.
        # deque automatically drops the oldest entry once maxlen is reached.
        self._count_history.append(customers_now)

        # Step 3 - Smooth the count by averaging the rolling window.
        # Absorbs single-frame detection misses - e.g. [6,6,6,1,6,6,6] → 5.9 → rounds to 6
        # len() used instead of SMOOTHING_FRAMES to handle startup where window isn't full yet
        smoothed = round(sum(self._count_history) / len(self._count_history))

        # Write metrics to data dict - Overlay reads this to render the HUD scoreboard
        data["metrics"] = {
            "customers_inside": smoothed,
            "coffees_made":     self.coffee_counter
        }
        return data