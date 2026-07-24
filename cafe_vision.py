# Main entry point - assembles the 5-filter pipeline and runs the frame processing loop.
# Run with: python cafe_vision.py --source 0 (webcam) or --source v1.mp4 (video file)

import sys
import os
import argparse

# Add current and parent directories to path so imports work regardless of where script is run from
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Try package import first, fall back to direct import if filters sit in the same directory

from filters_camera import Camera
from filters_yolo_detect_track import YoloDetectTrack
from filters_zone_logic import ZoneLogic
from filters_overlay import Overlay
from filters_reporter import Reporter
print("Imported directly")

from openfilter_stub import Graph
import cv2
import numpy as np


# ── ZONE CONFIGURATION ────────────────────────────────────────────────────────
# Pixel coordinates defining the seating and barista zones on the video frame.
# Format: [(x, y), ...] where (0, 0) is top-left, x increases right, y increases down.
# Use zone_helper.py to click zone corners on a paused frame and get exact coordinates.

# these are for webacm where resolution is 640*480
# SEATING_POLY = [
#     (0,   0),     # top-left
#     (480, 0),     # top-right (3/4 of 640)
#     (480, 480),   # bottom-right
#     (0,   480),   # bottom-left
# ]

# BARISTA_POLY = [
#     (480, 0),     # top-left (starts at 3/4 mark)
#     (640, 0),     # top-right (edge of frame)
#     (640, 480),   # bottom-right
#     (480, 480),   # bottom-left
# ]


#  #v1
SEATING_POLY = [(2, 7), (262, 174), (638, 175), (599, 421), (9, 431), (2, 9)]
BARISTA_POLY = [(265, 2), (269, 158), (634, 171), (608, 422), (766, 426), (765, 5), (268, 3)]

# ─────────────────────────────────────────────────────────────────────────────


def main():
    """
    Parses arguments, builds the filter pipeline, and runs the main frame loop.
    Handles keyboard interrupt and unexpected exceptions with clean shutdown via finally.
    """

    # --source accepts 0 for webcam or a file path - --threshold tunes Norfair's ID matching strictness
    parser = argparse.ArgumentParser(description="Cafe Vision - Track customers and coffee making")
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video source: 0 for webcam, or path to a video file"
    )
    parser.add_argument(
        "--threshold", type=int, default=50,
        help="Norfair distance threshold for tracking"
    )
    args = parser.parse_args()

    # isdigit() distinguishes webcam index "0" from a file path like "v1.mp4"
    source = int(args.source) if args.source.isdigit() else args.source
    print(f"Video source : {source}")
    print(f"Seating zone : {SEATING_POLY}")
    print(f"Barista zone : {BARISTA_POLY}")

    # Assemble pipeline in order - each filter assumes previous filters have populated their keys in data
    filters = [
        Camera(source=source),
        YoloDetectTrack(model_path="yolov8n.pt", distance_threshold=args.threshold),
        ZoneLogic(SEATING_POLY, BARISTA_POLY),
        Overlay(SEATING_POLY, BARISTA_POLY),
        Reporter(event_csv="cafe_events.csv"),
    ]

    graph = Graph(filters)
    print("\nStarting Cafe Vision... Press Q to quit.\n")

    frame_count = 0
    try:
        # graph.step() runs one full frame through all 5 filters, returns False when video ends
        while graph.step():
            output = graph.last_output()
            if "frame" not in output:
                print("End of video.")
                break

            # cv2.imshow updates the window every iteration - this is what makes it look like live video
            cv2.imshow("Cafe Vision", output["frame"])
            frame_count += 1

            # Print metrics every 30 frames (~1 second) to avoid spamming the terminal
            if frame_count % 30 == 0:
                m = output.get("metrics", {})
                print(
                    f"Frame {frame_count:>6} | "
                    f"Customers seated: {m.get('customers_inside', 0)} | "
                    f"Coffees made: {m.get('coffees_made', 0)}"
                )

            # cv2.waitKey(1) both listens for keypresses and processes the OpenCV window event queue
            # without this the window freezes - & 0xFF masks to 8 bits for cross-platform compatibility
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")

    except Exception as e:
        # Full traceback pinpoints exactly which filter and line caused the crash
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()

    finally:
        # finally always runs - guarantees cleanup even if an exception occurred mid-loop
        graph.close()
        cv2.destroyAllWindows()
        print(f"\nDone. Processed {frame_count} frames.")


# Prevents main() from running if this file is imported as a module elsewhere
if __name__ == "__main__":
    main()