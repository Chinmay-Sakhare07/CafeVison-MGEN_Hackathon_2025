import sys
import os
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from filters.filters_camera import Camera
    from filters.filters_yolo_detect_track import YoloDetectTrack
    from filters.filters_zone_logic import ZoneLogic
    from filters.filters_overlay import Overlay
    from filters.filters_reporter import Reporter
    print("Imported from filters package")
except ImportError:
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
#
# These are pixel coordinates drawn on top of your video frame.
# Format: [(left, top), (right, top), (right, bottom), (left, bottom)]
#
# HOW TO FIND THE RIGHT COORDINATES FOR YOUR VIDEO:
#   Run the script once, pause on a frame, and look at the pixel rulers
#   on the OpenCV window. The top-left corner of the window is (0, 0).
#   X increases to the right, Y increases downward.
#
#   Typical 1280x720 video layout example:
#     Seating area  → left half of frame,  mid-height
#     Barista counter → right portion of frame, upper area
#
# Adjust these four polygons to match where things actually are in YOUR video.

# SEATING_POLY = [
#     (50,  200),   # top-left
#     (600, 200),   # top-right
#     (600, 650),   # bottom-right
#     (50,  650),   # bottom-left
# ]

# BARISTA_POLY = [
#     (700, 50),    # top-left
#     (1270, 50),   # top-right
#     (1270, 400),  # bottom-right
#     (700, 400),   # bottom-left
# ]

SEATING_POLY = [(2, 7), (262, 174), (638, 175), (599, 421), (9, 431), (2, 9)]

BARISTA_POLY = [(265, 2), (269, 158), (634, 171), (608, 422), (766, 426), (765, 5), (268, 3)]

# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Cafe Vision - Track customers and coffee making"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video source: 0 for webcam, or path to a video file"
    )
    parser.add_argument(
        "--threshold", type=int, default=50,
        help="Norfair distance threshold for tracking (default 50). "
             "Increase if IDs keep resetting, decrease if nearby people swap IDs."
    )
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    print(f"Video source : {source}")
    print(f"Seating zone : {SEATING_POLY}")
    print(f"Barista zone : {BARISTA_POLY}")

    filters = [
        Camera(source=source),
        YoloDetectTrack(model_path="yolov8n.pt",
                        distance_threshold=args.threshold),
        ZoneLogic(SEATING_POLY, BARISTA_POLY),
        Overlay(SEATING_POLY, BARISTA_POLY),
        Reporter(event_csv="cafe_events.csv"),
    ]

    graph = Graph(filters)
    print("\nStarting Cafe Vision... Press Q to quit.\n")

    frame_count = 0
    try:
        while graph.step():
            output = graph.last_output()
            if "frame" not in output:
                print("End of video.")
                break

            cv2.imshow("Cafe Vision", output["frame"])
            frame_count += 1

            if frame_count % 30 == 0:
                m = output.get("metrics", {})
                print(
                    f"Frame {frame_count:>6} | "
                    f"Customers seated: {m.get('customers_inside', 0)} | "
                    f"Coffees made: {m.get('coffees_made', 0)}"
                )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        graph.close()
        cv2.destroyAllWindows()
        print(f"\nDone. Processed {frame_count} frames.")


if __name__ == "__main__":
    main()