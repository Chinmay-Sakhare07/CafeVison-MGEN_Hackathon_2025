"""
Zone Coordinate Helper
──────────────────────
Run this script, click the corners of your zones on the video frame,
and it will print the exact coordinates to paste into cafe_vision.py.

Usage:
    python zone_helper.py --source v1.mp4

Controls:
    Left click  → add a point to the current zone
    R           → reset current zone points
    S           → save current zone and move to next zone
    Q           → quit and print all coordinates
"""

import cv2
import numpy as np
import argparse

points = []
zones = []
zone_names = ["SEATING_POLY", "BARISTA_POLY"]
zone_colors = [(0, 255, 0), (255, 0, 0)]
current_zone = 0
frame_display = None


def mouse_click(event, x, y, flags, param):
    global points, frame_display
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"  Point added: ({x}, {y})")

        # Draw the point on screen
        cv2.circle(frame_display, (x, y), 5, zone_colors[current_zone], -1)
        if len(points) > 1:
            cv2.line(frame_display,
                     points[-2], points[-1],
                     zone_colors[current_zone], 2)
        cv2.imshow("Zone Helper", frame_display)


def main():
    global points, current_zone, frame_display

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Cannot open: {args.source}")
        return

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Could not read frame from video.")
        return

    h, w = frame.shape[:2]
    print(f"\nVideo resolution: {w} x {h}")
    print("─" * 50)

    cv2.namedWindow("Zone Helper")
    cv2.setMouseCallback("Zone Helper", mouse_click)

    while current_zone < len(zone_names):
        frame_display = frame.copy()

        # Draw already-saved zones
        for i, z in enumerate(zones):
            cv2.polylines(frame_display,
                          [np.array(z, int)], True,
                          zone_colors[i], 2)
            cv2.putText(frame_display, zone_names[i],
                        (z[0][0], z[0][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        zone_colors[i], 2)

        # Draw current in-progress points
        for p in points:
            cv2.circle(frame_display, p, 5,
                       zone_colors[current_zone], -1)
        if len(points) > 1:
            for i in range(1, len(points)):
                cv2.line(frame_display, points[i-1], points[i],
                         zone_colors[current_zone], 2)

        instruction = (
            f"Defining: {zone_names[current_zone]}  |  "
            f"Click corners  |  S=save zone  |  R=reset  |  Q=quit"
        )
        cv2.putText(frame_display, instruction, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        print(f"\nNow defining: {zone_names[current_zone]}")
        print("Click the corners of this zone on the image.")
        print("Press S when done, R to reset, Q to quit.")

        cv2.imshow("Zone Helper", frame_display)

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == ord("s"):
                if len(points) < 3:
                    print("Need at least 3 points to define a zone.")
                    continue
                zones.append(points.copy())
                print(f"\n{zone_names[current_zone]} saved: {points}")
                points = []
                current_zone += 1
                break

            elif key == ord("r"):
                points = []
                frame_display = frame.copy()
                for i, z in enumerate(zones):
                    cv2.polylines(frame_display,
                                  [np.array(z, int)], True,
                                  zone_colors[i], 2)
                print("Points reset.")
                cv2.imshow("Zone Helper", frame_display)

            elif key == ord("q"):
                current_zone = len(zone_names)  # exit loop
                break

    cv2.destroyAllWindows()

    # ── Print the final coordinates ────────────────────────────────────────
    print("\n" + "=" * 50)
    print("COPY THESE INTO cafe_vision.py")
    print("=" * 50)
    for i, z in enumerate(zones):
        print(f"\n{zone_names[i]} = {z}")
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()