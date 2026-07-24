☕ Cafe Vision – Plainsight × OpenFilter Hackathon 2025

Real-time Productivity & Customer Flow Tracker for Coffee Shops
Built during the Plainsight × OpenFilter Computer Vision Hackathon.

🎯 Overview

Cafe Vision turns any webcam or CCTV feed into a live analytics tool for café operators. It uses a five-stage computer vision pipeline to track:

👥 Seat occupancy — live count of customers currently seated
☕ Coffee production — how many coffees have been made per session
📊 Event logging — timestamped CSV export of all coffee events

All processing happens in real time, with no cloud dependency.

🧠 Pipeline Architecture

Each stage is an independent filter that reads from and writes to a shared data dictionary. No filter knows what comes before or after it — this keeps the pipeline loosely coupled and easy to extend.

Camera → YoloDetectTrack → ZoneLogic → Overlay → Reporter
Stage	Responsibility
Camera	Opens the video source (webcam or file) and injects one frame per step
YoloDetectTrack	Detects people and cups with YOLOv8 nano; tracks people across frames via Norfair (Kalman filter)
ZoneLogic	Tests detections against zone polygons using Shapely; counts seated customers and gates coffee events
Overlay	Draws zone outlines, bounding boxes, and a live HUD onto the frame
Reporter	Appends coffee events to a CSV file; skips disk I/O on frames with no events
⚙️ Tech Stack
Layer	Tools
Computer Vision	OpenCV, YOLOv8 nano (Ultralytics)
Object Tracking	Norfair (Kalman filter, Hungarian matching)
Zone Logic	Shapely (point-in-polygon)
ML Inference	PyTorch (CPU build)
Reporting	Pandas, CSV
Pipeline Framework	OpenFilter stub (compatible with Plainsight OpenFilter)
🧩 Project Structure
Cafe_Vision/
│
├── cafe_vision.py                  # Main entry point — assembles and runs the pipeline
├── openfilter_stub.py              # Minimal OpenFilter-compatible Filter and Graph interface
├── filters_camera.py               # Stage 1: video/webcam frame source
├── filters_yolo_detect_track.py    # Stage 2: YOLOv8 detection + Norfair tracking
├── filters_zone_logic.py           # Stage 3: zone containment, occupancy count, coffee events
├── filters_overlay.py              # Stage 4: bounding boxes, zone outlines, HUD
├── filters_reporter.py             # Stage 5: CSV event logger
├── zone_helper.py                  # Interactive tool for clicking zone coordinates on a frame
├── cafe_events.csv                 # Sample output log
├── yolov8n.pt                      # YOLOv8 nano weights
└── plainsight.mp4                  # Demo input video
🧰 Installation & Setup
1. Create Virtual Environment
bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
2. Install Dependencies
bash
pip install -r requirements.txt

Or manually:

bash
pip install numpy opencv-python torch torchvision ultralytics shapely pandas norfair
🎥 Calibrating Zones for Your Camera

Zone coordinates are hardcoded in cafe_vision.py and are calibrated for the included v1.mp4 demo video. If you use a different camera or video source, you must recalibrate the zones.

Use the interactive zone helper:

bash
python zone_helper.py --source your_video.mp4
Left click to add a corner point
S to save the current zone and move to the next
R to reset the current zone
Q to quit and print the coordinates

Copy the printed coordinates into cafe_vision.py to replace SEATING_POLY and BARISTA_POLY.

▶️ Usage
Run with video file
bash
python cafe_vision.py --source plainsight.mp4
Run with webcam
bash
python cafe_vision.py --source 0
Optional: tune tracking sensitivity
bash
python cafe_vision.py --source v1.mp4 --threshold 50

--threshold controls Norfair's distance threshold for matching detections to existing tracks. Lower = stricter matching; raise it if IDs flicker frequently.

Output
Annotated video window with live HUD
Console metrics printed every 30 frames (~1 second)
cafe_events.csv — append-only log of coffee events with timestamps
⚠️ Known Limitations

Cup-based coffee counting is approximate. A cup detected inside the barista zone triggers a coffee event. A 5-second cooldown prevents rapid-fire false positives, but a cup left sitting on the counter between orders can still trigger spurious events over time.

Zone coordinates are video-specific. The bundled coordinates are calibrated for v1.mp4. Any different camera angle or resolution requires recalibration via zone_helper.py.

Customer dwell time is not yet tracked. Entry/exit timestamps and per-customer duration are planned but not implemented in the current version. The duration_sec CSV column is reserved for this feature.

🧪 Hackathon Highlights
Criteria	How We Met It
Creativity	Dual tracking of customer occupancy and barista output from a single video feed
Functionality	Live feed → real-time metrics → CSV export
Technical Implementation	YOLOv8 + Norfair tracking + Shapely zone logic in a composable filter pipeline
Problem Fit	Applicable to any retail or hospitality setup with an existing camera
Presentation	Demo video + annotated output screenshots
🧾 Planned Improvements
Customer dwell time tracking (entry/exit timestamps per ID)
Live dashboard with real-time graphs (Plotly / Streamlit)
Queue length and wait-time estimation
Agentic alerting layer — automated notifications when occupancy is high or coffee output drops
Cloud export (AWS S3 + Lambda processing)
👥 Team

Chinmay Sakhare · Agnel Salve

🏁 Acknowledgments
Plainsight AI — OpenFilter framework & hackathon support
Ultralytics YOLOv8 — object detection
Norfair — lightweight object tracking
OpenCV — real-time video processing
Shapely — geometric zone logic

"Turning every cup of coffee into data."