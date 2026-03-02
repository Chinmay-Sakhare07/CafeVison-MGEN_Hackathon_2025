import sys
import os
import argparse

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Add parent directory to Python path (if needed)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Debug: Print what Python sees
print(f"Current directory: {current_dir}")
print(f"Looking for filters in: {os.path.join(current_dir, 'filters')}")
print(f"Filters folder exists: {os.path.exists(os.path.join(current_dir, 'filters'))}")

# Try different import methods based on what exists
try:
    # Method 1: Import from filters package
    from filters.filters_camera import Camera
    from filters.filters_yolo_detect_track import YoloDetectTrack
    from filters.filters_zone_logic import ZoneLogic
    from filters.filters_overlay import Overlay
    from filters.filters_reporter import Reporter
    print("Successfully imported from filters package")
except ImportError as e:
    print(f"Package import failed: {e}")
    # Method 2: Try direct import if files are in same directory
    try:
        from filters_camera import Camera
        from filters_yolo_detect_track import YoloDetectTrack
        from filters_zone_logic import ZoneLogic
        from filters_overlay import Overlay
        from filters_reporter import Reporter
        print("Successfully imported directly")
    except ImportError as e2:
        print(f"Direct import also failed: {e2}")
        raise

from openfilter_stub import Graph
import cv2
import numpy as np

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Cafe Vision - Track customers and coffee making')
    parser.add_argument('--source', type=str, default='0', 
                       help='Video source (0 for webcam, or path to video file)')
    args = parser.parse_args()
    
    # Convert source to int if it's a webcam index
    source = args.source
    if source.isdigit():
        source = int(source)
    
    print(f"Using video source: {source}")
    
    # Define zone polygons (adjust these based on your camera view)
    seating_poly = [(100, 200), (400, 200), (400, 400), (100, 400)]
    barista_poly = [(450, 100), (640, 100), (640, 300), (450, 300)]
    
    # Create filter pipeline
    print("Initializing filters...")
    filters = [
        Camera(source=source),
        YoloDetectTrack(model_path="yolov8n.pt"),
        ZoneLogic(seating_poly, barista_poly),
        Overlay(seating_poly, barista_poly),
        Reporter(event_csv="cafe_events.csv")
    ]
    
    # Create and run graph
    graph = Graph(filters)
    
    print("Starting Cafe Vision System...")
    print("Press 'q' to quit")
    
    try:
        frame_count = 0
        while graph.step():
            output = graph.last_output()
            if "frame" in output:
                cv2.imshow("Cafe Vision", output["frame"])
                frame_count += 1
                
                # Print progress every 30 frames
                if frame_count % 30 == 0:
                    metrics = output.get("metrics", {})
                    print(f"Frame {frame_count}: Customers: {metrics.get('customers_inside', 0)}, Coffees: {metrics.get('coffees_made', 0)}")
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("No frame in output, likely end of video")
                break
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        graph.close()
        cv2.destroyAllWindows()
        print(f"System closed. Processed {frame_count} frames.")

if __name__ == "__main__":
    main()