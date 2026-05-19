import os
import cv2
import time
import numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()
# --- Configuration ---
RTSP_URL = os.getenv("URL") 
TARGET_CLASS = 67  # 0 is 'person' in the COCO dataset. Change to 2 for 'car'.

# Define the Virtual Zone (A polygon/rectangle in the center of the frame)
# Format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
# You can adjust these coordinates based on your camera's resolution.
ZONE_PTS = np.array([[100, 100], [1820, 100], [1820, 980], [100, 980]], np.int32)
ZONE_PTS = ZONE_PTS.reshape((-1, 1, 2)) # Reshape for OpenCV drawing functions

# --- Initialize YOLO Model & Video Capture ---
model = YOLO('yolov8n.pt') 
# Use 0 for webcam or replace with RTSP_URL if testing live IP stream
cap = cv2.VideoCapture(0)

cv2.namedWindow("Omni-Directional Analytics", cv2.WINDOW_NORMAL)

# --- Tracking State Variables ---
# Now keeps track of whether the object was inside or outside: {track_id: is_inside (Boolean)}
track_history = {}  
entry_times = {}    

entry_count = 0
exit_count = 0

print("Starting omni-directional zone tracking. Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video stream ended or failed to fetch frame.")
        break

    # Run YOLO tracking
    results = model.track(frame, persist=True, classes=[TARGET_CLASS], verbose=False)
    
    # Draw the virtual zone (Blue Polygon)
    cv2.polylines(frame, [ZONE_PTS], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.putText(frame, "Tracking Zone", (160, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Check if any objects were detected
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            
            # Calculate the centroid (center point) of the object
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Draw the bounding box and centroid
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # --- Zone Detection Logic ---
            # pointPolygonTest returns >= 0 if the point is inside or on the polygon, -1 if outside
            is_inside_now = cv2.pointPolygonTest(ZONE_PTS, (cx, cy), False) >= 0

            # If we have seen this object in a previous frame, check for state changes
            if track_id in track_history:
                was_inside_before = track_history[track_id]

                # Case 1: Was outside, is now inside -> ENTRY
                if not was_inside_before and is_inside_now:
                    entry_count += 1
                    entry_times[track_id] = time.time()
                    print(f"[ENTRY] ID {track_id} entered the zone. Total Entries: {entry_count}")

                # Case 2: Was inside, is now outside -> EXIT
                elif was_inside_before and not is_inside_now:
                    exit_count += 1
                    print(f"[EXIT] ID {track_id} left the zone. Total Exits: {exit_count}")
                    
                    # Calculate duration
                    if track_id in entry_times:
                        duration = time.time() - entry_times[track_id]
                        print(f"⏱️ ID {track_id} spent {duration:.2f} seconds inside.")
                        del entry_times[track_id]
                    else:
                        print(f"⚠️ ID {track_id} exited without a recorded entry timestamp.")

            # Save the current state (True/False) for the next frame's comparison
            track_history[track_id] = is_inside_now

    # --- UI Overlay ---
    
    # 1. Create a copy of the frame for the overlay
    overlay = frame.copy()
    
    # 2. Draw a smaller black rectangle on the overlay
    cv2.rectangle(overlay, (10, 10), (160, 75), (0, 0, 0), -1)
    
    # 3. Blend the overlay with the original frame (0.4 means 40% opacity)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # 4. Draw the text slightly smaller (0.6 scale) on top of the blended frame
    cv2.putText(frame, f"Entries: {entry_count}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Exits: {exit_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Show the resizable video feed
    cv2.imshow("Omni-Directional Analytics", frame)

    # Quit application if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("\nStream closed. Analytics complete.")