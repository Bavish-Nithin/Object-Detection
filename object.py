import cv2
import time
from ultralytics import YOLO
from dotenv import load_dotenv, dotenv_values

load_dotenv()
# --- Configuration ---
# Replace with your actual RTSP URL, or use 0 for a local webcam webcam
RTSP_URL = os.getenv("URL") 
TARGET_CLASS = 0  # 0 is 'person' in the COCO dataset. Change to 2 for 'car'.

# Define the virtual tripwire (Vertical line at x = 400 pixels)
LINE_X = 100

# --- Initialize YOLO Model & Video Capture ---
# 'yolov8n.pt' is the nano model, optimized for real-time tracking on CPUs
model = YOLO('yolov8n.pt') 
cap = cv2.VideoCapture(RTSP_URL)

# --- Tracking State Variables ---
track_history = {}  # Keeps track of the previous X coordinate: {track_id: prev_cx}
entry_times = {}    # Stores the timestamp when an object enters: {track_id: entry_timestamp}

entry_count = 0
exit_count = 0

print("Starting live stream tracking. Press 'q' to quit.")
print(f"Virtual boundary set at X = {LINE_X}\n")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video stream ended or failed to fetch frame.")
        break

    # Get frame dimensions
    height, width, _ = frame.shape

    # Run YOLO tracking with persistence (persist=True keeps IDs stable across frames)
    results = model.track(frame, persist=True, classes=[TARGET_CLASS], verbose=False)
    
    # Draw the virtual tripwire line (Blue line)
    cv2.line(frame, (LINE_X, 0), (LINE_X, height), (255, 0, 0), 3)
    cv2.putText(frame, "Boundary Line", (LINE_X + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Check if any objects were detected and tracked in this frame
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            
            # Calculate the centroid (center point) of the object
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Draw the bounding box, centroid, and unique tracking ID
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # --- Cross Detection Logic ---
            if track_id in track_history:
                prev_cx = track_history[track_id]

                # Case 1: Crossed from Left to Right -> ENTRY
                if prev_cx < LINE_X and cx >= LINE_X:
                    entry_count += 1
                    entry_times[track_id] = time.time()  # Record entry timestamp
                    
                    print(f"[ENTRY] ID {track_id} crossed the line. Total Entries: {entry_count}")

                # Case 2: Crossed from Right to Left -> EXIT
                elif prev_cx > LINE_X and cx <= LINE_X:
                    exit_count += 1
                    print(f"[EXIT] ID {track_id} crossed the line. Total Exits: {exit_count}")
                    
                    # Calculate duration if we have a matching entry record
                    if track_id in entry_times:
                        duration = time.time() - entry_times[track_id]
                        print(f"⏱️ ID {track_id} spent {duration:.2f} seconds inside the tracked zone.")
                        # Remove the record since the object has exited
                        del entry_times[track_id]
                    else:
                        print(f"⚠️ ID {track_id} exited without a recorded entry timestamp (started on the right side).")

            # Save the current position as the historical position for the next frame
            track_history[track_id] = cx

    # --- UI Overlay ---
    # Draw a clean HUD block to display the current counts
    cv2.rectangle(frame, (10, 10), (220, 100), (0, 0, 0), -1)  # Black background box
    cv2.putText(frame, f"Entries: {entry_count}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Exits: {exit_count}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Show the video feed
    cv2.imshow("Live Entry/Exit Analytics", frame)

    # Quit application if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("\nStream closed. Analytics complete.")