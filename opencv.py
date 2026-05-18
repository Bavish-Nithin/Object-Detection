import cv2

# Open the laptop webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Variables to store the baseline background frame
static_back = None

print("Motion detection started. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    motion = 0 # Flag to keep track of whether motion is happening

    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply Gaussian Blur to smooth out pixel noise
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # If it's the first frame, save it as our static baseline background
    if static_back is None:
        static_back = gray
        continue

    # Step 3: Find the Absolute Difference between the baseline and current frame
    diff_frame = cv2.absdiff(static_back, gray)

    # Step 4: Apply a Threshold to get a clean binary (black/white) image
    # If a pixel value change is greater than 30, change it to white (255)
    _, thresh_frame = cv2.threshold(diff_frame, 60, 255, cv2.THRESH_BINARY)
    
    # Dilate the white areas to fill in small holes inside the moving object
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # Step 5: Find the outlines (contours) of the moving white shapes
    contours, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Filter out tiny movements (like a insect or a curtain twitching)
        if cv2.contourArea(contour) < 10000:
            continue
        
        motion = 1 # Motion detected!

        M = cv2.moments(contour)

        # To prevent a divide-by-zero error (if a contour is impossibly small)
        if M["m00"] != 0:
            # Calculate the true X and Y center of mass
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        # Extract the box coordinates from the moving contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Draw a custom green rectangle around the movement zone
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(frame, "MOTION DETECTED", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the different stages of the computer vision pipeline
    cv2.imshow("1. Original Feed (With Boxes)", frame)
    cv2.imshow("2. Difference Map (absdiff)", diff_frame)
    cv2.imshow("3. Threshold Frame (Binary)", thresh_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    # Optional: Press 'r' to reset the background baseline frame manually
    elif key == ord('r'):
        static_back = None
        print("Background frame reset.")

cap.release()
cv2.destroyAllWindows()