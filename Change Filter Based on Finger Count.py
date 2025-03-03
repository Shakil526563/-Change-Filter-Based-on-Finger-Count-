import cv2  # Import OpenCV for computer vision functions
import sys  # Import sys for command-line argument handling
import numpy as np  # Import NumPy for numerical operations
import mediapipe as mp  # Import MediaPipe for hand detection

# Filter modes
PREVIEW  = 0  # Default preview mode (no filter)
BLUR     = 1  # Apply a blurring effect
FEATURES = 2  # Detect and highlight corner features
CANNY    = 3  # Apply Canny edge detection
SEPIA    = 4  # Apply a sepia effect
GRAY     = 5  # Convert to grayscale
INVERT   = 6  # Invert colors
SKETCH   = 7  # Apply a pencil sketch effect
EMBOSS   = 8  # Apply an emboss effect
SHARPEN  = 9  # Sharpen the image

# Parameters for corner feature detection (goodFeaturesToTrack)
feature_params = dict(maxCorners=500, qualityLevel=0.2, minDistance=15, blockSize=9)

# Set default video source to 0 (default webcam)
s = 0  # Default source
if len(sys.argv) > 1:  # If a command-line argument is given, use it as the video source
    s = sys.argv[1]

image_filter = PREVIEW  # Start with preview mode (no filter applied)
alive = True  # Control loop execution

win_name = "Camera Filters"  # Window name
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)  # Create a resizable window
result = None  # Placeholder for the processed frame

# Open video capture from the specified source
source = cv2.VideoCapture(s)

# Initialize MediaPipe hands for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.7)

# Main loop to process video frames
while alive:
    has_frame, frame = source.read()  # Read a frame from the video source
    if not has_frame:  # If no frame is captured, exit loop
        break

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally (mirror effect)

    # Convert the frame to RGB for MediaPipe hand tracking
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # If hands are detected, count the number of fingers
    num_fingers = 0
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Count fingers based on landmarks positions
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # Count the number of extended fingers (tips are above the middle part of the hand)
            if index_tip.y < thumb_tip.y: num_fingers += 1
            if middle_tip.y < thumb_tip.y: num_fingers += 1
            if ring_tip.y < thumb_tip.y: num_fingers += 1
            if pinky_tip.y < thumb_tip.y: num_fingers += 1

    # Change filter based on the number of fingers
    if num_fingers == 1:
        image_filter = BLUR  # Change to blur filter
    elif num_fingers == 2:
        image_filter = INVERT  # Change to invert filter
    elif num_fingers == 3:
        image_filter = SEPIA  # Change to sepia filter
    elif num_fingers == 4:
        image_filter = GRAY  # Change to grayscale filter
    else:
        image_filter = PREVIEW  # Default to preview mode (no filter applied)

    # Apply the selected filter
    if image_filter == PREVIEW:
        result = frame  # No filter, show the original frame
    elif image_filter == CANNY:
        result = cv2.Canny(frame, 80, 150)  # Apply Canny edge detection (thresholds: 80, 150)
    elif image_filter == BLUR:
        result = cv2.blur(frame, (13, 13))  # Apply a blur filter with a 13x13 kernel
    elif image_filter == FEATURES:
        result = frame.copy()  # Copy the frame to draw detected features
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        corners = cv2.goodFeaturesToTrack(frame_gray, **feature_params)  # Detect corners
        if corners is not None:
            for x, y in np.float32(corners).reshape(-1, 2):  # Iterate over detected points
                cv2.circle(result, (int(x), int(y)), 10, (0, 255, 0), 1)  # Draw circles on detected corners
    elif image_filter == SEPIA:
        # Sepia filter transformation matrix
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        result = cv2.transform(frame, kernel)  # Apply the sepia filter
    elif image_filter == GRAY:
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    elif image_filter == INVERT:
        result = cv2.bitwise_not(frame)  # Invert the colors
    elif image_filter == SKETCH:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        result = cv2.Canny(gray_frame, 50, 200)  # Apply Canny edge detection for a sketch effect
    elif image_filter == EMBOSS:
        kernel = np.array([[ -2, -1,  0],
                           [ -1,  1,  1],
                           [  0,  1,  2]])
        result = cv2.filter2D(frame, -1, kernel)  # Apply emboss filter
    elif image_filter == SHARPEN:
        kernel = np.array([[ 0, -1,  0],
                           [-1,  5, -1],
                           [ 0, -1,  0]])
        result = cv2.filter2D(frame, -1, kernel)  # Apply sharpening filter

    # Display the processed frame
    cv2.imshow(win_name, result)

    # Handle keyboard input for quitting
    key = cv2.waitKey(1)
    if key == ord("Q") or key == ord("q") or key == 27:  # Quit on 'Q', 'q', or 'Esc'
        alive = False

# Release video source and close window
source.release()  # Stop capturing video
cv2.destroyWindow(win_name)  # Close the display window
