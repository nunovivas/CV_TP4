# Create conda environment and install dependencies
# conda create -n mediapipe python=3.11 -y & conda activate mediapipe & pip install opencv-python typing-extensions mediapipe

# Adapted from https://google.github.io/mediapipe/solutions/holistic.html
import cv2 as cv2
import mediapipe as mp
import numpy as np

from statistics import mean as mean


mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities
mp_drawing_styles = mp.solutions.drawing_styles

# Camera source
source = "m"  # M: Mobile phone, W: Webcam, V: Video file
match source.casefold():
    case "m":
        ip_address = "10.10.10.75"
        cap = cv2.VideoCapture(f"http://{ip_address}:4747/video/force/640x480")
    case "w":
        cap = cv2.VideoCapture(1)
    case "v":
        cap = cv2.VideoCapture("video/workout.webm")


def draw_fps(t_start, frame):
    t_end = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (t_end - t_start)
    fps_list.append(fps)
    if len(fps_list) > 10:
        del fps_list[0]

    cv2.putText(
        frame,
        f"FPS: {fps:.1f} AVG10s: {mean(fps_list):.1f}",
        (0, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(0, 128, 26),
        thickness=1,
        lineType=cv2.LINE_AA,
    )


# Store fps values for calculating mean
fps_list = []
# Load the image of the orange
orange_image = cv2.imread('images/laranja.jpg')  # Replace 'orange.jpg' with the path to your orange image
# Check if the orange image is loaded successfully
if orange_image is None:
    print("Error: Unable to load the orange image.")
    exit()

with mp_holistic.Holistic() as holistic:
    
    while cv2.pollKey() == -1:
        t_start = cv2.getTickCount()

        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Rotate frame if needed (e.g. for mobile phone streams)
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) < -1.0:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        #blacken the media
        #frame = np.zeros_like(frame)
        for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand_landmarks:
                # Get the landmarks for the palm of the hand
                landmarks = hand_landmarks.landmark
                palm_center_x = int(landmarks[mp_holistic.HandLandmark.WRIST].x * frame.shape[1])
                palm_center_y = int(landmarks[mp_holistic.HandLandmark.WRIST].y * frame.shape[0])

                # Calculate the bounding box coordinates for the hand
                x_min = int(min(landmark.x * frame.shape[1] for landmark in landmarks))
                y_min = int(min(landmark.y * frame.shape[0] for landmark in landmarks))
                x_max = int(max(landmark.x * frame.shape[1] for landmark in landmarks))
                y_max = int(max(landmark.y * frame.shape[0] for landmark in landmarks))

                # Calculate the new center based on the palm of the hand
                new_center_x = palm_center_x - (x_max - x_min) // 2
                new_center_y = palm_center_y - (y_max - y_min) // 2

                # Create a circular mask for the orange image
                mask = np.zeros_like(orange_image)
                cv2.circle(mask, ((x_max - x_min) // 2, (y_max - y_min) // 2), min((x_max - x_min) // 2, (y_max - y_min) // 2), (255, 255, 255), thickness=cv2.FILLED)

                # Extract the circular region from the orange image
                orange_circular = cv2.bitwise_and(orange_image, mask)

                # Resize the circular orange image to fit the bounding box
                orange_circular = cv2.resize(orange_circular, (x_max - x_min, y_max - y_min))

                # Replace the circular orange image in the original frame
                frame[new_center_y:new_center_y + (y_max - y_min), new_center_x:new_center_x + (x_max - x_min), :3] = orange_circular


            # Draw FPS values
            draw_fps(t_start, frame)
            cv2.imshow("MediaPipe Holistic", frame)

cap.release()
cv2.destroyAllWindows()
