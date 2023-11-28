# Create conda environment and install dependencies
# conda create -n mediapipe python=3.11 -y & conda activate mediapipe & pip install opencv-python typing-extensions mediapipe

# Adapted from https://google.github.io/mediapipe/solutions/holistic.html
import cv2 as cv
import mediapipe as mp
import numpy as np

from statistics import mean as mean


mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities
mp_drawing_styles = mp.solutions.drawing_styles

# Camera source
source = "w"  # M: Mobile phone, W: Webcam, V: Video file
match source.casefold():
    case "m":
        ip_address = "10.20.49.185"
        cap = cv.VideoCapture(f"http://{ip_address}:4747/video/force/640x480")
    case "w":
        cap = cv.VideoCapture(1)
    case "v":
        cap = cv.VideoCapture("video/workout.webm")


def draw_fps(t_start, frame):
    t_end = cv.getTickCount()
    fps = cv.getTickFrequency() / (t_end - t_start)
    fps_list.append(fps)
    if len(fps_list) > 10:
        del fps_list[0]

    cv.putText(
        frame,
        f"FPS: {fps:.1f} AVG10s: {mean(fps_list):.1f}",
        (0, 25),
        cv.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(0, 128, 26),
        thickness=1,
        lineType=cv.LINE_AA,
    )


# Store fps values for calculating mean
fps_list = []

with mp_holistic.Holistic() as holistic:
    
    while cv.pollKey() == -1:
        t_start = cv.getTickCount()

        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Rotate frame if needed (e.g. for mobile phone streams)
        if cap.get(cv.CAP_PROP_FRAME_COUNT) < -1.0:
            frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)

        results = holistic.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

        #blacken the media
        frame = np.zeros_like(frame)

        # Draw landmark annotation on the image.
        mp_drawing.dra
        mp_drawing.draw_landmarks(
            frame,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
        )
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )

        # Draw FPS values
        draw_fps(t_start, frame)
        cv.imshow("MediaPipe Holistic", frame)

cap.release()
cv.destroyAllWindows()
