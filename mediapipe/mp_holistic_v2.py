# Create conda environment and install dependencies
# conda create -n mediapipe python=3.11 -y & conda activate mediapipe & pip install opencv-python typing-extensions mediapipe

import cv2 as cv
import mediapipe as mp

if __name__ == "__main__":
    # Camera source
    source = "m"  # M: Mobile phone, W: Webcam, V: Video file
    match source.casefold():
        case "m":
            ip_address = "10.10.10.75"
            cap = cv.VideoCapture(f"http://{ip_address}:4747/video/force/640x480")
        case "w":
            cap = cv.VideoCapture(0)
        case "v":
            cap = cv.VideoCapture("video/workout.webm")

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def draw_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
    )
    # Draw pose connections
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 255), thickness=1, circle_radius=3),
        mp_drawing.DrawingSpec(color=(80, 44, 255), thickness=1, circle_radius=1),
    )
    # Draw left hand connections
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(255, 22, 76), thickness=1, circle_radius=3),
        mp_drawing.DrawingSpec(color=(255, 44, 250), thickness=1, circle_radius=1),
    )
    # Draw right hand connections
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 255, 66), thickness=1, circle_radius=3),
        mp_drawing.DrawingSpec(color=(245, 255, 230), thickness=1, circle_radius=1),
    )


mediapipe_detection = lambda image: holistic.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
# SAME AS:
# def mediapipe_detection(image):
#    return holistic.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))

if __name__ == "__main__":
    # Set mediapipe model
    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        while cv.pollKey() == -1:
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Rotate frame if needed (e.g. for mobile phone streams)
            if cap.get(cv.CAP_PROP_FRAME_COUNT) < -1.0:
                frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)

            # Make detections
            results = mediapipe_detection(frame)
            # print(results.pose_landmarks.landmark) #results.face_landmarks; print(results.left_hand_landmarks)

            draw_landmarks(frame, results)

            cv.imshow("MediaPipe Result", frame)

        cap.release()
        cv.destroyAllWindows()
