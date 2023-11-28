# Create conda environment and install dependencies
# conda create -n mediapipe python=3.11 -y & conda activate mediapipe & pip install opencv-python typing-extensions mediapipe

import cv2 as cv
import mediapipe as mp  # Extract landmarks #https://google.github.io/mediapipe/getting_started/python.html

from mp_holistic_v2 import draw_landmarks

# Camera source
source = "V"  # M: Mobile phone, W: Webcam, V: Video file
match source.casefold():
    case "m":
        ip_address = "10.20.49.185"
        cap = cv.VideoCapture(f"http://{ip_address}:4747/video/force/640x480")
    case "w":
        cap = cv.VideoCapture(0)
    case "v":
        cap = cv.VideoCapture("video/workout.webm")


mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

# Accessing landmark indexes using dot notation
pose_lmk = mp_holistic.PoseLandmark
# nose_index = pose_lmk.NOSE
# left_eye_inner_index = pose_lmk.LEFT_EYE_INNER
# left_eye_index = pose_lmk.LEFT_EYE
# ... and so on

# Invert the underlying dictionary of lmk, to also allow accessing names by index
inv_pose_lmk = {k: v for k, v in enumerate(pose_lmk._member_names_)} # {0: 'NOSE', 1: 'LEFT_EYE_INNER', ...}


# Extract Keypoint structures, if they exist
def extract_keypoints(results):
    norm_pose = results.pose_landmarks and results.pose_landmarks.landmark
    world_pose = results.pose_world_landmarks and results.pose_world_landmarks.landmark
    face = results.face_landmarks and results.face_landmarks.landmark
    left_hand = results.left_hand_landmarks and results.left_hand_landmarks.landmark
    right_hand = results.right_hand_landmarks and results.right_hand_landmarks.landmark

    return norm_pose, world_pose, face, left_hand, right_hand


def print_world_pose_landmarks(results):
    # pose, world_pose, face, lh, rh = extract_keypoints(results)
    world_pose = extract_keypoints(results)[1]

    if not world_pose:
        return

    # Accessing landmark object by name:
    # world_pose[pose_lmk.name].x, world_pose[pose_lmk.name].y, ...

    # Accessing landmark object by index:
    # world_pose[#landmark].x, world_pose[#landmark].y, ...

    # Accessing landmark name by index:
    # inv_pose_lmk[landmark#]

    for i, p in enumerate(world_pose):
        print(
            f"x:{p.x:.2f} y:{p.y:.2f} z:{p.z:.2f} visibility:{p.visibility:.2f} | Landmark {i}: {inv_pose_lmk[i]}"
        )
    # OR, if no landmark name is required
    # for p in world_pose:
    #    print(f"x:{p.x:.2f} y:{p.y:.2f} visibility:{p.visibility:.2f}")


mediapipe_detection = lambda image: holistic.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
# SAME AS:
# def mediapipe_detection(image):
#    return holistic.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))

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

        # Draw landmarks
        draw_landmarks(frame, results)

        # Print landmarks
        print_world_pose_landmarks(results)

        # Show to screen
        cv.imshow("MediaPipeResult", frame)

    cap.release()
    cv.destroyAllWindows()
