# Create conda environment and install dependencies
# conda create -n mediapipe python=3.11 -y & conda activate mediapipe & pip install opencv-python typing-extensions mediapipe

# Adapted from https://google.github.io/mediapipe/solutions/holistic.html
import cv2 as cv2
import mediapipe as mp
import numpy as np
import functions as f


def main():
    mp_holistic = mp.solutions.holistic  # Holistic model
    mp_drawing = mp.solutions.drawing_utils  # Drawing utilities
    mp_drawing_styles = mp.solutions.drawing_styles

    # Camera source
    source = "w"  # M: Mobile phone, W: Webcam, V: Video file
    match source.casefold():
        case "m":
            ip_address = "10.10.10.75"
            cap = cv2.VideoCapture(f"http://{ip_address}:4747/video/force/640x480")
        case "w":
            cap = cv2.VideoCapture(1)
        case "v":
            cap = cv2.VideoCapture("video/workout.webm")

    # Load the image of the orange
    orange_image = cv2.imread(
        "images/laranja.png", cv2.IMREAD_UNCHANGED
    )  # Replace 'orange.jpg' with the path to your orange image
    # Replacement image
    replacement_image = cv2.imread("images/basketball_alpha.png", cv2.IMREAD_UNCHANGED)
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

            # blacken the media
            # frame = np.zeros_like(frame)
            f.doFaceV2(replacement_image, frame, results)
            f.doHands(mp_holistic, orange_image, frame, results)
            # TODO:
            # ok. aqui tem que ver mao a mao e retornar o xy de cada mão para o swipe
            # depois será preciso ver POSES para ver se se tem :
            # Um braço no ar;
            # Dois braços no ar;
            # isto já está tudo feito no holistic v4 do Professor.
            
            mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(255, 22, 76), thickness=1, circle_radius=3
                ),
                mp_drawing.DrawingSpec(
                    color=(255, 44, 250), thickness=1, circle_radius=1
                ),
            )
            # Draw FPS values
            f.draw_fps(t_start, frame)
            cv2.imshow("MediaPipe Holistic", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
