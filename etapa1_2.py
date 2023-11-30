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
    orange_image = cv2.imread('images/laranja.png',cv2.IMREAD_UNCHANGED)  # Replace 'orange.jpg' with the path to your orange image
    # Replacement image
    replacement_image=cv2.imread('images/basketball_alpha.png',cv2.IMREAD_UNCHANGED)
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
            f.doHands(mp_holistic, orange_image, t_start, frame, results)
            # só funciona uma hand de cada vez? WHY?
             # Draw left hand connections
            faceBox = f.detect_face (frame)
            if (faceBox):
                # Draw a circle at the face center
                # cv2.rectangle(frame, faceBox, (0, 255, 0), 2)
                # Paste the replacement image onto the original image
                # needs to check for boundaries.
                
                # treatedImage = cv2.resize(replacement_image, (faceBox[2], faceBox[3]))
                f.doMask(frame,faceBox,replacement_image)

            mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 22, 76), thickness=1, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 44, 250), thickness=1, circle_radius=1),
            )
            # Draw right hand connections
            mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 255, 66), thickness=1, circle_radius=3),
                mp_drawing.DrawingSpec(color=(245, 255, 230), thickness=1, circle_radius=1),
            )
            cv2.imshow("MediaPipe Holistic", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
