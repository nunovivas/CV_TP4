import cv2
import cv2 as cv2
from statistics import mean as mean
import mediapipe as mp  # started using it here after the stage 2

import numpy as np
import imageFunctions as imgF
import math  # for the euclidean distance

# Store fps values for calculating mean
fps_list = []


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


def writeStringBottomLeftFrame(frame, text):
    height, _ = frame.shape[:2]
    position = (10, height - 20)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 50, 255), thickness=2, )


def writeStringBottomRightFrame(frame, text):
    height, width = frame.shape[:2]  # Fix: Extract height and width correctly
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)[0]

    position = (width - 10 - text_size[0], height - 20)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 50, 255), thickness=2)


def findCenterX(mp_holistic, frame, landmarks):
    palm_center_x = int(abs((landmarks[mp_holistic.HandLandmark.THUMB_MCP].x +
                             landmarks[mp_holistic.HandLandmark.INDEX_FINGER_MCP].x +
                             landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].x +
                             landmarks[mp_holistic.HandLandmark.RING_FINGER_MCP].x +
                             landmarks[mp_holistic.HandLandmark.PINKY_MCP].x) / 5 * frame.shape[1]))

    return int(palm_center_x)


def findCenterY(mp_holistic, frame, landmarks):
    palm_center_y = int(abs((landmarks[mp_holistic.HandLandmark.THUMB_MCP].y +
                             landmarks[mp_holistic.HandLandmark.INDEX_FINGER_MCP].y +
                             landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].y +
                             landmarks[mp_holistic.HandLandmark.RING_FINGER_MCP].y +
                             landmarks[mp_holistic.HandLandmark.PINKY_MCP].y) / 5 * frame.shape[0]))

    return int(palm_center_y)


def doFaceV2(replacement_image, frame, results):
    if results.face_landmarks:
        # Extract face landmarks
        face_landmarks = results.face_landmarks.landmark

        # Get bounding box coordinates around the face
        x_values = [landmark.x for landmark in face_landmarks]
        y_values = [landmark.y for landmark in face_landmarks]

        min_x, max_x = min(x_values), max(x_values)
        min_y, max_y = min(y_values), max(y_values)

        ih, iw, _ = frame.shape
        bbox = (
            int(min_x * iw),
            int(min_y * ih),
            int((max_x - min_x) * iw),
            int((max_y - min_y) * ih)
        )
        doMask(frame, bbox, replacement_image)
        return int(min_x * iw), int(min_y * ih)
def checkHandsAboveHead(handPos,headPos):
    if handPos and headPos:
        #print(f"HandPos:{handPos[0]} | HeadPos:{headPos[0]}")
        #  hand and head must be visible
        if handPos[0] > 0 and headPos[0] > 0 and (handPos[0] < headPos[0]):
            return True
        else:
            return False
    else:
        return False
def doHands(mp_holistic, orange_image, frame, results):
    # for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:

    for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
        if hand_landmarks:
            # here needs to test if the PALM is visible
            # Get the landmarks for the palm of the hand
            landmarks = hand_landmarks.landmark
            # Calculate the palm center as the average of the base of the fingers
            palm_center_x = findCenterX(mp_holistic, frame, landmarks)
            palm_center_y = findCenterY(mp_holistic, frame, landmarks)

            min_box_width = 20
            thumb_tip = (landmarks[mp_holistic.HandLandmark.THUMB_TIP].x * frame.shape[1],
                         landmarks[mp_holistic.HandLandmark.THUMB_TIP].y * frame.shape[0])
            pinky_tip = (landmarks[mp_holistic.HandLandmark.PINKY_TIP].x * frame.shape[1],
                         landmarks[mp_holistic.HandLandmark.PINKY_TIP].y * frame.shape[0])
            hand_width = int(math.dist(thumb_tip, pinky_tip))
            # Adjust box width to be 1/4 of the hand width
            box_width = int(hand_width / 4)
            box_height = int(box_width)
            if box_width > min_box_width:
                # Adjust box coordinates to ensure it stays within the valid range
                box = (
                    max(0, palm_center_x - box_width // 2),
                    max(0, palm_center_y - box_height // 2),
                    min(frame.shape[1], box_width),
                    min(frame.shape[0], box_height)
                )
                doMask(frame, box, orange_image)


def doLeftHand(mp_holistic, orange_image, frame, results):
    # for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:

    for hand_landmarks in [results.left_hand_landmarks]:
        if hand_landmarks:
            # here needs to test if the PALM is visible
            # Get the landmarks for the palm of the hand
            landmarks = hand_landmarks.landmark
            # Calculate the palm center as the average of the base of the fingers
            palm_center_x = findCenterX(mp_holistic, frame, landmarks)
            palm_center_y = findCenterY(mp_holistic, frame, landmarks)
            # Draw a circle at the center of the palm
            # cv2.circle(frame, (palm_center_x, palm_center_y), radius=10, color=(0, 255, 0), thickness=-1)
            # box_width = 200
            # box_height = 200
            min_box_width = 20
            # Adjust box width to be 1/4 of the hand width
            thumb_tip = (landmarks[mp_holistic.HandLandmark.THUMB_TIP].x * frame.shape[1],
                         landmarks[mp_holistic.HandLandmark.THUMB_TIP].y * frame.shape[0])
            pinky_tip = (landmarks[mp_holistic.HandLandmark.PINKY_TIP].x * frame.shape[1],
                         landmarks[mp_holistic.HandLandmark.PINKY_TIP].y * frame.shape[0])
            hand_width = int(math.dist(thumb_tip, pinky_tip))
            box_width = int(hand_width / 4)
            box_height = int(box_width)
            print(f"Palm center (x){palm_center_x} palm center (y){palm_center_y} Hand width={hand_width}")

            if (box_width > min_box_width):
                # Adjust box coordinates to ensure it stays within the valid range
                box = (
                    max(0, palm_center_x - box_width // 2),
                    max(0, palm_center_y - box_height // 2),
                    min(frame.shape[1], box_width),
                    min(frame.shape[0], box_height)
                )
                doMask(frame, box, orange_image)


def doMask(frame, box, newImage):
    # Resize the image to fit the bounding box
    newImage_resized = cv2.resize(newImage, (box[2], box[3]))
    # Print shapes for troubleshooting
    # print("Image resized shape:", newImage_resized.shape)
    # print("Box:", box)
    # print("Frame shape:", frame.shape[0], frame.shape[1])

    # check if it has alpha channel
    if (not has_alpha_opencv(newImage)):
        print("No alpha channel on the image")
        return 0
    # Validate the box coordinates
    if (
            0 <= box[0] < frame.shape[1] and
            0 <= box[1] < frame.shape[0] and
            0 <= box[0] + box[2] <= frame.shape[1] and
            0 <= box[1] + box[3] <= frame.shape[0]
    ):
        # Create a mask based on the alpha channel (transparency)
        alpha_channel = newImage_resized[:, :, 3] / 255.0

        # Extract the region of interest from the frame
        roi = frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]

        # Blend the images using the alpha channel
        for c in range(0, 3):
            roi[:, :, c] = (alpha_channel * newImage_resized[:, :, c] +
                            (1.0 - alpha_channel) * roi[:, :, c])

        # Update the frame with the blended result
        frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]] = roi
    else:
        print("Invalid box size")


def has_alpha_opencv(img):
    return img.shape[-1] == 4


def doFace(image, replacement_image, results):
    # Initialize Mediapipe Face Detection
    face_detection = mp.solutions.face_detection
    face_detection_module = face_detection.FaceDetection(
        min_detection_confidence=0.2)  # , device=0 uses the GPU if available

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run face detection
    results = face_detection_module.process(image_rgb)

    # Check if faces are detected
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                int(bboxC.width * iw), int(bboxC.height * ih)

            # # Calculate face center
            # center_x = bbox[0] + bbox[2] // 2
            # center_y = bbox[1] + bbox[3] // 2

        # Return the center coordinates
        # return center_x, center_y
        face_detection_module.close()  # try to prevent the error
        # if it has a box then do the mask
        if (bbox):
            # Draw a circle at the face center
            # cv2.rectangle(frame, faceBox, (0, 255, 0), 2)
            # Paste the replacement image onto the original image
            # needs to check for boundaries.

            # treatedImage = cv2.resize(replacement_image, (faceBox[2], faceBox[3]))
            doMask(image, bbox, replacement_image)
    else:
        print("No faces detected.")
        face_detection_module.close()
        return None
