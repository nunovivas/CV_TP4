import cv2 as cv2
from statistics import mean as mean
import mediapipe as mp # started using it here after the stage 2

import numpy as np
import imageFunctions as imgF
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

def doHands(mp_holistic, orange_image, t_start, frame, results):
    #for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
    for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
        if hand_landmarks:
            #here needs to test if the PALM is visible
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
            hand_width = max(landmarks[mp_holistic.HandLandmark.THUMB_TIP].x * frame.shape[1] - landmarks[mp_holistic.HandLandmark.PINKY_TIP].x * frame.shape[1], 1)
            box_width = int(hand_width / 4)
            box_height = int(box_width)
            if (box_width>min_box_width):
            # Adjust box coordinates to ensure it stays within the valid range
                box = (
                            max(0, palm_center_x - box_width // 2),
                            max(0, palm_center_y - box_height // 2),
                            min(frame.shape[1], box_width),
                            min(frame.shape[0], box_height)
                        )
                doMask(frame,box,orange_image)
               

                

        # Draw FPS values
        draw_fps(t_start, frame)

def doMask (frame, box, newImage):
    # Resize the orange image to fit the bounding box
    newImage_resized = cv2.resize(newImage, (box[2], box[3]))
    # Print shapes for troubleshooting
    print("Orange image  resized shape:", newImage_resized.shape)
    print("Box:", box)
    print("Frame shape:", frame.shape[0], frame.shape[1])
    #Validate the box coordinates
    if (
                0 <= box[0] < frame.shape[1] and
                0 <= box[1] < frame.shape[0] and
                0 <= box[0] + box[2] <= frame.shape[1] and
                0 <= box[1] + box[3] <= frame.shape[0]
            ):
        # Replace the region inside the rectangle with the orange image
        #orange_resized= imgF.make_transparent(orange_resized)
        #frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]] = orange_resized
        # Create a mask based on the white background
        # in the 4th channel.
        mask = np.all(newImage_resized[:, :, :3] == [255, 255, 255], axis=-1)
        # Invert the mask
        inverse_mask = ~mask
        # Create an alpha channel by converting the inverse mask to float (1 for white, 0 for non-white)
        alpha_channel = inverse_mask.astype(float)

        # Extract the region of interest from the frame
        roi = frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]

        # Blend the images using the alpha channel
        for c in range(0, 3):
            roi[:, :, c] = (alpha_channel * newImage_resized[:, :, c] +
                            (1.0 - alpha_channel) * roi[:, :, c])

        # Update the frame with the blended result
        frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]] = roi
    else :
        print ("Invalid box size")

def detect_face(image):
    
    # Initialize Mediapipe Face Detection
    face_detection = mp.solutions.face_detection
    face_detection_module = face_detection.FaceDetection(min_detection_confidence=0.2) #, device=0 uses the GPU if available

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
        return bbox
    else:
        print("No faces detected.")
        return None