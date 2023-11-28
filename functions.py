import cv2 as cv2
from statistics import mean as mean

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


def overlay_transparent(background, overlay):
    # Split the overlay image into channels
    overlay_image = overlay[:, :, :3]  # RGB channels
    overlay_mask = overlay[:, :, 3:]  # Alpha channel

    # Invert the alpha mask
    background_mask = 255 - overlay_mask

    # Extract the region of interest from the background
    background_region = cv2.bitwise_and(background, background, mask=background_mask)

    # Extract the region of interest from the overlay
    overlay_region = cv2.bitwise_and(overlay_image, overlay_image, mask=overlay_mask)

    # Combine the two regions
    combined = cv2.add(background_region, overlay_region)

    # Update the background with the combined region
    background[y_min:y_max, x_min:x_max] = combined

    return background


def findCenterX(mp_holistic, frame, landmarks):
    palm_center_x = int((landmarks[mp_holistic.HandLandmark.THUMB_MCP].x +
                                    landmarks[mp_holistic.HandLandmark.INDEX_FINGER_MCP].x +
                                    landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].x +
                                    landmarks[mp_holistic.HandLandmark.RING_FINGER_MCP].x +
                                    landmarks[mp_holistic.HandLandmark.PINKY_MCP].x) / 5 * frame.shape[1])
                        
    return int(palm_center_x)

def findCenterY(mp_holistic, frame, landmarks):
    palm_center_y = int((landmarks[mp_holistic.HandLandmark.THUMB_MCP].y +
                                    landmarks[mp_holistic.HandLandmark.INDEX_FINGER_MCP].y +
                                    landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].y +
                                    landmarks[mp_holistic.HandLandmark.RING_FINGER_MCP].y +
                                    landmarks[mp_holistic.HandLandmark.PINKY_MCP].y) / 5 * frame.shape[0])
                        
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
                # Resize the orange image to fit the bounding box
                orange_resized = cv2.resize(orange_image, (box[2], box[3]))
                # Print shapes for troubleshooting
                print("Orange image  resized shape:", orange_resized.shape)
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
                    frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]] = orange_resized
                else :
                    print ("Invalid box size")

        # Draw FPS values
        draw_fps(t_start, frame)
