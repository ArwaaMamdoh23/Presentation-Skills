import cv2
import mediapipe as mp
import numpy as np
import time
from collections import Counter

mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils  # Optional: for visualization

gesture_counter = Counter()
last_counted_frame = {}

FRAME_INTERVAL = 60
current_frame = 0  

gesture_to_body_language = {
    "Open Palm": "Honesty",
    "Closed Fist": "Determination",
    "Pointing Finger": "Confidence",
    "Thumbs Up": "Encouragement",
    "Thumbs Down": "Criticism",
    "Victory Sign": "Peace",
    "OK Sign": "Agreement",
    "Rock Sign": "Excitement",
    "Call Me": "Friendly"
}

gesture_counter = {
    "Open Palm": 0,
    "Closed Fist": 0,
    "Pointing Finger": 0,
    "Thumbs Up": 0,
    "Thumbs Down": 0,
    "Victory Sign": 0,
    "OK Sign": 0,
    "Rock Sign": 0,
    "Call Me": 0
}

def classify_hand_gesture(hand_landmarks):

    landmarks = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark])

    THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 4, 8, 12, 16, 20
    THUMB_IP, INDEX_DIP, MIDDLE_DIP, RING_DIP, PINKY_DIP = 3, 7, 11, 15, 19
    THUMB_MCP = 2

    def is_finger_extended(tip, dip):
        return landmarks[tip][1] < landmarks[dip][1] 

    
    thumb_extended = landmarks[THUMB_TIP][1] < landmarks[THUMB_IP][1]
    index_extended = is_finger_extended(INDEX_TIP, INDEX_DIP)
    middle_extended = is_finger_extended(MIDDLE_TIP, MIDDLE_DIP)
    ring_extended = is_finger_extended(RING_TIP, RING_DIP)
    pinky_extended = is_finger_extended(PINKY_TIP, PINKY_DIP)

    if all([index_extended, middle_extended, ring_extended, pinky_extended]) and not thumb_extended:
        return "Open Palm"

    if not any([index_extended, middle_extended, ring_extended, pinky_extended, thumb_extended]):
        return "Closed Fist"

    if index_extended and not any([middle_extended, ring_extended, pinky_extended]):
        return "Pointing Finger"

    if thumb_extended and not any([index_extended, middle_extended, ring_extended, pinky_extended]):
        return "Thumbs Up"
    
    if not thumb_extended and not any([index_extended, middle_extended, ring_extended, pinky_extended]):
        return "Thumbs Down"

    if not thumb_extended and all([index_extended, middle_extended]) and not any([ring_extended, pinky_extended]):
        return "Victory Sign"

    if thumb_extended and index_extended and not any([middle_extended, ring_extended, pinky_extended]):
        return "OK Sign"

    if index_extended and pinky_extended and not any([middle_extended, ring_extended]):
        return "Rock Sign"

    if thumb_extended and pinky_extended and not any([index_extended, middle_extended, ring_extended]):
        return "Call Me"

    return "Unknown Gesture"

use_webcam = False 
if use_webcam:
    cap = cv2.VideoCapture(0)  
    output_video_path = "webcam_output.mp4"
else:
    input_video_path = "D:/4th Year 1st Term/Graduation Project/Presentation-Skills/Videos/TedTalk.mp4"  
    cap = cv2.VideoCapture(input_video_path)


paused = False  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    current_frame += 1  
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = mp_hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            gesture_name = classify_hand_gesture(hand_landmarks)
            if gesture_name == "Unknown Gesture":
                continue  
            if gesture_name not in last_counted_frame or (current_frame - last_counted_frame[gesture_name]) >= FRAME_INTERVAL:
                gesture_counter[gesture_name] += 1  
                last_counted_frame[gesture_name] = current_frame  

            body_language_meaning = gesture_to_body_language.get(gesture_name, "Unknown Meaning")

            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            gesture_display = gesture_name if gesture_name in gesture_to_body_language else "Unknown Gesture"
            body_language_display = body_language_meaning if body_language_meaning != "Unknown Meaning" else "No Interpretation"
            cv2.putText(frame, f"Gesture: {gesture_name}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(frame, f"Body: {body_language_meaning}", (x_min, y_max + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)



    cv2.imshow("Hand Gesture & Body Language Recognition", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord(' '):  
        paused = not paused  
        while paused:
            key2 = cv2.waitKey(0) & 0xFF  
            if key2 == ord(' '):  
                paused = False
                break
            elif key2 == ord('q'):  
                cap.release()
                cv2.destroyAllWindows()
                exit()   


print("\nGesture Summary:")
for gesture, count in gesture_counter.items():
    meaning = gesture_to_body_language.get(gesture, "Unknown Meaning")
    print(f"{gesture}: {count} times - Meaning: {meaning}")

    
cap.release()
cv2.destroyAllWindows()

