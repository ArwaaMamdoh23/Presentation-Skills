import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import time
import mediapipe as mp
from collections import Counter

# Load PoseNet model from TensorFlow Hub
posenet_model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
model = hub.load(posenet_model_url)
print("PoseNet model loaded!")

# Access the correct signature for inference
infer = model.signatures['serving_default']  # This is the inference function

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils  # Optional: for visualization

# Define body language meanings for postures and gestures
posture_meanings = {
    "Standing": "Confidence",
    "Sitting": "Comfort",
    "Slouching": "Lack of confidence",
    "Leaning Forward": "Interest",
    "Head Down": "Lack of confidence",
    "Leaning Back": "Relaxation",
    "Arms on Hips": "Confidence"
}

gesture_to_body_language = {
    "Open Palm 🖐️": "Honesty",
    "Closed Fist ✊": "Determination",
    "Pointing Finger ☝️": "Confidence",
    "Thumbs Up 👍": "Encouragement",
    "Thumbs Down 👎": "Criticism",
    "Victory Sign ✌️": "Peace",
    "OK Sign 👌": "Agreement",
    "Rock Sign 🤘": "Excitement",
    "Call Me 🤙": "Friendly"
}

# Function to classify posture based on keypoints
def classify_posture(keypoints):
    # Extract keypoints for important joints
    shoulder_y = keypoints[5]['y']
    hip_y = keypoints[11]['y']
    knee_y = keypoints[13]['y']
    ankle_y = keypoints[15]['y']
    head_y = keypoints[0]['y']

    if shoulder_y < hip_y:
        return "Standing"
    elif shoulder_y > hip_y:
        return "Sitting"
    if shoulder_y > hip_y and knee_y > shoulder_y and ankle_y > knee_y:
        return "Slouching"
    if head_y < shoulder_y and knee_y > hip_y:
        return "Leaning Forward"
    if head_y > shoulder_y:
        return "Head Down"
    if shoulder_y < hip_y and knee_y < hip_y and ankle_y < knee_y:
        return "Leaning Back"
    if abs(keypoints[7]['x'] - keypoints[8]['x']) < 0.1 and abs(keypoints[6]['x'] - keypoints[9]['x']) < 0.1:
        return "Arms on Hips"
    return "Unknown Posture"

# Function to detect gestures based on hand landmarks
def classify_hand_gesture(hand_landmarks):
    landmarks = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark])

    thumb_extended = landmarks[4][1] < landmarks[3][1]
    index_extended = landmarks[8][1] < landmarks[7][1]
    middle_extended = landmarks[12][1] < landmarks[11][1]
    ring_extended = landmarks[16][1] < landmarks[15][1]
    pinky_extended = landmarks[20][1] < landmarks[19][1]

    if all([index_extended, middle_extended, ring_extended, pinky_extended]) and not thumb_extended:
        return "Open Palm 🖐️"
    if not any([index_extended, middle_extended, ring_extended, pinky_extended, thumb_extended]):
        return "Closed Fist ✊"
    if index_extended and not any([middle_extended, ring_extended, pinky_extended]):
        return "Pointing Finger ☝️"
    if thumb_extended and not any([index_extended, middle_extended, ring_extended, pinky_extended]):
        return "Thumbs Up 👍"
    if not thumb_extended and not any([index_extended, middle_extended, ring_extended, pinky_extended]):
        return "Thumbs Down 👎"
    if index_extended and pinky_extended and not any([middle_extended, ring_extended]):
        return "Rock Sign 🤘"
    if thumb_extended and pinky_extended and not any([index_extended, middle_extended, ring_extended]):
        return "Call Me 🤙"
    return "Unknown Gesture"

# Function to extract keypoints from PoseNet output
def extract_keypoints(results):
    keypoints = []
    for i in range(17):  # PoseNet detects 17 keypoints
        x = results['output_0'][0][0][i][1].numpy()
        y = results['output_0'][0][0][i][0].numpy()
        confidence = results['output_0'][0][0][i][2].numpy()
        keypoints.append({"x": x, "y": y, "confidence": confidence})
    return keypoints

# Read input video file instead of webcam
# input_video_path = "D:/4th Year 1st Term/Graduation Project/Presentation-Skills/TasnimGesture.mp4"  # Change this to your video file path
# input_video_path = "D:/4th Year 1st Term/Graduation Project/Presentation-Skills/MayarGesture.mp4"  # Change this to your video file path
# input_video_path = "D:/4th Year 1st Term/Graduation Project/Presentation-Skills/TedTalk.mp4"  # Change this to your video file path
input_video_path = "D:/4th Year 1st Term/Graduation Project/Presentation-Skills/TedTalk2.mp4"  # Change this to your video file path
cap = cv2.VideoCapture(input_video_path)


paused = False  # Variable to track pause state
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (192, 192))
    rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    rgb_frame = np.expand_dims(rgb_frame, axis=0)
    rgb_frame = tf.convert_to_tensor(rgb_frame, dtype=tf.int32)
    
    results = infer(rgb_frame)  # Get the output from PoseNet
    keypoints = extract_keypoints(results)
    
    # Classify posture
    posture = classify_posture(keypoints)
    posture_meaning = posture_meanings.get(posture, "Unknown Meaning")
    
    # Process frame with MediaPipe Hands for gestures
    rgb_frame_for_gesture = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hands = mp_hands.process(rgb_frame_for_gesture)

    gesture = "Unknown Gesture"
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            gesture = classify_hand_gesture(hand_landmarks)
    
    gesture_meaning = gesture_to_body_language.get(gesture, "Unknown Meaning")

    # Combine posture and gesture meanings for overall body language
    overall_body_language = f"{posture_meaning}, {gesture_meaning}"

    # Display the detected posture, gesture, and combined body language
    # cv2.putText(frame, f"Posture: {posture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # cv2.putText(frame, f"Posture Meaning: {posture_meaning}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    # cv2.putText(frame, f"Gesture: {gesture}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # cv2.putText(frame, f"Gesture Meaning: {gesture_meaning}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"Overall Body Language: {overall_body_language}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Show the video feed with posture, gesture, and combined body language
    cv2.imshow("PoseNet Posture and Gesture Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord(' '):  
        paused = not paused
        while paused:
            key2 = cv2.waitKey(0) & 0xFF
            if key2 == ord(' '):  # Resume when SPACE is pressed again
                paused = False
                break
            elif key2 == ord('q'):  # Quit if 'q' is pressed while paused
                cap.release()
                cv2.destroyAllWindows()
                exit()

cap.release()
cv2.destroyAllWindows()
